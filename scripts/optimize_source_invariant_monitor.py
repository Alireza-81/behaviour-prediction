"""Search a shared checkpoint monitor design while fitting each bank on one architecture."""
import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import (
    AlertRule, GEOMETRY, activation_features, matrix, score_prefix,
)
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, PRIORITY, TRAITS, evidence, smooth
from behaviour_prediction.olmo_pretraining.protocol import read_json, write_json
from compare_single_source_trajectories import Cohort, NEVER, STEPS, classify, counts, fit, serial

ARCHITECTURES = ("llama", "olmo", "qwen", "mistral")
OUTPUT = Path("outputs/source-invariant-monitor-v1")
METHODS = ("ridge", "paired_ridge", "lda", "logistic", "neighbors", "kernel", "reservoir", "pair_difference")
CROSS_PAIRS = [(s, t) for s in ARCHITECTURES for t in ARCHITECTURES if s != t]
SEEDS = (101, 202)


@dataclass(frozen=True)
class Candidate:
    features: str
    learner: str
    rule: AlertRule


def directional(groups, layers, traits):
    return tuple(f"d/{group}/L{layer}/{trait}" for group, layer, trait in product(groups, layers, traits))


def geometric(groups, layers):
    return tuple(f"g/{group}/L{layer}/{metric}" for group, layer in product(groups, layers)
                 for metric in (GEOMETRY[:1] if group.startswith("angle") else GEOMETRY))


def catalog():
    features = {}
    layers_mid = (14, 15)
    direction_groups = {
        "raw_projection": ("projection",), "signed_cosine": ("cosine",),
        "first_checkpoint": ("reset5",), "direction_reset": ("reset10",),
        "direction_reset_wide": ("reset10_wide",), "direction_reset_by_norm": ("reset10_scale",),
        "direction_reset_unscaled": ("reset10_raw",), "motion": ("velocity", "acceleration"),
        "normalized_motion": ("velocity_norm", "ratio_velocity"),
        "balanced_direction": ("relative_direction", "vector_ratio"),
        "projection_change": ("projection_reset",),
        "direction_history": ("reset5", "reset10", "velocity", "ratio_velocity"),
        "projection_history": ("projection", "projection_reset", "ratio_velocity"),
    }
    for name, group in direction_groups.items():
        for axes, scope in ((PRIORITY, "priority"), (TRAITS, "all"),
                             (("honesty", "helpfulness"), "behavior")):
            features[f"{name}_{scope}"] = directional(group, LAYERS if scope == "all" else layers_mid, axes)
    for name, layers in (("late", (31,)), ("middle", layers_mid), ("layer15", (15,))):
        features[f"reset_{name}"] = directional(("reset10",), layers, PRIORITY)
        features[f"history_{name}"] = directional(("reset10", "velocity", "reset5"), layers, PRIORITY)
    for trait in TRAITS:
        features["single_" + trait] = directional(("reset10", "velocity_norm"), LAYERS, (trait,))
    geometry = {
        "raw": ("raw",), "growth": ("first_ratio", "second_ratio", "rate", "acceleration"),
        "cross_layer": ("balance", "balance_first", "balance_rate", "balance_reset"),
        "angular_shape": ("angle", "angle_first", "angle_rate"),
        "relative": ("first_ratio", "second_ratio", "rate", "balance_first", "balance_reset"),
    }
    for name, group in geometry.items():
        for scope, layers in (("last", (31,)), ("all", LAYERS)):
            features[f"geometry_{name}_{scope}"] = geometric(group, layers)
    features["joint_rate"] = features["direction_history_priority"] + geometric(("rate", "balance_rate"), LAYERS)
    features["joint_relative"] = features["direction_reset_priority"] + features["geometry_relative_all"]
    features["joint_angular"] = features["direction_reset_unscaled_priority"] + features["geometry_angular_shape_all"]
    features["joint_sizes"] = features["direction_history_priority"] + features["geometry_raw_last"]
    return features


def cutoffs(values, rule, steps):
    keep = steps >= rule.earliest
    clean = values[:, keep]
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Empty calibration set")
    if rule.calibration == "all":
        peak, median = clean.max(), np.median(clean)
        return float(peak + rule.margin * (peak - median))
    # Use the held-out source controls at the same observed prefix length.
    limit = np.full(len(steps), float(clean.max()))
    per_step = values[:, keep]
    populated = np.isfinite(per_step).all(0)
    levels = per_step[:, populated]
    peak = levels.max(0)
    median = np.median(levels, axis=0)
    # A global spread stabilizes the calibration when a corpus has only two runs.
    limit[np.flatnonzero(keep)[populated]] = peak + rule.margin * np.maximum(peak - median, .25 * np.ptp(clean))
    return limit


def rules():
    for mode in ("instant", "mean2", "persistent"):
        for earliest in (10, 15):
            for calibration, margin in product(("all", "checkpoint"), (0., .3, .8)):
                for quorum in (.51, .8):
                    yield AlertRule(mode, earliest, calibration, margin, quorum)


def fit_member(method, values, train, data):
    if method != "pair_difference":
        return fit(method, values, train, data)
    # Fit the weight vector to within-pair differences at the same checkpoint.
    # A run is scored on its own at inference, using the source-fit weights.
    labels = data.labels[train]
    y = labels == 1
    control = data.partner[train]
    same = y & (data.labels[control] == 0)
    raw, negatives = values[train][same], values[control][same]
    if not len(raw):
        raise ValueError("No matched eligible source windows")
    scale = np.concatenate((raw, negatives)).std(0).clip(1e-6)
    z = (raw - negatives) / scale
    gram = z.T @ z / len(z)
    weights = np.linalg.solve(gram + .75 * np.eye(len(scale)), z.mean(0))
    return dict(method=method, center=negatives.mean(0), scale=scale, coef=weights)


class Study:
    def __init__(self, data, seeds=SEEDS):
        self.data = data
        self.archive = [activation_features(p) for p in data.points]
        self.arch_index = {arch: i for i, arch in enumerate(ARCHITECTURES)}
        self.clean = np.array([o["earliest_possible"] is None for o in data.outcomes])
        self.seed = np.array([int(n.split("/")[2]) in seeds for n in data.names])
        self.train_groups = {}
        for source in ARCHITECTURES:
            eligible = self.seed & (source == data.arch)
            self.train_groups[source] = eligible
        self.rules = tuple(rules())
        self.masks = {
            (s, t): (self.arch_index[s], data.arch == t)
            for s, t in product(ARCHITECTURES, ARCHITECTURES)
        }

    def arrays(self, columns):
        return np.stack([matrix(rows, columns, len(STEPS)) for rows in self.archive])

    def members(self, source, excluded):
        eligible = self.train_groups[source]
        corpora = [c for c in self.data.corpora if c not in excluded]
        for calibration in corpora:
            fit_mask = eligible & ~np.isin(self.data.corpus, (*excluded, calibration))
            clean = eligible & self.clean & (self.data.corpus == calibration)
            if not clean.any():
                raise ValueError(f"Missing clean controls for {source}/{calibration}")
            yield calibration, np.flatnonzero(fit_mask), clean

    def forecasts(self, values, method):
        memory = np.zeros((len(self.rules), len(ARCHITECTURES), len(self.data.names), len(STEPS)), dtype=np.uint8)
        for source in ARCHITECTURES:
            src = self.arch_index[source]
            # Both omitted corpora give a fit reused in two different calibration/test assignments.
            for a, b in combinations(self.data.corpora, 2):
                eligible = self.train_groups[source]
                fit_mask = eligible & ~np.isin(self.data.corpus, (a, b))
                model = fit_member(method, values, np.flatnonzero(fit_mask), self.data)
                ids = np.flatnonzero(np.isin(self.data.corpus, (a, b)))
                scores = evidence(model, values[ids])
                by_mode = {m: smooth(scores, m) for m in ("instant", "mean2", "persistent")}
                for calibration, test in ((a, b), (b, a)):
                    ref = (eligible & self.clean & (self.data.corpus == calibration))[ids]
                    target = self.data.corpus[ids] == test
                    for j, rule in enumerate(self.rules):
                        values_by_mode = by_mode[rule.evidence_mode]
                        cutoff = cutoffs(values_by_mode[ref], rule, STEPS)
                        memory[j, src, ids[target]] += values_by_mode[target] > cutoff
        needed = np.array([r.quorum * (len(self.data.corpora) - 1) for r in self.rules])
        earliest = np.array([r.earliest for r in self.rules])
        active = (memory >= needed[:, None, None, None]) & (STEPS >= earliest[:, None, None, None])
        return np.where(active, STEPS, NEVER).min(-1).astype(np.uint16)

    def export(self, candidate, columns, source, excluded=()):
        values = self.arrays(columns)
        members = []
        for calibration, train, control in self.members(source, excluded):
            model = fit_member(candidate.learner, values, train, self.data)
            scores = smooth(evidence(model, values[control]), candidate.rule.evidence_mode)
            limit = cutoffs(scores, candidate.rule, STEPS)
            model.update(calibration=calibration, cutoff=limit)
            members.append(serial(model))
        return dict(source_architecture=source, fit_seeds=SEEDS, excluded_corpora=excluded, horizon=10, interval=5,
                    columns=columns, learner=candidate.learner, features=candidate.features,
                    rule=candidate.rule.export(), members=members)


def rates(data, times):
    onset = np.array([o["onset"] or NEVER for o in data.outcomes])
    clean = np.array([o["earliest_possible"] is None for o in data.outcomes])
    lead = onset - times
    metrics = {}
    for s, source in enumerate(ARCHITECTURES):
        for target in ARCHITECTURES:
            ids = data.arch == target
            positive = ids & (onset != NEVER)
            below = ids & clean
            timely = ((lead[:, s, positive] > 0) & (lead[:, s, positive] <= 10)).mean(1)
            false = (times[:, s, below] < NEVER).mean(1)
            advance = (lead[:, s, positive] > 0).mean(1)
            metrics[source, target] = np.stack((timely, false, advance), axis=-1)
    return metrics


def select(metrics, pairs, complexity, max_false=1 / 6):
    values = np.stack([metrics[p] for p in pairs])
    timely, false, advance = (values[..., j] for j in range(3))
    # Force every ranked pair to obey the same false-alert budget if feasible.
    excess = np.maximum(0, false.max(0) - max_false)
    rank = np.lexsort((complexity, false.mean(0), -advance.mean(0), -timely.mean(0), -timely.min(0), excess))
    return int(rank[0])


def summarize(data, times, pairs):
    results = {}
    for source, target in pairs:
        i = ARCHITECTURES.index(source)
        mask = data.arch == target
        row = {"ordinary": counts(times[i], data.outcomes, mask)}
        if target != "llama":
            row["conservative"] = counts(times[i], data.strict, mask)
        labels = classify(times[i], data.outcomes)
        row["runs"] = {
            n: dict(first_alert=int(times[i, j]) if times[i, j] < NEVER else None,
                    onset=data.outcomes[j]["onset"], status=labels[j])
            for j, n in enumerate(data.names) if mask[j]
        }
        results.setdefault(source, {})[target] = row
    return results


def verify_and_export(study, candidates, catalog_, idx, times, root):
    candidate = candidates[idx]
    columns = catalog_[candidate.features]
    for src, source in enumerate(ARCHITECTURES):
        deploy = study.export(candidate, columns, source)
        write_json(root / "banks" / f"{source}.json", deploy)
        for excluded in study.data.corpora:
            fold = study.export(candidate, columns, source, (excluded,))
            ids = np.flatnonzero(study.data.corpus == excluded)
            for i in ids:
                pts = study.data.points[i]
                expected = times[src, i]
                for j in range(len(pts)):
                    got = score_prefix(pts[:j+1], fold)["first_alert"]
                    if got != (int(expected) if expected <= pts[j]["step"] else None):
                        raise AssertionError((source, study.data.names[i], pts[j]["step"], expected, got))
    return candidate


def run(output):
    torch.set_num_threads(1)
    data = Cohort()
    study = Study(data)
    features = catalog()
    candidates, matrices = [], []
    for name, columns in features.items():
        values = study.arrays(columns)
        for method in METHODS:
            if method in ("neighbors", "kernel", "reservoir") and name.startswith("single_"):
                continue
            matrices.append(study.forecasts(values, method))
            candidates.extend(Candidate(name, method, r) for r in study.rules)
        print(f"Scored {name}", flush=True)
    times = np.concatenate(matrices)
    metrics = rates(data, times)
    complexity = [len(features[c.features]) for c in candidates]
    budgets = {"zero": 0., "strict": 1/12, "moderate": 1/6}
    chosen = {name: select(metrics, CROSS_PAIRS, complexity, limit) for name, limit in budgets.items()}
    reserve_source = {}
    reserve_target = {}
    source_only = {}
    for excluded in ARCHITECTURES:
        held_targets = [(s, t) for s, t in CROSS_PAIRS if t == excluded]
        # This excludes the held-out architecture from policy ranking in either role.
        development = [(s, t) for s, t in CROSS_PAIRS if excluded not in (s, t)]
        idx = select(metrics, development, complexity)
        reserve_target[excluded] = dict(candidate=asdict(candidates[idx]), metrics=summarize(data, times[idx], held_targets))
        held_sources = [(s, t) for s, t in CROSS_PAIRS if s == excluded]
        reserve_source[excluded] = dict(candidate=asdict(candidates[idx]), metrics=summarize(data, times[idx], held_sources))
        idx = select(metrics, [(excluded, excluded)], complexity)
        source_only[excluded] = dict(candidate=asdict(candidates[idx]), metrics=summarize(data, times[idx], held_sources))
    all_pairs = list(product(ARCHITECTURES, ARCHITECTURES))
    choice = verify_and_export(study, candidates, features, chosen["moderate"], times[chosen["moderate"]], output)
    report = dict(archive=data.identity, source_seeds=SEEDS, candidates=len(candidates), feature_groups=len(features),
                  methods=METHODS, steps=STEPS.tolist(), reserve_architecture_as_source=reserve_source,
                  reserve_architecture_as_target=reserve_target, source_only_selection=source_only,
                  selected={k: dict(candidate=asdict(candidates[i]), metrics=summarize(data, times[i], all_pairs))
                            for k, i in chosen.items()})
    write_json(output / "report.json", report)
    with (output / "policy-comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        keys = list(candidates[0].rule.export())
        writer.writerow(["features", "learner", *keys, *[f"{s}_to_{t}_{m}" for s, t in all_pairs
                                                          for m in ("timely_rate", "false_alert_rate", "advance_rate")]])
        for i, candidate in enumerate(candidates):
            values = [round(float(v), 9) for p in all_pairs for v in metrics[p][i]]
            writer.writerow([candidate.features, candidate.learner, *candidate.rule.export().values(), *values])
    print(f"Verified and exported {choice}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
