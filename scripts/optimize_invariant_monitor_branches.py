"""Test phase-aware early and later branches, each fitted on a single source."""
import argparse
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import AlertRule, phase_mask, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import evidence, smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import Cohort, NEVER, STEPS, serial
from optimize_source_invariant_monitor import (
    ARCHITECTURES, CROSS_PAIRS, SEEDS, Study, catalog, cutoffs, directional, fit_member, rates, select, summarize,
)

OUTPUT = Path("outputs/source-invariant-monitor-v4")
LATE_METHODS = ("ridge", "paired_ridge", "logistic", "lda", "kernel", "neighbors", "pair_difference")
EARLY_METHODS = ("ridge", "paired_ridge", "logistic", "lda", "pair_difference")


@dataclass(frozen=True)
class Branch:
    stage: str
    feature: str
    learner: str
    rule: AlertRule
    training_labels: str


def designs(learned_only=False):
    pool = catalog()
    angle = "g/angle/L31/magnitude"
    prior = "g/angle_previous/L31/magnitude"
    middle = ("g/angle/L14/magnitude", "g/angle/L15/magnitude")
    balance = "g/angle_balance/L31/magnitude"
    motion = ("g/angle_rate/L31/magnitude", "g/angle_balance_change/L31/magnitude")
    request = ("d/vector_ratio/L31/request_contrast_candidate",)
    confidence = ("d/cosine/L31/confidence",)
    honesty = ("d/velocity/L31/honesty",)
    early = {
        "angle": (angle,), "angle_previous": (prior,),
        "angle_balance": (balance,), "two_angles": (angle, prior),
        "angle_motion": (angle, *motion), "angle_depth": (angle, *middle),
        "angle_depth_motion": (balance, *motion),
        "angle_and_request": (angle, *request), "angle_and_confidence": (angle, *confidence),
        "angle_and_honesty": (angle, *honesty), "angle_and_semantics": (angle, *request, *confidence, *honesty),
        "angle_balance_and_request": (balance, *request),
        "angle_balance_and_honesty": (balance, *honesty),
        "late_semantics": (*request, *confidence, *honesty),
        "late_geometry": pool["geometry_raw_last"], "late_shape": pool["geometry_angular_shape_last"],
        "projection_priority": directional(("projection",), (31,), ("honesty", "confidence", "request_contrast_candidate")),
        "projection_normalized": directional(("vector_ratio",), (31,), ("honesty", "confidence", "request_contrast_candidate")),
    }
    if learned_only:
        for name in ("cosine", "angular_projection", "angular_projection_motion", "vector_motion",
                     "relative_direction", "reset5", "ratio_velocity"):
            sign_features = directional((name,), (31,), ("request_contrast_candidate",))
            early[f"early_request_{name}"] = sign_features
            early[f"joint_angular_{name}"] = (angle, prior, balance, *sign_features)
        early["joint_rates"] = (angle, prior, balance, "g/rate/L31/magnitude", *request, *honesty)
    late_names = ("direction_reset_priority", "direction_reset_wide_priority", "direction_reset_wide_all",
                  "direction_reset_wide_behavior", "direction_reset_all", "direction_reset_by_norm_priority",
                  "direction_reset_unscaled_priority", "direction_reset_unscaled_behavior",
                  "direction_reset_behavior", "reset_layer15", "reset_late", "joint_angular", "joint_relative")
    late = {k: pool[k] for k in late_names}
    for t in ("honesty", "sycophancy", "helpfulness", "harmlessness", "request_contrast_candidate"):
        late[f"reset_{t}"] = directional(("reset10",), (14, 15, 31), (t,))
    for excluded in ("honesty", "sycophancy", "request_contrast_candidate"):
        traits = tuple(t for t in ("honesty", "sycophancy", "request_contrast_candidate") if t != excluded)
        late[f"reset_without_{excluded}"] = directional(("reset10",), (14, 15), traits)
    return early, late


def specification(stage, feature, method, learned_only):
    if stage == "early":
        gates = ("any", "slow/0.2", "slow/0.5")
        if not learned_only:
            gates += ("slow_with_projection/0.2", "slow_with_projection/0.5")
        for labels, margin, gate, quorum in product(
            ("eventual", "next_15"), (-3., -1.5, -.6, 0., .3, 1.0), gates, (.51, .8)
        ):
            yield Branch(stage, feature, method, AlertRule("instant", 10, "checkpoint", margin, quorum, gate), labels)
        return
    for mode, calibration, margin, gate, quorum in product(
        ("instant", "mean2"), ("all", "zero_reference"),
        (0., .25, .5, 1.0), ("any", "slow_or_persistent/0.5", "slow_or_persistent/0.65"), (.51, .8)
    ):
        if calibration == "zero_reference" and (margin == 0 or mode == "mean2"):
            continue
        yield Branch(stage, feature, method, AlertRule(mode, 15, calibration, margin, quorum, gate), "imminent")


def for_fit(data, stage, label_mode):
    """Keep the study's checkpoint labels, or label early examples by eventual measured onset."""
    import copy

    view = copy.copy(data)
    labels = data.labels.copy()
    if stage == "early":
        if label_mode == "eventual":
            for i, outcome in enumerate(data.outcomes):
                if outcome["onset"] is not None and outcome["onset"] > 10 and labels[i, 1] >= 0:
                    labels[i, 1] = 1
        else:
            # Widen early training only enough to provide eligible events in every OLMo fold.
            # The deployed forecast horizon and all evaluation labels stay at ten steps.
            for i, outcome in enumerate(data.outcomes):
                if outcome["onset"] is not None and 10 < outcome["onset"] <= 25 and labels[i, 1] >= 0:
                    labels[i, 1] = 1
        labels[:, STEPS != 10] = -1
    view.labels = labels
    return view


def branch_cutoff(samples, rule, steps, zero):
    if rule.calibration == "zero_reference":
        peak = samples[:, steps >= rule.earliest].max()
        return float(peak + rule.margin * abs(peak - zero))
    return cutoffs(samples, rule, steps)


def vote_grid(study, values, items):
    stage, learner = items[0].stage, items[0].learner
    views = {key: for_fit(study.data, stage, key) for key in {item.training_labels for item in items}}
    n_sources, n_runs = len(ARCHITECTURES), len(study.data.names)
    votes = np.zeros((len(items), n_sources, n_runs, len(STEPS)), dtype=np.uint8)
    cal_steps = STEPS if stage == "late" else STEPS[1:2]
    for src, source in enumerate(ARCHITECTURES):
        eligible = study.train_groups[source]
        for a, b in combinations(study.data.corpora, 2):
            train = np.flatnonzero(eligible & ~np.isin(study.data.corpus, (a, b)))
            ids = np.flatnonzero(np.isin(study.data.corpus, (a, b)))
            unique_modes = {(x.training_labels, x.rule.evidence_mode) for x in items}
            models = {key: fit_member(learner, values, train, view) for key, view in views.items()}
            raw = {key: evidence(model, values[ids]) for key, model in models.items()}
            zeros = {key: evidence(model, np.zeros_like(values[:1])).item(2) for key, model in models.items()}
            by_mode = {(key, mode): smooth(raw[key], mode) for key, mode in unique_modes}
            for calibration, test in ((a, b), (b, a)):
                reference = (eligible & study.clean & (study.data.corpus == calibration))[ids]
                target = study.data.corpus[ids] == test
                thresholds = {}
                for j, item in enumerate(items):
                    rule = item.rule
                    values_by_mode = by_mode[item.training_labels, rule.evidence_mode]
                    key = (item.training_labels, rule.evidence_mode, rule.calibration, rule.margin)
                    if key not in thresholds:
                        samples = values_by_mode[reference]
                        if stage == "early":
                            samples = samples[:, 1:2]
                        thresholds[key] = branch_cutoff(samples, rule, cal_steps, zeros[item.training_labels])
                    if stage == "early":
                        votes[j, src, ids[target], 1] += values_by_mode[target, 1] > thresholds[key][0]
                    else:
                        votes[j, src, ids[target]] += values_by_mode[target] > thresholds[key]
    steps = STEPS[None, :]
    times = []
    for item, member_votes in zip(items, votes, strict=True):
        rule = item.rule
        active = (member_votes >= rule.quorum * (len(study.data.corpora)-1)) & (steps >= rule.earliest)
        if stage == "early":
            active &= steps == 10
        if rule.gate != "any":
            gate = np.stack([phase_mask(cols, rule.gate) for cols in study.archive])
            if stage == "late":
                gate = gate | np.pad(active[..., :-1], ((0,0), (0,0), (1,0)))
            active &= gate
        times.append(np.where(active, STEPS, NEVER).min(-1))
    return np.asarray(times, dtype=np.uint16)


def prepare(study, stage, features, methods, learned_only):
    items, times = [], [np.full((len(ARCHITECTURES), len(study.data.names)), NEVER, dtype=np.uint16)]
    items.append(Branch(stage, "disabled", "none", AlertRule("instant", 15, "all", 0, .51), "none"))
    for feature, columns in features.items():
        values = study.arrays(columns)
        for method in methods:
            batch = list(specification(stage, feature, method, learned_only))
            items.extend(batch)
            times.extend(vote_grid(study, values, batch))
        print(f"Scored {stage} {feature}", flush=True)
    return items, np.asarray(times)


def slim(data, times, pairs, limit=256):
    """Shortlist from development pairs only, retaining low-false-alarm operating points."""
    selected = set()
    indices = [ARCHITECTURES.index(src) for src, _ in pairs]
    # Drop duplicate development predictions before the shortlist is chosen.
    blocks = [times[:, s, data.arch == t] for s, (_, t) in zip(indices, pairs, strict=True)]
    _, unique = np.unique(np.concatenate(blocks, axis=1), return_index=True, axis=0)
    metrics = rates(data, times[unique])
    measure = np.stack([metrics[p] for p in pairs])
    for budget in (0., 1/12, 1/6, .25):
        early, false, advance = measure.transpose(2, 0, 1)
        over = np.maximum(0, false.max(0) - budget)
        for by_worst in (False, True):
            lead = early.min(0) if by_worst else early.mean(0)
            order = np.lexsort((unique, false.mean(0), -advance.mean(0), -early.mean(0), -lead, over))
            selected.update(unique[order[:limit//8]])
    selected.add(0)
    return np.array(sorted(selected))


def fuse(data, early, late, pairs, budget):
    eitems, etimes = early
    litems, ltimes = late
    en = slim(data, etimes, pairs)
    ln = slim(data, ltimes, pairs)
    ids = list(product(en, ln))
    candidates = np.minimum(etimes[en, None], ltimes[None, ln]).reshape(-1, *ltimes.shape[1:])
    # Break exact ties in favor of one branch and fewer model parameters.
    sizes = [int(i != 0) + int(j != 0) for i, j in ids]
    winner = select(rates(data, candidates), pairs, sizes, budget)
    i, j = ids[winner]
    return (eitems[i], litems[j]), candidates[winner]


def make_bank(study, item, features, source, excluded):
    if item.learner == "none":
        return None
    columns = features[item.feature]
    values = study.arrays(columns)
    view = for_fit(study.data, item.stage, item.training_labels)
    members = []
    for calibration, train, control in study.members(source, excluded):
        model = fit_member(item.learner, values, train, view)
        scores = smooth(evidence(model, values[control]), item.rule.evidence_mode)
        if item.stage == "early":
            limits = np.zeros(len(STEPS))
            limits[1] = cutoffs(scores[:, 1:2], item.rule, STEPS[1:2])[0]
        else:
            zero = evidence(model, np.zeros_like(values[:1])).item(2)
            limits = branch_cutoff(scores, item.rule, STEPS, zero)
        model.update(calibration=calibration, cutoff=limits)
        members.append(serial(model))
    return dict(source_architecture=source, fit_seeds=SEEDS, excluded_corpora=excluded, horizon=10, interval=5,
                columns=columns, learner=item.learner, features=item.feature, training_labels=item.training_labels,
                rule=item.rule.export(), last_step=10 if item.stage == "early" else int(STEPS[-1]), members=members)


def save_banks(root, study, choice, features, times):
    stored = {}
    for s, source in enumerate(ARCHITECTURES):
        stages = [make_bank(study, item, design, source, ()) for item, design in zip(choice, features, strict=True)]
        banks = dict(format="parallel", source_architecture=source, horizon=10,
                     branches={stage: bank for stage, bank in zip(("early", "late"), stages) if bank is not None})
        path = root / f"{source}.json"
        write_json(path, banks)
        stored[source] = file_digest(path)
        for corpus in study.data.corpora:
            folds = [make_bank(study, item, design, source, (corpus,)) for item, design in zip(choice, features, strict=True)]
            for i in np.flatnonzero(study.data.corpus == corpus):
                points = study.data.points[i]
                for j in range(len(points)):
                    active = [score_prefix(points[:j+1], fold)["first_alert"] for fold in folds if fold is not None]
                    first = min((x for x in active if x is not None), default=None)
                    expected = times[s, i] if times[s, i] <= points[j]["step"] else None
                    if first != expected:
                        raise AssertionError((source, study.data.names[i], points[j]["step"], first, expected))
    return stored


def run(output, learned_only):
    torch.set_num_threads(1)
    data = Cohort()
    study = Study(data)
    early_features, late_features = designs(learned_only)
    early = prepare(study, "early", early_features, EARLY_METHODS, learned_only)
    late = prepare(study, "late", late_features, LATE_METHODS, learned_only)
    report = dict(archive=data.identity, matched_fitting_seeds=SEEDS,
                  early_policies=len(early[0]), late_policies=len(late[0]),
                  retrospective={}, excluded_architecture={})
    for name, budget in (("zero", 0.), ("strict", 1/12), ("moderate", 1/6)):
        choice, times = fuse(data, early, late, CROSS_PAIRS, budget)
        print(name, choice, flush=True)
        entry = dict(policy=list(map(asdict, choice)), metrics=summarize(data, times, product(ARCHITECTURES, repeat=2)))
        entry["banks"] = save_banks(output / name, study, choice, (early_features, late_features), times)
        report["retrospective"][name] = entry
    for arch in ARCHITECTURES:
        dev = [(s, t) for s, t in CROSS_PAIRS if arch not in (s, t)]
        folds = {}
        for name, budget in (("zero", 0.), ("strict", 1/12), ("moderate", 1/6)):
            choice, times = fuse(data, early, late, dev, budget)
            folds[name] = dict(policy=list(map(asdict, choice)),
                metrics=summarize(data, times, [(s, t) for s, t in CROSS_PAIRS if arch in (s, t)]))
        report["excluded_architecture"][arch] = folds
        print(f"Finished design selection excluding {arch}", flush=True)
    write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-projection-gate", action="store_true",
                        help="Learn the early directional contribution without an explicit projection sign gate")
    args = parser.parse_args()
    run(args.output, args.no_projection_gate)


if __name__ == "__main__":
    main()
