"""Evaluate causal warning models with entire architectures and corpora held out."""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from math import asinh, log
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, OUT, dataset, read_split
from run_monitor_transfer import classify
from stage_monitor_transfer import knn
from sweep_activation_monitors import summarize
from transfer_advice_early_warning import (
    AXES, EPS, LAYERS, REQUEST, feature_groups, feature_row, feature_sets, scoped_layers, target_outcomes, target_pairs,
)


TARGETS = {
    "olmo": Path("outputs/monitor-advice-transfer-v1/olmo"),
    "qwen": Path("outputs/qwen-advice-transfer-v1/full"),
}
OUTPUT = Path("outputs/cross-architecture-monitor-v1")
HORIZON = 10
PRIORITY_AXES = ("honesty", "sycophancy", REQUEST)
MID_LAYERS = ("14", "15")
INITIAL_MOTION_FLOOR = 0.02


@dataclass(frozen=True)
class Policy:
    method: str
    features: str
    persistence: int = 2
    evidence: str = "level"
    calibration: str = "max"

    @property
    def name(self):
        return "/".join(map(str, (self.method, self.features, self.persistence, self.evidence, self.calibration)))


def column_catalog():
    groups, directions = feature_groups()
    earlier = feature_sets(groups, directions)
    geom = groups["magnitude_growth"] + groups["orientation_growth"] + groups["growth_rate"]
    motion = {
        axis: tuple(f"{measure}_L{layer}_projection_{axis}" for layer in LAYERS
                    for measure in ("transition", "trend", "acceleration", "from_start", "prefix_mean"))
        for axis in AXES
    }
    ratio = {axis: directions[axis] for axis in AXES}
    recent = {axis: tuple(k for k in ratio[axis] if k.startswith("direction_rate_"))
              + tuple(k for k in motion[axis] if k.startswith(("transition_", "trend_", "acceleration_")))
              for axis in AXES}
    catalog = {
        key: earlier[key] for key in (
            "checkpoint_number", "raw_full", "raw_mid_layers", "raw_geometry", "raw_projections", "normalized_full",
            "normalized_mid_layers", "normalized_geometry", "normalized_directions", "normalized_growth_and_rate",
        )
    }
    catalog.update(
        directional_changes=sum(motion.values(), ()),
        priority_directions=sum((ratio[axis] for axis in PRIORITY_AXES), ()),
        priority_changes=sum((ratio[axis] + motion[axis] for axis in PRIORITY_AXES), ()),
        priority_recent=sum((recent[axis] for axis in PRIORITY_AXES), ()),
        all_recent=sum(recent.values(), ()),
    )
    for axes, name in ((PRIORITY_AXES, "priority"), (AXES, "all_axes")):
        history = sum((ratio[axis] + motion[axis] for axis in axes), ())
        catalog[f"{name}_history"] = history
        catalog[f"{name}_and_growth"] = history + geom
        for scope, layers in (("mid", ("14", "15")), ("layer15", ("15",)), ("layer31", ("31",))):
            catalog[f"{name}_{scope}"] = scoped_layers(history, layers)
    for axis in AXES:
        catalog[f"{axis}_history"] = ratio[axis] + motion[axis]
    for axis in PRIORITY_AXES:
        for layer in ("15", "31"):
            catalog[f"{axis}_layer{layer}"] = scoped_layers(catalog[f"{axis}_history"], (layer,))
    return catalog


def relative_direction(current, reference, first):
    return asinh((current - reference) / (INITIAL_MOTION_FLOOR + abs(reference - first)))


def prefix_features(points, last_step):
    """Every feature at a checkpoint depends only on that run's observed prefix."""
    signals = [point["signals"] for point in points]
    rows = []
    for i, point in enumerate(points[1:], 1):
        if point["step"] > last_step:
            break
        row = feature_row(points[0], points[1], points[i - 1], point)
        # Use step 10 as an anchor only once it has actually been observed.
        entry, reference = signals[1], signals[min(i, 2)]
        for layer in LAYERS:
            m = f"L{layer}_magnitude"
            for metric in ("magnitude", "orientation"):
                key = f"L{layer}_{metric}"
                change = log(max(signals[i][key], EPS) / max(reference[key], EPS))
                early = log(max(reference[key], EPS) / max(entry[key], EPS))
                row[f"after_initial_{key}"] = change
                row[f"relative_phase_{key}"] = change / max(abs(early), 0.2)
            for axis in AXES:
                p = f"L{layer}_projection_{axis}"
                recent = [s[p] / max(s[m], EPS) for s in signals[max(1, i - 2):i + 1]]
                now, before = signals[i], signals[i - 1]
                row[f"transition_{p}"] = (now[p] - before[p]) / max(now[m] + before[m], EPS)
                row[f"trend_{p}"] = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
                row[f"acceleration_{p}"] = recent[2] - 2 * recent[1] + recent[0] if len(recent) == 3 else 0.0
                row[f"from_start_{p}"] = (now[p] - signals[1][p]) / max(now[m] + signals[1][m], EPS)
                row[f"prefix_mean_{p}"] = sum(s[p] / max(s[m], EPS) for s in signals[1:i + 1]) / i
                early_ratio = reference[p] / max(reference[m], EPS)
                change = recent[-1] - early_ratio
                row[f"reset_direction_{p}"] = change
                row[f"relative_direction_{p}"] = relative_direction(recent[-1], early_ratio, entry[p] / max(entry[m], EPS))
                row[f"reset_projection_{p}"] = (now[p] - reference[p]) / max(now[m] + reference[m], EPS)
        rows.append(row)
    return rows


def load_cohorts():
    protocol_path = CONFIG / "protocol.json"
    protocol = read_json(protocol_path)
    protocol["forecast_steps"] = HORIZON
    signatures = set()
    cohorts = {"llama": read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)}
    identity = {
        "protocol_sha256": file_digest(protocol_path),
        "llama_summaries_sha256": {
            f"{corpus}/{seed}": file_digest(OUT / "summaries" / protocol["corpora"][corpus] / f"{corpus}-{seed}.json")
            for corpus, seed in sorted(cohorts["llama"])
        },
    }
    strict = {}
    for domain, root in TARGETS.items():
        source_path, report_path = root / f"{domain}-advice-refit-inputs.json", root / "report.json"
        archive, report = read_json(source_path), read_json(report_path)
        if (archive["reference_report_sha256"] != file_digest(report_path)
                or signatures != {tuple(archive["judge_identity"])}
                or report["steps"] != protocol["capture_steps"]):
            raise ValueError(f"Captured features or outcome rubric differ: {domain}")
        cohorts[domain] = target_pairs(archive, report)
        strict[domain] = target_outcomes(cohorts[domain], report, protocol)["conservative"]
        identity[domain] = dict(inputs_sha256=file_digest(source_path), report_sha256=file_digest(report_path))
    return protocol, cohorts, strict, identity


class Archive:
    def __init__(self, protocol, cohorts, conservative, catalog):
        self.corpora = tuple(sorted(protocol["corpora"]))
        steps = protocol["capture_steps"]
        self.steps = np.array([t for t in steps[1:] if t <= steps[-1] - HORIZON])
        self.ids, self.outcomes, self.conservative, self.points, self.features, self.labels = [], [], [], [], [], []
        self.match = []
        for model, cohort in cohorts.items():
            eligible, outcomes = dataset(cohort, protocol, sensitivity=False, activation_only=True)
            labels = {(r["run"], r["step"]): r["label"] for r in eligible}
            for (corpus, seed), pair in sorted(cohort.items()):
                i = len(self.ids)
                self.match.extend([i + 1, i])
                for arm in (0, 1):
                    points = [p for p in pair["records"] if p["branch"] == arm]
                    if [point["step"] for point in points] != steps:
                        raise ValueError(f"Unexpected checkpoint schedule: {model}/{corpus}/{seed}/{arm}")
                    run = f"{corpus}/{seed}/{arm}"
                    self.ids.append(f"{model}/{run}")
                    self.points.append(points)
                    self.features.append(prefix_features(points, self.steps[-1]))
                    self.outcomes.append(outcomes[run])
                    self.conservative.append(conservative.get(model, outcomes)[run])
                    self.labels.append([labels.get((run, t), -1) for t in self.steps])
        self.domains = np.array([name.split("/")[0] for name in self.ids])
        self.families = np.array([name.split("/")[1] for name in self.ids])
        self.clean = np.array([o["earliest_possible"] is None for o in self.outcomes])
        self.labels = torch.tensor(self.labels, dtype=torch.float64)
        self.match = np.array(self.match)
        self.catalog = catalog

    @lru_cache(maxsize=None)
    def matrix(self, features):
        columns = self.catalog[features]
        z = torch.tensor([[[point[k] for k in columns] for point in series] for series in self.features],
                         dtype=torch.float64)
        if not torch.isfinite(z).all():
            raise ValueError(f"Nonfinite data in feature set {features}")
        transform = features.partition(":")[0] if ":" in features else ""
        if transform == "center_first":
            z = z - z[:, :1]
        elif transform == "center_first_two":
            z = torch.cat((torch.zeros_like(z[:, :2]), z[:, 2:] - z[:, :2].mean(1, keepdim=True)), dim=1)
        elif transform == "scale_first_change":
            change = (z[:, 1:2] - z[:, :1]).abs()
            # No knowledge of the second point enters the first output.
            z = torch.cat((torch.zeros_like(z[:, :1]), torch.asinh((z[:, 1:] - z[:, :1]) / (0.02 + change))), dim=1)
        return z

    def select(self, domains, corpora):
        return np.flatnonzero(np.isin(self.domains, domains) & np.isin(self.families, corpora))


class SmallSequenceModel(torch.nn.Module):
    def __init__(self, features, recurrent):
        super().__init__()
        self.input = torch.nn.Linear(features, 12)
        self.history = torch.nn.GRU(12, 12, batch_first=True) if recurrent else None
        self.output = torch.nn.Linear(12, 1)

    def forward(self, x):
        state = torch.tanh(self.input(x))
        if self.history is not None:
            state, _ = self.history(state)
        return self.output(state).squeeze(-1)


def linear_weights(method, x, binary, deltas=None):
    if method == "lda":
        pos, neg = x[binary == 1], x[binary == 0]
        residual = torch.cat((pos - pos.mean(0), neg - neg.mean(0)))
        gram = residual.T @ residual / len(x)
        rhs = pos.mean(0) - neg.mean(0)
    else:
        gram = x.T @ x / len(x)
        rhs = x.T @ (binary - binary.mean()) / len(x)
        if deltas is not None and len(deltas):
            strength = 1.0 if method == "paired_ridge" else 0.20
            gram = gram + strength * (deltas.T @ deltas) / len(deltas)
            rhs = rhs + strength * deltas.mean(0)
    penalty = {"ridge_light": 0.05, "ridge_heavy": 2.0}.get(method, 0.5)
    return torch.linalg.solve(gram + penalty * torch.eye(x.shape[1], dtype=x.dtype), rhs)


def fit_scores(method, sequence, labels, training, partner):
    """Fit on eligible source windows and score each trajectory independently."""
    src = sequence[training]
    y = labels[training]
    eligible = y >= 0
    rows, binary = src[eligible], y[eligible]
    if torch.unique(binary).numel() != 2:
        raise ValueError("Training folds must have safe windows and windows preceding onset")
    center = rows.mean(dim=0)
    scale = rows.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (sequence - center) / scale
    x = standardized[training][eligible]
    z = standardized.flatten(end_dim=1)
    d = z.shape[-1]
    if method in ("mlp", "gru"):
        # Fix initialization, updates and regularization independent of the target.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(401)
            model = SmallSequenceModel(d, method == "gru").float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.016, weight_decay=0.04)
        inputs = standardized[training].float().clamp(-8, 8)
        for _ in range(72):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(inputs)[eligible], binary.float())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return model(standardized.float().clamp(-8, 8)).double()
    if method == "knn":
        values = torch.tensor(knn(x, binary, z), dtype=z.dtype)
    elif method == "kernel":
        k = torch.exp(-torch.cdist(x, x).square() / d)
        alpha = torch.linalg.solve(k + torch.eye(len(x), dtype=k.dtype), binary - binary.mean())
        values = torch.exp(-torch.cdist(z, x).square() / d) @ alpha
    else:
        deltas = None
        if method in ("paired_ridge", "pair_light"):
            matched = standardized[partner[training]]
            pair_ok = (y == 1) & (labels[partner[training]] == 0)
            deltas = (standardized[training] - matched)[pair_ok]
        elif method not in ("ridge", "ridge_light", "ridge_heavy", "lda"):
            raise ValueError(f"Unknown learner: {method}")
        values = z @ linear_weights(method, x, binary, deltas)
    return values.reshape(sequence.shape[:2])


def forecast_grid(catalog, *, neural):
    grid = [Policy("ridge", "checkpoint_number"), Policy("knn", "raw_full")]
    if neural:
        features = ("normalized_mid_layers", "priority_changes", "priority_mid", "all_axes_and_growth")
        return tuple(Policy(method, f, p, e) for method in ("mlp", "gru") for f in features
                     for p, e in ((1, "level"), (2, "level"), (1, "rise")))
    for name in catalog:
        if name == "checkpoint_number":
            continue
        for method in ("knn", "ridge", "paired_ridge", "lda", "kernel"):
            grid.extend(Policy(method, name, p) for p in (1, 2))
            grid.append(Policy(method, name, 1, "rise"))
            if name in ("raw_mid_layers", "normalized_mid_layers", "priority_changes", "priority_recent",
                        "priority_mid", "all_axes_mid", "all_axes_and_growth"):
                grid.extend((Policy(method, name, 2, "rise"), Policy(method, name, 1, calibration="margin"),
                             Policy(method, name, 2, calibration="margin"),
                             Policy(method, name, 1, "recent"), Policy(method, name, 1, calibration="step")))
    return tuple(dict.fromkeys(grid))


def prefix_catalog(catalog):
    choices = {key: catalog[key] for key in ("normalized_mid_layers", "priority_directions", "priority_mid",
                                             "raw_mid_layers", "normalized_directions", "normalized_full")}
    geo = tuple(f"{state}_L{layer}_{metric}" for layer in LAYERS for metric in ("magnitude", "orientation")
                for state in ("after_initial", "relative_phase"))
    block = {axis: tuple(f"{state}_L{layer}_projection_{axis}" for layer in LAYERS
                         for state in ("reset_direction", "relative_direction", "reset_projection")) for axis in AXES}
    for axes, name in ((PRIORITY_AXES, "three_directions"), (AXES, "all_directions")):
        prefix = sum((block[axis] for axis in axes), ())
        for scope, layers in (("all", LAYERS), ("mid", ("14", "15")), ("late", ("31",))):
            choices[f"reset_{name}_{scope}"] = scoped_layers(prefix, layers)
            choices[f"reset_{name}_{scope}_with_growth"] = scoped_layers(prefix + geo, layers)
            if name == "three_directions":
                choices[f"reset_relative_{scope}"] = tuple(k for k in scoped_layers(prefix, layers)
                                                           if k.startswith("relative_direction_"))
    choices["reset_honesty_sycophancy"] = block["honesty"] + block["sycophancy"]
    choices["reset_request"] = block[REQUEST]
    choices["reset_geometry"] = geo
    return choices


def prefix_grid(choices):
    grid = []
    for features in choices:
        for method in ("ridge", "paired_ridge", "lda", "kernel", "knn"):
            grid.extend(Policy(method, features, p, e) for p, e in (
                (1, "level"), (1, "rise"), (1, "initial_peak"), (1, "two_step"), (1, "rising_level"),
                (2, "initial_peak"),
            ))
            grid.append(Policy(method, features, 1, "rise", "quantile"))
            if features in ("normalized_mid_layers", "priority_directions"):
                grid.extend(Policy(method, features, 1, evidence) for evidence in
                            ("initial_mean", "rising_level_strong", "history_centered"))
    return tuple(dict.fromkeys(grid))


def centered_catalog(base, prefix):
    choices = {name: base[name] for name in (
        "raw_mid_layers", "raw_full", "normalized_mid_layers", "normalized_full", "priority_directions",
        "priority_changes", "priority_mid", "all_axes_mid", "priority_and_growth",
    )}
    choices["reset_relative_mid"] = prefix["reset_relative_mid"]
    return {f"{transform}:{name}": columns
            for transform in ("center_first", "center_first_two", "scale_first_change")
            for name, columns in choices.items()}


def centered_grid(choices):
    return tuple(Policy(method, columns, persistence, calibration=criterion)
                 for method in ("knn", "ridge", "ridge_light", "ridge_heavy", "paired_ridge", "pair_light", "kernel")
                 for columns in choices for persistence, criterion in ((1, "max"), (2, "max"), (1, "quantile")))


def adjust_scores(scores, evidence, persistence):
    if evidence == "level":
        adjusted = scores
    elif evidence == "rise":
        adjusted = scores - scores[:, :1]
    elif evidence == "recent":
        adjusted = torch.cat((torch.zeros_like(scores[:, :1]), scores[:, 1:] - scores[:, :-1]), dim=1)
    elif evidence in ("initial_peak", "initial_mean"):
        anchor = scores[:, :2].max(1, keepdim=True).values if evidence == "initial_peak" else scores[:, :2].mean(1, keepdim=True)
        adjusted = torch.cat((torch.zeros_like(scores[:, :2]), scores[:, 2:] - anchor), dim=1)
    elif evidence == "two_step":
        adjusted = scores - torch.cat((scores[:, :1].expand(-1, 2), scores[:, :-2]), dim=1)
    elif evidence == "history_centered":
        previous = torch.cat((scores[:, :1], scores[:, :-1].cumsum(1) /
                              torch.arange(1, scores.shape[1], dtype=scores.dtype)), dim=1)
        adjusted = scores - previous
    elif evidence in ("rising_level", "rising_level_strong"):
        recent = torch.cat((torch.zeros_like(scores[:, :1]), scores[:, 1:] - scores[:, :-1]), dim=1)
        adjusted = scores - scores[:, :1] + (recent if evidence == "rising_level" else 2 * recent)
    else:
        raise ValueError(f"Unknown evidence transform: {evidence}")
    if persistence == 2:
        return torch.cat((torch.full_like(adjusted[:, :1], -torch.inf),
                          torch.minimum(adjusted[:, 1:], adjusted[:, :-1])), dim=1)
    return adjusted


class Experiment:
    def __init__(self, archive):
        self.data = archive

    @lru_cache(maxsize=256)
    def raw_scores(self, method, features, sources, training):
        ids = self.data.select(sources, training)
        if not np.isin(self.data.match[ids], ids).all():
            raise ValueError("Matched training counterparts must be in the same fold")
        return fit_scores(method, self.data.matrix(features), self.data.labels, ids, self.data.match)

    @lru_cache(maxsize=768)
    def member(self, policy, sources, training, calibration):
        scores = self.raw_scores(policy.method, policy.features, sources, training)
        scores = adjust_scores(scores, policy.evidence, policy.persistence)
        ids = self.data.select(sources, (calibration,))
        safe = ids[self.data.clean[ids]]
        if not len(safe):
            raise ValueError(f"Empty no-onset calibration set in {calibration}")
        scores = scores.numpy()
        top = scores[safe].max()
        if policy.calibration == "step":
            cutoff = scores[safe].max(0)
        elif policy.calibration == "quantile":
            cutoff = np.quantile(scores[safe].max(1), 0.75)
        elif policy.calibration == "margin":
            positives = scores[ids][self.data.labels[ids].numpy() == 1]
            positives = positives[np.isfinite(positives)]
            cutoff = top + 0.5 * max(0.0, np.quantile(positives, 0.25) - top) if len(positives) else top
        else:
            cutoff = top
        return scores > cutoff

    def forecast(self, policy, sources, test, held_out, references):
        if held_out in references:
            raise ValueError("The outer corpus appeared in training or calibration")
        indices = self.data.select((test,), (held_out,))
        votes = np.zeros((len(indices), len(self.data.steps)), dtype=int)
        for c in references:
            fits = tuple(k for k in references if k != c)
            votes += self.member(policy, sources, fits, c)[indices]
        first = votes * 2 > len(references)
        return {
            int(i): dict(horizon=HORIZON, first_alert=int(self.data.steps[f.argmax()]) if f.any() else None,
                         scored_steps=self.data.steps.tolist(), votes=v.tolist())
            for i, v, f in zip(indices, votes, first, strict=True)
        }

    def outcomes(self, forecasts, *, strict=False):
        data = self.data.conservative if strict else self.data.outcomes
        return {self.data.ids[i]: classify(p, data[i]["onset"], data[i]["earliest_possible"])
                for i, p in forecasts.items()}

    def evaluate(self, policy, sources, test):
        return {i: results for c in self.data.corpora
                for i, results in self.forecast(policy, sources, test, c,
                tuple(other for other in self.data.corpora if other != c)).items()}

    def inner_quality(self, policy, sources, outer):
        remaining = tuple(c for c in self.data.corpora if c != outer)
        # Mimic source-to-new-architecture transfer inside each two-source fold.
        twins = [(sources, sources[0])] if len(sources) == 1 else [((a,), b) for a in sources for b in sources if a != b]
        scores = []
        for donors, domain in twins:
            results = {i: p for corpus in remaining
                       for i, p in self.forecast(policy, donors, domain, corpus,
                       tuple(c for c in remaining if c != corpus)).items()}
            stats = summarize(self.outcomes(results))
            scores.append(stats)
        return scores

    def clear(self):
        self.raw_scores.cache_clear()
        self.member.cache_clear()


MODES = {
    "llama_only": {"olmo": ("llama",), "qwen": ("llama",)},
    "held_out_architecture": {"olmo": ("llama", "qwen"), "qwen": ("llama", "olmo"),
                              "llama": ("olmo", "qwen")},
}


def rank_metrics(values):
    return (max(v["false_alert"] / max(1, v["no_onset_runs"]) for v in values),
            -min(v["timely"] / max(1, v["onset_runs"]) for v in values),
            -sum(v["timely"] for v in values), sum(v["false_alert"] for v in values),
            -sum(v["early"] for v in values))


def benchmark(archive, policies):
    engine = Experiment(archive)
    predictions = {name: {} for name in MODES}
    selections = {(mode, target, c): None for mode, ends in MODES.items()
                  for target in ends for c in archive.corpora}
    for index, policy in enumerate(policies):
        for name, ends in MODES.items():
            predictions[name][policy.name] = {}
            for target, sources in ends.items():
                predictions[name][policy.name][target] = engine.evaluate(policy, sources, target)
                for c in archive.corpora:
                    quality = engine.inner_quality(policy, sources, c)
                    rank = (*rank_metrics(quality), len(archive.catalog[policy.features]), index)
                    slot = (name, target, c)
                    incumbent = selections[slot]
                    if incumbent is None or rank < incumbent[0]:
                        selections[slot] = (rank, policy, quality)
        if (index + 1) % 100 == 0:
            print(f"Replayed {index+1}/{len(policies)} candidates", flush=True)
    selected = {name: {} for name in MODES}
    for name, ends in MODES.items():
        for target, sources in ends.items():
            decisions = {}
            folds = {}
            for c in archive.corpora:
                _, policy, quality = selections[name, target, c]
                f = predictions[name][policy.name][target]
                decisions.update({i: r for i, r in f.items() if archive.families[i] == c})
                folds[c] = dict(candidate=policy.name, inner_validation=quality)
            selected[name][target] = dict(
                source_architectures=sources, by_corpus=folds,
                metrics=summarize(engine.outcomes(decisions)), conservative=summarize(engine.outcomes(decisions, strict=True)),
                runs=engine.outcomes(decisions),
            )
    engine.clear()
    # Target ratings are used here to score each fixed candidate, without
    # affecting any fitted parameter, threshold or nested selection decision.
    fixed = {
        mode: {candidate: {target: dict(
            metrics=summarize(engine.outcomes(rows)), conservative=summarize(engine.outcomes(rows, strict=True)))
            for target, rows in ends.items()} for candidate, ends in grid.items()}
        for mode, grid in predictions.items()
    }
    ordering = {
        mode: sorted(grid, key=lambda candidate: (
            *rank_metrics([grid[candidate][target]["metrics"] for target in ("olmo", "qwen")]), candidate))
        for mode, grid in fixed.items()
    }
    for mode, ranked in ordering.items():
        for candidate in set(ranked[:20]) | ({Policy("knn", "raw_full").name} & set(ranked)):
            for target, rows in predictions[mode][candidate].items():
                fixed[mode][candidate][target]["runs"] = engine.outcomes(rows)
    return dict(fixed=fixed, source_selected=selected, retrospective_rank=ordering)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--centered", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol, cohorts, strict, identity = load_cohorts()
    catalog = column_catalog()
    if sum((args.neural, args.prefix, args.centered)) > 1:
        parser.error("Select one experiment at a time.")
    if args.centered:
        catalog = centered_catalog(catalog, prefix_catalog(catalog))
    elif args.prefix:
        catalog = prefix_catalog(catalog)
    policies = centered_grid(catalog) if args.centered else (
        prefix_grid(catalog) if args.prefix else forecast_grid(catalog, neural=args.neural))
    filename = "centered" if args.centered else "prefix" if args.prefix else "neural" if args.neural else "classical"
    design = dict(
        identity=identity, models=MODES, forecast_steps=HORIZON, feature_columns=catalog,
        candidates=[p.name for p in policies],
        evaluation="Leave entire corpus out of fitting, calibration and source-only selection. "
                   "The target architecture is never part of those source sets.",
        calibration="Disjoint corpus. Compare to persistent safe scores, optionally with a margin "
                    "from that corpus's early-positive scores. A strict majority votes to alert.",
        selection="For two sources, select on held-out corpora and cross-architecture transfers "
                  "between those two sources. Retrospective ranking uses both target archives.",
    )
    write_json(args.output / f"{filename}-design.json", design)
    print(f"Replaying {len(policies)} {filename} candidates on three architectures", flush=True)
    result = benchmark(Archive(protocol, cohorts, strict, catalog), policies)
    result["design_sha256"] = file_digest(args.output / f"{filename}-design.json")
    write_json(args.output / f"{filename}-report.json", result)
    for mode, winners in result["retrospective_rank"].items():
        print(mode, {target: r["metrics"] for target, r in result["source_selected"][mode].items()}, flush=True)
        for candidate in winners[:4]:
            print(candidate, {target: cell["metrics"] for target, cell in result["fixed"][mode][candidate].items()}, flush=True)


if __name__ == "__main__":
    main()
