"""Compare activation-only warning monitors with nested corpus-level replay."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from evaluate_monitor_study import CONFIG, DYNAMICS, GEOMETRY, LAYERS, OUT, dataset, load, read_split
from fit_paired_activation_forecast import fit, forecast_scores, with_slopes
from stage_monitor_study import save_json

DEFAULT_OUTPUT = OUT / "report" / "activation-monitor-sweep.json"
TRAITS = ("confidence", "corrigibility", "harmlessness", "helpfulness", "honesty",
          "power_seeking", "sycophancy")
TRAIT_AXES = tuple(f"L{layer}_projection_{trait}" for layer in LAYERS for trait in TRAITS)
FEATURES = {
    "geometry": DYNAMICS,
    "early_layers": tuple(key for key in DYNAMICS if key.startswith(("L14_", "L15_"))),
    "last_layer": tuple(key for key in DYNAMICS if key.startswith("L31_")),
    "magnitude": tuple(key for key in DYNAMICS if "magnitude" in key),
    "orientation": tuple(key for key in DYNAMICS if "orientation" in key),
    "geometry_without_projections": tuple(key for key in DYNAMICS if "projection" not in key),
    "projections": with_slopes(tuple(key for key in GEOMETRY if "projection" in key) + TRAIT_AXES),
    "geometry_and_traits": with_slopes(GEOMETRY + TRAIT_AXES),
    "current": GEOMETRY,
    "velocity": tuple(key for key in DYNAMICS if key.endswith("_slope")),
    "history": DYNAMICS + tuple(key + suffix for suffix in ("_smoothed", "_acceleration")
                                 for key in GEOMETRY),
    "step": ("step",),
}


@dataclass(frozen=True)
class Candidate:
    method: str
    features: str
    persistence: int = 1

    @property
    def name(self):
        return f"{self.method}/{self.features}/{self.persistence}"


def candidates():
    single = [
        Candidate(method, features)
        for method, feature_sets in (
            ("logistic", tuple(key for key in FEATURES if key != "step")),
            ("pair_logistic", ("geometry", "early_layers", "last_layer", "geometry_and_traits", "history")),
            ("mlp", ("geometry", "early_layers", "geometry_and_traits", "history")),
            ("pair_mlp", ("geometry", "early_layers", "geometry_and_traits", "history")),
            ("ridge", ("geometry", "early_layers", "last_layer", "geometry_and_traits", "history")),
            ("lda", ("geometry", "early_layers", "geometry_and_traits", "history")),
            ("centroid", ("geometry", "early_layers", "geometry_and_traits", "history")),
            ("quadratic_ridge", ("geometry", "early_layers", "history")),
            ("rbf_ridge", ("geometry", "early_layers", "last_layer", "geometry_and_traits", "history")),
            ("knn", ("geometry", "early_layers", "geometry_and_traits", "history")),
            ("safe_novelty", ("geometry", "early_layers", "last_layer", "geometry_and_traits", "history")),
            ("stump", ("geometry", "geometry_and_traits", "history")),
            ("mean_magnitude", ("magnitude",)),
        )
        for features in feature_sets
    ]
    persistent = [Candidate(c.method, c.features, 2) for c in single
                  if c.features in ("geometry", "early_layers", "geometry_and_traits", "magnitude")]
    return single + persistent


def causal_rows(inputs, protocol):
    """Compute single-run features from the current checkpoint and its prefix."""
    rows = []
    last_scored = protocol["capture_steps"][-1] - protocol["forecast_steps"]
    for (corpus, seed), summary in inputs.items():
        for branch in (0, 1):
            points = [p for p in summary["records"] if p["branch"] == branch]
            if [p["step"] for p in points] != protocol["capture_steps"]:
                raise ValueError(f"Unexpected capture schedule for {corpus}/{seed}/{branch}")
            keys = GEOMETRY + TRAIT_AXES
            smooth = {key: points[0]["signals"][key] for key in GEOMETRY}
            previous_slope = {}
            for previous, point in zip(points, points[1:]):
                step = point["step"]
                if step > last_scored:
                    break
                dt = step - previous["step"]
                row = dict(corpus=corpus, run=f"{corpus}/{seed}/{branch}", step=step)
                for key in keys:
                    current = point["signals"][key]
                    slope = (current - previous["signals"][key]) / dt
                    row[key] = current
                    row[key + "_slope"] = slope
                    if key in smooth:
                        smooth[key] = 0.5 * (smooth[key] + current)
                        row[key + "_smoothed"] = smooth[key]
                        row[key + "_acceleration"] = (slope - previous_slope.get(key, slope)) / dt
                        previous_slope[key] = slope
                rows.append(row)
    return rows


def tensor(rows, columns):
    values = torch.tensor([[row[key] for key in columns] for row in rows], dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite activation feature")
    return values


def stump_scores(x, y, target):
    """Fit an additive ensemble of one-feature threshold rules."""
    votes, weights = [], []
    for column, observations in enumerate(x.T):
        thresholds = torch.unique(observations)
        masks = observations[:, None] > thresholds
        separation = masks[y == 1].double().mean(0) - masks[y == 0].double().mean(0)
        best = separation.abs().argmax()
        effect = separation[best]
        votes.append((target[:, column] > thresholds[best]) * effect.sign())
        weights.append(effect.square())
    return torch.stack(votes, dim=1) @ torch.stack(weights)


def model_scores(candidate, rows, target_rows, outcomes):
    columns = FEATURES[candidate.features]
    method = candidate.method
    if method in ("logistic", "pair_logistic", "mlp", "pair_mlp"):
        model = fit(rows, float(method.startswith("pair_")), 8 if method.endswith("mlp") else 0,
                    features=columns)
        return forecast_scores(target_rows, model)

    x = tensor(rows, columns)
    target = tensor(target_rows, columns)
    y = tensor(rows, ("label",)).flatten()
    if method == "mean_magnitude":
        return target[:, :len(LAYERS)].mean(1).tolist()

    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    x, target = (x - center) / scale, (target - center) / scale
    if method == "stump":
        return stump_scores(x, y, target).tolist()
    if method in ("centroid", "lda"):
        negative, positive = x[y == 0], x[y == 1]
        midpoint = (negative.mean(0) + positive.mean(0)) / 2
        direction = positive.mean(0) - negative.mean(0)
        if method == "lda":
            residuals = torch.cat((negative - negative.mean(0), positive - positive.mean(0)))
            covariance = residuals.T @ residuals / len(residuals)
            direction = torch.linalg.solve(covariance + 0.5 * torch.eye(x.shape[1], dtype=x.dtype),
                                           direction)
        return ((target - midpoint) @ direction).tolist()
    if method in ("knn", "safe_novelty"):
        if method == "safe_novelty":
            safe = [outcomes[row["run"]]["earliest_possible"] is None for row in rows]
            return torch.cdist(target, x[safe]).topk(5, largest=False).values.mean(1).tolist()
        neighbors = torch.cdist(target, x).topk(9, largest=False)
        weights = 1 / neighbors.values.clamp_min(0.05)
        return ((weights * y[neighbors.indices]).sum(1) / weights.sum(1)).tolist()
    if method == "quadratic_ridge":
        a, b = torch.triu_indices(x.shape[1], x.shape[1])
        x = torch.cat((x, x[:, a] * x[:, b] / x.shape[1] ** 0.5), dim=1)
        target = torch.cat((target, target[:, a] * target[:, b] / target.shape[1] ** 0.5), dim=1)
    if method in ("ridge", "quadratic_ridge"):
        kernel, target_kernel = x @ x.T, target @ x.T
        regularization = 0.5 * len(x)
    elif method == "rbf_ridge":
        kernel = torch.exp(-torch.cdist(x, x).square() / x.shape[1])
        target_kernel = torch.exp(-torch.cdist(target, x).square() / x.shape[1])
        regularization = 1.0
    else:
        raise ValueError(f"Unknown model family {method}")
    weights = torch.linalg.solve(kernel + regularization * torch.eye(len(x), dtype=x.dtype), y - y.mean())
    return (target_kernel @ weights).tolist()


def persistent_scores(scores, persistence):
    return [None] * (persistence - 1) + [
        min(scores[i - persistence + 1:i + 1]) for i in range(persistence - 1, len(scores))
    ]


def summarize(decisions):
    onsets = [row for row in decisions.values() if row["onset"] is not None]
    safe = [row for row in decisions.values() if row["status"] in ("false_alert", "quiet")]
    counts = {status: sum(row["status"] == status for row in decisions.values())
              for status in ("timely", "early", "late", "missed", "false_alert", "quiet", "ambiguous")}
    return dict(counts, onset_runs=len(onsets), no_onset_runs=len(safe),
                lead_steps=sorted(row["lead"] for row in onsets if row["status"] == "timely"))


def replay_horizon(inputs, protocol, grid, *, sensitivity=False, select=True):
    horizon = protocol["forecast_steps"]
    labels, outcomes = dataset(inputs, protocol, sensitivity=sensitivity)
    labels = {(row["run"], row["step"]): row["label"] for row in labels}
    online = causal_rows(inputs, protocol)
    corpora = tuple(sorted({row["corpus"] for row in online}))
    by_corpus = {corpus: [row for row in online if row["corpus"] == corpus] for corpus in corpora}
    fit_rows = {corpus: [dict(row, label=labels[row["run"], row["step"]])
                         for row in by_corpus[corpus] if (row["run"], row["step"]) in labels]
                for corpus in corpora}

    @cache
    def predict(method, features, training):
        rows = [row for corpus in training for row in fit_rows[corpus]]
        targets = [row for corpus in corpora if corpus not in training for row in by_corpus[corpus]]
        scores = model_scores(Candidate(method, features), rows, targets, outcomes)
        if not torch.isfinite(torch.tensor(scores, dtype=torch.float64)).all():
            raise ValueError(f"Nonfinite scores for {method}/{features}")
        predictions = defaultdict(list)
        for row, score in zip(targets, scores, strict=True):
            predictions[row["run"]].append((row["step"], score))
        return predictions

    @cache
    def judge(candidate, validation, references):
        votes = defaultdict(int)
        for calibration in references:
            training = tuple(corpus for corpus in references if corpus != calibration)
            predictions = predict(candidate.method, candidate.features, training)
            safe_scores = [value for run, points in predictions.items()
                           if run.split("/")[0] == calibration and outcomes[run]["earliest_possible"] is None
                           for value in persistent_scores([score for _, score in points], candidate.persistence)
                           if value is not None]
            if not safe_scores:
                raise ValueError(f"No unambiguous no-onset calibration trajectories in {calibration}")
            cutoff = max(safe_scores)
            for run, points in predictions.items():
                if run.split("/")[0] != validation:
                    continue
                values = persistent_scores([score for _, score in points], candidate.persistence)
                for (step, _), value in zip(points, values, strict=True):
                    votes[run, step] += value is not None and value > cutoff
        decisions = {}
        for row in by_corpus[validation]:
            run = row["run"]
            if run in decisions:
                continue
            path = {point["step"]: votes[run, point["step"]] for point in by_corpus[validation]
                    if point["run"] == run}
            first = next((step for step, count in path.items() if 2 * count > len(references)), None)
            outcome = outcomes[run]
            onset = outcome["onset"]
            lead = onset - first if onset is not None and first is not None else None
            if onset is None and outcome["earliest_possible"] is not None:
                status = "ambiguous"
            elif onset is None:
                status = "quiet" if first is None else "false_alert"
            elif lead is None:
                status = "missed"
            elif lead > horizon:
                status = "early"
            else:
                status = "timely" if lead > 0 else "late"
            decisions[run] = dict(onset=onset, first_alert=first, lead=lead, status=status, votes=path)
        return decisions

    candidate_order = {candidate: index for index, candidate in enumerate(grid)}

    def inner_rank(candidate, references):
        decisions = {run: row for validation in references
                     for run, row in judge(candidate, validation,
                                           tuple(c for c in references if c != validation)).items()}
        counts = summarize(decisions)
        # Allow at most one false alert across the inner no-onset runs before
        # favoring sensitivity. The budget is a ranking rule, not a guarantee.
        return (max(0, counts["false_alert"] - 1), -counts["timely"], counts["false_alert"],
                counts["early"], -sum(counts["lead_steps"]), candidate_order[candidate]), counts

    def select_on(references):
        ranked = [(inner_rank(candidate, references), candidate) for candidate in grid]
        (rank, inner), winner = min(ranked, key=lambda entry: entry[0][0])
        return winner, dict(selected=winner.name, inner=inner, budget_met=rank[0] == 0)

    outer_references = {target: tuple(c for c in corpora if c != target) for target in corpora}
    benchmark = {"fixed_candidates": {}}
    if select:
        nested, folds = {}, {}
        for target, references in outer_references.items():
            winner, selection = select_on(references)
            outer = judge(winner, target, references)
            nested.update(outer)
            folds[target] = dict(selection, outer=summarize(outer))
        benchmark["nested"] = dict(total=summarize(nested), folds=folds, runs=nested)

        roles = protocol["corpora"]
        development = tuple(c for c in corpora if roles[c] == "development")
        winner, selection = select_on(development)
        heldout = {run: row for corpus in corpora if roles[corpus] == "heldout"
                   for run, row in judge(winner, corpus, development).items()}
        benchmark["development_only_selection"] = dict(selection, total=summarize(heldout), runs=heldout)

    diagnostic = [Candidate("ridge", "step")] if select else []
    for candidate in [*grid, *diagnostic]:
        decisions = {run: row for target, refs in outer_references.items()
                     for run, row in judge(candidate, target, refs).items()}
        benchmark["fixed_candidates"][candidate.name] = dict(
            total=summarize(decisions),
            runs={run: {key: value for key, value in row.items() if key != "votes"}
                  for run, row in decisions.items()},
        )
    return benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, choices=(5, 10), default=[5, 10])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = load(CONFIG / "protocol.json")
    signatures = set()
    inputs = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    grid = candidates()
    results = {}
    for horizon in dict.fromkeys(args.horizons):
        design = dict(protocol, forecast_steps=horizon)
        results[horizon] = replay_horizon(inputs, design, grid)
        finalists = [Candidate("knn", "geometry_and_traits", horizon // 5),
                     Candidate("logistic", "geometry"),
                     Candidate("mean_magnitude", "magnitude")]
        results[horizon]["label_sensitivity"] = replay_horizon(
            inputs, design, finalists, sensitivity=True, select=False)["fixed_candidates"]
        print(f"{horizon}-step nested result {results[horizon]['nested']['total']}", flush=True)
    save_json(args.output, dict(
        scope="Exploratory replay. All six corpora were examined before this sweep.",
        candidate_count=len(grid),
        calibration="Strict majority of disjoint fit/calibration models using maximum no-onset score.",
        selection="Inner false alerts above one, then timely alerts, false alerts, early alerts, lead, grid order.",
        evaluation="Runs scored through final checkpoint minus horizon. First alert determines timeliness.",
        label_sensitivity="Retrospective finalists refitted and evaluated without flagged or truncated positive ratings.",
        horizons=results,
    ))
    print(args.output)


if __name__ == "__main__":
    main()
