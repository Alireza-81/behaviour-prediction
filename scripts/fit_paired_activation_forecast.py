"""Fit a single-run forecaster using matched pairs only during training."""
import argparse
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_monitor_study import CONFIG, DYNAMICS, GEOMETRY, OUT, dataset, load, matrix, read_split, scores
from stage_monitor_study import digest, save_json

PAIR_WEIGHT = 1.0
L2 = 1.0
MLP_WIDTH = 8
MLP_L2 = 0.1
MLP_SEED = 401
MLP_METHOD = "future_label_and_pair_rank_mlp"
METHODS = {"future_label_only": 0.0, "future_label_and_pair_rank": PAIR_WEIGHT}
HARMLESSNESS = tuple(f"L{layer}_projection_harmlessness" for layer in ("14", "15", "31"))
REQUEST_PROJECTIONS = tuple(key for key in GEOMETRY if "_projection_" in key)
DIRECTION_METHODS = {
    "geometry_without_projections": tuple(key for key in GEOMETRY if key not in REQUEST_PROJECTIONS),
    "geometry_and_harmlessness": GEOMETRY + HARMLESSNESS,
    "request_and_harmlessness_projections_only": REQUEST_PROJECTIONS + HARMLESSNESS,
}


def with_slopes(columns):
    return columns + tuple(key + "_slope" for key in columns)


def discordant_pairs(rows):
    slots = defaultdict(dict)
    for i, row in enumerate(rows):
        corpus, seed, branch = row["run"].split("/")
        group = slots[corpus, seed, row["step"]]
        if branch in group:
            raise ValueError("Duplicate branch at a matched checkpoint")
        group[branch] = i
    pairs = []
    for group in slots.values():
        if set(group) != {"0", "1"}:
            continue
        a, b = (group[k] for k in ("0", "1"))
        if rows[a]["label"] != rows[b]["label"]:
            # The label determines the preferred ordering, not the branch.
            pairs.append((a, b) if rows[a]["label"] == 1 else (b, a))
    return pairs


def fit(rows, pair_weight, hidden_width=0, *, features=DYNAMICS):
    x = matrix(rows, features)
    y = matrix(rows, ("label",)).flatten()
    if y.unique().numel() != 2:
        raise ValueError("Fit data must contain both future-outcome labels")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    pairs = discordant_pairs(rows)
    if pair_weight and not pairs:
        raise ValueError("No matched checkpoints with different future-harm labels")
    indices = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    weight = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    bias = torch.zeros((), dtype=x.dtype, requires_grad=True)
    parameters = [weight, bias]
    if hidden_width:
        generator = torch.Generator().manual_seed(MLP_SEED)
        hidden = (torch.randn(x.shape[1], hidden_width, dtype=x.dtype, generator=generator)
                  / x.shape[1] ** .5).requires_grad_()
        hidden_bias = torch.zeros(hidden_width, dtype=x.dtype, requires_grad=True)
        output = torch.zeros(hidden_width, dtype=x.dtype, requires_grad=True)
        parameters.extend([hidden, hidden_bias, output])
    optimizer = torch.optim.LBFGS(parameters, max_iter=200 if hidden_width else 100,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logit = x @ weight + bias
        penalty = .5 * L2 * weight.square().sum()
        if hidden_width:
            logit = logit + torch.tanh(x @ hidden + hidden_bias) @ output
            penalty = penalty + .5 * MLP_L2 * sum(p.square().sum()
                                                   for p in (hidden, hidden_bias, output))
        loss = F.binary_cross_entropy_with_logits(logit, y) + penalty
        if pair_weight:
            difference = logit[indices[:, 0]] - logit[indices[:, 1]]
            loss = loss + pair_weight * F.softplus(-difference).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = dict(mean=mean.tolist(), scale=scale.tolist(),
                  weight=weight.detach().tolist(), bias=bias.item())
    if hidden_width:
        fitted.update(hidden=hidden.detach().tolist(), hidden_bias=hidden_bias.detach().tolist(),
                      output=output.detach().tolist())
    return {
        "features": features,
        "parameters": fitted,
        "fit_windows": len(rows),
        "positive_fit_windows": int(y.sum().item()),
        "discordant_fit_pairs": len(pairs),
        "pair_weight": pair_weight,
    }


def online_rows(inputs, protocol, geometry=GEOMETRY):
    """Create features for every scoreable checkpoint without looking at ratings."""
    records = []
    last_scored = protocol["capture_steps"][-1] - protocol["forecast_steps"]
    for (corpus, seed), pair in inputs.items():
        for branch in (0, 1):
            points = [p for p in pair["records"] if p["branch"] == branch]
            if [p["step"] for p in points] != protocol["capture_steps"]:
                raise ValueError(f"Unexpected checkpoint schedule for {corpus}/{seed}/{branch}")
            for previous, point in zip(points, points[1:]):
                step = point["step"]
                if step > last_scored:
                    break
                dt = step - previous["step"]
                row = dict(corpus=corpus, run=f"{corpus}/{seed}/{branch}", step=step)
                for name in geometry:
                    row[name] = point["signals"][name]
                    row[name + "_slope"] = (row[name] - previous["signals"][name]) / dt
                records.append(row)
    return records


def forecast_scores(rows, model):
    p = model["parameters"]
    if "hidden" not in p:
        return scores(rows, model)
    x = matrix(rows, model["features"])
    z = (x - x.new_tensor(p["mean"])) / x.new_tensor(p["scale"])
    logits = z @ x.new_tensor(p["weight"]) + p["bias"]
    logits += torch.tanh(z @ x.new_tensor(p["hidden"]) + x.new_tensor(p["hidden_bias"])) @ x.new_tensor(p["output"])
    return torch.sigmoid(logits).tolist()


def score_runs(rows, model):
    predictions = defaultdict(list)
    for row, score in zip(rows, forecast_scores(rows, model), strict=True):
        predictions[row["run"]].append(dict(step=row["step"], score=score))
    return predictions


def calibrate(predictions, outcomes):
    no_onset = [max(p["score"] for p in trajectory)
                for run, trajectory in predictions.items()
                if outcomes[run]["earliest_possible"] is None]
    if not no_onset:
        raise ValueError("Calibration corpus has no unambiguous no-onset runs")
    # The threshold is calibrated on max run score because one alert is
    # enough to make a run a false alarm, regardless of its length in windows.
    return max(no_onset), len(no_onset)


def report_runs(predictions, outcomes, cutoff, horizon):
    result = {}
    for run, trajectory in predictions.items():
        first = next((p["step"] for p in trajectory if p["score"] > cutoff), None)
        onset = outcomes[run]["onset"]
        earliest_possible = outcomes[run]["earliest_possible"]
        lead = onset - first if onset is not None and first is not None else None
        timely = lead is not None and 0 < lead <= horizon
        result[run] = dict(
            onset=onset,
            earliest_possible_onset=earliest_possible,
            first_alert=first,
            timely=timely if onset is not None else None,
            lead_steps=lead if timely else None,
            false_alert=first is not None if earliest_possible is None else None,
            scores=trajectory,
        )
    return result


def totals(runs):
    return {
        "timely_onset_runs": sum(r["timely"] is True for r in runs),
        "onset_runs": sum(r["timely"] is not None for r in runs),
        "false_alert_runs": sum(r["false_alert"] is True for r in runs),
        "no_onset_runs": sum(r["false_alert"] is not None for r in runs),
        "lead_steps": sorted(r["lead_steps"] for r in runs if r["lead_steps"] is not None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--nonlinear", action="store_true", help="Add an eight-unit residual MLP")
    options.add_argument("--direction-ablation", action="store_true",
                         help="Compare fixed candidate direction sets using paired logistic fits")
    args = parser.parse_args()
    filename = "paired-loss-development.json"
    if args.nonlinear:
        filename = "nonlinear-activation-development.json"
    elif args.direction_ablation:
        filename = "direction-ablation-development.json"
    output_path = args.output or OUT / "report" / filename
    methods = METHODS | ({MLP_METHOD: PAIR_WEIGHT} if args.nonlinear else {})
    if args.direction_ablation:
        methods = {"future_label_and_pair_rank": PAIR_WEIGHT} | dict.fromkeys(DIRECTION_METHODS, PAIR_WEIGHT)
    protocol = load(CONFIG / "protocol.json")
    inputs = read_split(protocol, "development", set())
    online = online_rows(inputs, protocol, GEOMETRY + HARMLESSNESS if args.direction_ablation else GEOMETRY)
    observed_features = {(row["run"], row["step"]): row for row in online}
    corpora = sorted({row["corpus"] for row in online})
    if len(corpora) != 3:
        raise ValueError("This three-way exploratory design requires three development corpora")
    results = {}
    rotations = [tuple(corpora[i:] + corpora[:i]) for i in range(3)]
    for variant, sensitivity in (("primary", False), ("exclude_flagged_positives", True)):
        supervised, outcomes = dataset(inputs, protocol, sensitivity)
        if args.direction_ablation:
            supervised = [observed_features[row["run"], row["step"]] | {"label": row["label"]}
                          for row in supervised]
        folds = []
        for training, calibration, testing in permutations(corpora):
            models = {}
            training_rows = [r for r in supervised if r["corpus"] == training]
            calibration_rows = [r for r in online if r["corpus"] == calibration]
            testing_rows = [r for r in online if r["corpus"] == testing]
            for name, pair_weight in methods.items():
                columns = with_slopes(DIRECTION_METHODS[name]) if name in DIRECTION_METHODS else DYNAMICS
                fitted = fit(training_rows, pair_weight, MLP_WIDTH if name == MLP_METHOD else 0,
                             features=columns)
                cutoff, controls = calibrate(score_runs(calibration_rows, fitted), outcomes)
                observed = report_runs(score_runs(testing_rows, fitted), outcomes, cutoff, protocol["forecast_steps"])
                models[name] = dict(fitted, cutoff=cutoff, calibration_no_onset_runs=controls,
                                    summary=totals(list(observed.values())), evaluation_runs=observed)
            folds.append(dict(training=training, calibration=calibration, testing=testing,
                              rotation="main" if (training, calibration, testing) in rotations else "reverse",
                              models=models))
        results[variant] = {"folds": folds}
        for rotation in ("main", "reverse"):
            results[variant][rotation] = {
                name: totals([run for fold in folds if fold["rotation"] == rotation
                              for run in fold["models"][name]["evaluation_runs"].values()])
                for name in methods
            }
            print(variant, rotation, results[variant][rotation])
    report = {
        "status": "Exploratory three-way corpus rotation within development",
        "horizon_optimizer_steps": protocol["forecast_steps"],
        "last_scored_step": protocol["capture_steps"][-1] - protocol["forecast_steps"],
        "objective": "Mean future-label BCE plus 0.5 times squared L2 norm plus weighted mean softplus of negative positive-minus-negative logit on discordant matches.",
        "features": "Single trajectory; 9 baseline-relative activation summaries and their 9 rates of change. Pairs are used only in the training loss.",
        "cutoff_policy": "Strictly above the highest score on any no-onset run in the calibration corpus. No observed calibration false alarms, not a population guarantee.",
        "evaluation_policy": "Score all eligible checkpoint times up to the fixed follow-up horizon without labels. Evaluate first alert per run. Early out-of-horizon and late alarms do not count as timely.",
        "limitations": "Each rotation scores the same nine onset and nine no-onset runs once. Rotations are not independent. All onsets are at step 20. Previously held-out corpus data were not loaded.",
        "source_sha256": {
            f"{corpus}-{seed}": digest((OUT / "summaries/development" / f"{corpus}-{seed}.json").read_bytes())
            for corpus, seed in inputs
        },
        "results": results,
    }
    if args.nonlinear:
        report["nonlinear_model"] = {
            "architecture": "Linear logit plus eight trainable tanh units and an output weight per unit.",
            "hidden_width": MLP_WIDTH,
            "hidden_seed": MLP_SEED,
            "hidden_l2_coefficient": MLP_L2,
            "linear_l2_coefficient": L2,
            "comparison": "Same supervision, checkpoint features, corpus rotations, and cutoff policy as the paired linear model.",
        }
    if args.direction_ablation:
        report["features"] = "Single-run magnitudes, orientation, and selected projections at layers 14/15/31, plus their rates of change. No response text or label is an evaluation input."
        report["direction_candidates"] = {
            "request_contrast_candidate": "Fixed risky-request versus matched-benign contrast; not validated against refusal behavior.",
            "harmlessness": "Fixed system-message contrast; not validated against harmful outputs.",
        }
    save_json(output_path, report)


if __name__ == "__main__":
    main()
