"""Score a single run from two saved activation checkpoint feature files."""
import argparse
import json
from pathlib import Path
from statistics import mean

from evaluate_monitor_study import GEOMETRY
from fit_paired_activation_forecast import MLP_METHOD, forecast_scores
from stage_monitor_study import load


def activation_features(path, monitor_ids):
    capture = load(path)
    ids = capture["probe_ids"]
    if len(ids) != len(set(ids)) or not set(monitor_ids) <= set(ids):
        raise ValueError(f"Incorrect or missing monitoring probe IDs: {path}")
    indices = [ids.index(probe) for probe in monitor_ids]
    contrast = capture["names"].index("request_contrast_candidate")
    features = {}
    for layer in ("14", "15", "31"):
        source = capture["features"][layer]
        features[f"L{layer}_magnitude"] = mean(source["relative_l2"][i] for i in indices)
        features[f"L{layer}_orientation"] = mean(1 - source["cosine"][i] for i in indices)
        features[f"L{layer}_projection_request_contrast_candidate"] = mean(
            source["relative_projection_delta"][i][contrast] for i in indices
        )
    return capture["step"], features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fold", choices=("career", "edu", "finance"), required=True,
                        help="Evaluation corpus of the chosen exploratory main-rotation model")
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--nonlinear", action="store_true", help="Score the residual MLP from a nonlinear fit report")
    args = parser.parse_args()
    if args.previous.resolve().parent != args.current.resolve().parent:
        parser.error("Both checkpoint files must come from the same branch")
    report = load(args.fit_report)
    fold, = [f for f in report["results"]["primary"]["folds"]
             if f["rotation"] == "main" and f["testing"] == args.fold]
    method = MLP_METHOD if args.nonlinear else "future_label_and_pair_rank"
    if method not in fold["models"]:
        parser.error(f"Fit report does not contain {method}")
    model = fold["models"][method]
    monitor = [r["id"] for r in load(Path("configs/safety-loop/panel-v1.json")) if r["split"] == "monitor"]
    previous_step, previous = activation_features(args.previous, monitor)
    step, features = activation_features(args.current, monitor)
    if step - previous_step != 5:
        parser.error("Use adjacent checkpoints five optimizer steps apart")
    features.update({key + "_slope": (features[key] - previous[key]) / 5 for key in GEOMETRY})
    score, = forecast_scores([features], model)
    print(json.dumps(dict(
        checkpoint=step, forecast_steps=report["horizon_optimizer_steps"], score=score,
        alert_cutoff=model["cutoff"], alert=score > model["cutoff"],
        fit_corpus=fold["training"], calibration_corpus=fold["calibration"],
        status="development prototype; score is uncalibrated",
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
