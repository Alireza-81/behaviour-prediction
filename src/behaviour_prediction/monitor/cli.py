"""Prepare labels, fit an experimental monitor, and score saved trajectories."""
import argparse
import csv
import json
from pathlib import Path

from .classifier import fit, predict, read_labels
from .features import extract_runs


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Export causal features and a blank checkpoint-label template")
    prepare.add_argument("--run-dir", nargs="+", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs/monitor-data"))
    train = commands.add_parser("fit", help="Fit, calibrate and audit on three disjoint groups of runs")
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("outputs/monitor/model.json"))
    train.add_argument("--l2", type=float, default=1.0)
    train.add_argument("--alert-threshold", type=float, default=0.9)
    train.add_argument("--target-group-reliability", type=float, default=0.9)
    train.add_argument("--min-alert-groups", type=int, default=30)
    score = commands.add_parser("predict", help="Score snapshots, preserving an uncertain outcome when evidence is weak")
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--run-dir", type=Path, nargs="+", required=True)
    score.add_argument("--output", type=Path, default=Path("outputs/monitor/predictions.json"))
    capture = commands.add_parser("capture", help="Recapture revised probes from saved models; requires CUDA")
    capture.add_argument("--run-dir", type=Path, nargs="+", required=True)
    capture.add_argument("--probe-path", type=Path, default=Path("data/probes/behavior_eval.jsonl"))
    capture.add_argument("--steps", type=int, nargs="+", help="Defaults to all actual saved checkpoints, plus base step 0")
    capture.add_argument("--layers", type=int, nargs="+", help="Defaults to each run's recorded layers")
    capture.add_argument("--max-length", type=int, default=256)
    capture.add_argument("--output-root", type=Path, required=True)
    responses = commands.add_parser("responses", help="Generate checkpoint responses for human review; requires CUDA")
    responses.add_argument("--run-dir", type=Path, required=True)
    responses.add_argument("--probe-path", type=Path, default=Path("data/probes/behavior_eval.jsonl"))
    responses.add_argument("--steps", type=int, nargs="+", help="Defaults to all saved checkpoints, plus step 0")
    responses.add_argument("--output", type=Path, required=True)
    responses.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            features_path = args.output_dir / "features.json"
            labels_path = args.output_dir / "labels.csv"
            if features_path.exists() or labels_path.exists():
                raise ValueError("Monitor data already exists; use a new output directory to preserve reviewed labels")
            dataset = extract_runs(args.run_dir)
            write_json(features_path, dataset)
            with labels_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run", "step", "label", "split", "group_id", "label_source"])
                writer.writeheader()
                writer.writerows({"run": row["run"], "step": row["step"]} for row in dataset["rows"])
            print(f"Saved {features_path} and blank {labels_path}; no behaviour labels were inferred")
        elif args.command == "fit":
            dataset = json.loads(args.features.read_text())
            labelled = read_labels(args.labels, dataset["rows"])
            model = fit(dataset, labelled, l2=args.l2, threshold=args.alert_threshold,
                        target_precision=args.target_group_reliability, min_alert_groups=args.min_alert_groups)
            write_json(args.output, model)
            write_json(args.output.with_name(args.output.stem + "-audit.json"), model["audit"])
            print(f"Saved {args.output}; confidence status: {model['confidence_status']}")
        elif args.command == "predict":
            model = json.loads(args.model.read_text())
            write_json(args.output, predict(model, extract_runs(args.run_dir)))
            print(f"Saved {args.output}")
        elif args.command == "capture":
            from .capture import capture_runs
            capture_runs(args.run_dir, args.probe_path, args.output_root, args.steps, args.layers, args.max_length)
        elif args.command == "responses":
            from .responses import generate_responses
            generate_responses(args.run_dir, args.probe_path, args.steps, args.output, args.max_new_tokens)
            print(f"Saved {args.output}; human_harmful is blank for review")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
