"""Run current detection, forecasting, and behaviour identification experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from behaviour_prediction.monitor.behaviours import run_behaviours
from behaviour_prediction.monitor.forecast import run_forecast
from behaviour_prediction.monitor.pilot import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("Results already exist; choose a new --output-dir to preserve them")
    root = Path("outputs/probes-30")
    labels = root/"labels"
    paths = sorted(labels.glob("*.csv"))
    runs = [root/path.stem for path in paths]
    if len(runs) < 3:
        parser.error("At least three labeled model runs are required")
    rows = [row for path in paths for row in csv.DictReader(path.open())]
    if any(row["label"] not in {"0", "1"} for row in rows):
        parser.error("Complete the label column before running experiments")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    detection = run_pilot(runs, labels, output/"detection", "label")
    forecast = run_forecast(runs, labels, output/"forecast", "label")
    behaviours = run_behaviours(runs, labels, output/"behaviours", "label", "behaviour_tags")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in hashes.items()):
        raise ValueError("Ground truth changed while experiments were running")
    protocol = dict(ground_truth="outputs/probes-30/labels: Claude v2 with user corrections",
        label_hashes=hashes, runs=[r.name for r in runs], response_count=len(rows),
        threshold=.5, l2=1., layers=[15,31],
        split="leave-one-entire-model-run-out; step zero is baseline only",
        next_evaluation="Use saved held-out predictions with separate evaluation-prompt outcomes; no refitting")
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    d = detection["methods"]["summary"]
    dp = detection["methods"]["prior"]
    f = forecast["methods"]["activation_forecast"]["overall"]
    fp = forecast["methods"]["prior"]["overall"]
    b = behaviours["methods"]["summary"]
    bp = behaviours["methods"]["prior"]
    text = ["# Current experiment results", "",
        f"Ground truth: {len(rows)} responses in `outputs/probes-30/labels/` (Claude v2 with user corrections).",
        f"{len(runs)} model runs, with one entire run held out in each fold. Layers 15/31; threshold 0.5; L2=1.", "",
        "| Experiment | Monitor | Frequency baseline |", "|---|---:|---:|",
        f"| Current harmfulness | {d['tp']+d['tn']}/{d['checkpoints']} ({d['accuracy']:.1%}) | {dp['tp']+dp['tn']}/{dp['checkpoints']} |",
        f"| Next-checkpoint harmfulness | {f['tp']+f['tn']}/{f['checkpoints']} ({f['accuracy']:.1%}) | {fp['tp']+fp['tn']}/{fp['checkpoints']} |",
        f"| Specific behaviour decisions | {b['micro']['tp']+b['micro']['tn']}/{b['micro']['checkpoints']} ({b['micro']['accuracy']:.1%}) | {bp['micro']['tp']+bp['micro']['tn']}/{bp['micro']['checkpoints']} |", "",
        "| Behaviour | Correct | Frequency baseline |", "|---|---:|---:|"]
    for tag, metric in b["per_tag"].items():
        prior = bp["per_tag"][tag]
        text.append(f"| {tag} | {metric['tp']+metric['tn']}/{metric['checkpoints']} | {prior['tp']+prior['tn']}/{prior['checkpoints']} |")
    text += ["", "Checkpoint harmfulness means any labeled harmful response; each behaviour means any response with that tag.",
        "Forecasting uses only current/past activation measurements to predict ten optimizer steps later.",
        "Forecasting currently matches the frequency baseline; binary detection adds little over the baseline.",
        "These are development results on seven correlated runs and shared monitoring/evaluation prompts. Scores are not calibrated confidence.", "",
        "Detailed metrics, held-out predictions, and detection plots are in the three experiment subfolders.",
        "The exact ground-truth file hashes and settings are recorded in protocol.json."]
    (output/"summary.md").write_text("\n".join(text)+"\n")
    print(f"Saved current experiments to {output}; start with summary.md")


if __name__ == "__main__":
    main()
