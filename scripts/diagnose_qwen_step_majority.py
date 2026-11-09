"""Compare activation predictions with a same-step training-run majority lookup."""
import argparse
import json
from pathlib import Path
from run_qwen_full_experiments import load_outcomes
from run_early_step_experiments import metric

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-full-v2/results"))
root = parser.parse_args().output_dir
protocol = json.loads((root / "protocol.json").read_text())
predictions = [r for r in json.loads((root / "predictions-before-evaluation.json").read_text())
               if r["task"] == "behaviours" and r["method"] == "summary42"]
monitor, _ = load_outcomes(protocol["runs"], "monitoring")
report = {}
for suite in ("monitoring", "evaluation"):
    outcomes, _ = load_outcomes(protocol["runs"], suite)
    rows = []
    for prediction in predictions:
        run, step, tag = [prediction[k] for k in ("run", "target_step", "tag")]
        label = outcomes[run, step]["tags"][tag]
        if label is None:
            continue
        train = [monitor[other, step]["tags"][tag] for other in protocol["runs"] if other != run]
        known = [value for value in train if value is not None]
        assert known
        prior = sum(known) / len(known)
        rows.append(dict(prediction, suite=suite, label=label, step_majority_score=prior,
                         step_majority_prediction=int(prior >= .5)))
    baseline = [dict(r, score=r["step_majority_score"], predicted_label=r["step_majority_prediction"]) for r in rows]
    exceptions = [r for r in rows if r["label"] != r["step_majority_prediction"]]
    report[suite] = dict(
        monitor=metric(rows), step_majority=metric(baseline), minority_cases=len(exceptions),
        monitor_correct_on_minority=sum(r["label"] == r["predicted_label"] for r in exceptions),
        agreement=sum(r["predicted_label"] == r["step_majority_prediction"] for r in rows),
    )
print(json.dumps(report, indent=2))
(root / "step-majority-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
