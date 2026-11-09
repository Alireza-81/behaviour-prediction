"""Summarize the matched input comparison, keeping historical-label caveats visible."""
import argparse
import csv
import json
from pathlib import Path


def main(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root/"results"
    report = json.loads((out/"report.json").read_text())
    before = json.loads((out/"predictions-before-evaluation.json").read_text())
    after = json.loads((out/"evaluation-input-predictions.json").read_text())
    scored = json.loads((out/"scored-predictions.json").read_text())
    replay = json.loads((root/"replay-audit/report.json").read_text())
    key = lambda r: tuple(r[k] for k in ("run", "task", "tag", "method", "input_step", "target_step"))
    lookup = {key(r): r for r in before}
    evaluation = [r for r in scored if r["suite"] == "evaluation"
                  and r["task"] == "behaviours" and r["method"] == "summary42"]
    correct_before = sum(lookup[key(r)]["predicted_label"] == r["label"] for r in evaluation)
    correct_after = sum(r["predicted_label"] == r["label"] for r in evaluation)
    labels = {key(r): r["label"] for r in evaluation}
    changes = [dict(run=r["run"], step=r["target_step"], behaviour=r["tag"],
                    monitoring_input_prediction=lookup[key(r)]["predicted_label"],
                    evaluation_input_prediction=r["predicted_label"],
                    monitoring_input_score=lookup[key(r)]["score"], evaluation_input_score=r["score"],
                    historical_evaluation_label=labels.get(key(r), "unknown"))
               for r in after if r["task"] == "behaviours" and r["method"] == "summary42"
               and lookup[key(r)]["predicted_label"] != r["predicted_label"]]
    with (out/"changed-behaviour-predictions.csv").open("w", newline="") as handle:
        fields = list(changes[0]) if changes else ["run", "step", "behaviour"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)
    metrics = report["evaluation"]["behaviours"]
    conditions = ["Monitoring inputs\nsame fitted monitor", "Evaluation inputs\nsame fitted monitor",
                  "Overall drift\nweaker regularization", "Training frequency", "Always present", "Always absent"]
    counts = [(correct_before, len(evaluation)), (correct_after, len(evaluation))]
    counts += [(metrics[m]["tp"]+metrics[m]["tn"], metrics[m]["checkpoints"])
               for m in ("drift_weaker_l2", "frequency", "always_present", "always_absent")]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    values = [100*a/b for a,b in counts]
    bars = ax.bar(conditions, values, color=["#8095aa", "#2166ac"]+["#9aa4af"]*4)
    for bar, value, (a,b) in zip(bars, values, counts):
        ax.text(bar.get_x()+bar.get_width()/2, value+1, f"{value:.1f}%\n{a}/{b}", ha="center", fontsize=9)
    ax.set(ylim=(0, 100), ylabel="Behaviour presence/absence accuracy (%)",
           title="Separate evaluation prompts: matched comparison\nPROVISIONAL — historical labels reused for repeated training runs")
    ax.tick_params(axis="x", labelsize=9)
    fig.savefig(out/"evaluation_input_comparison.png", dpi=170)
    plt.close(fig)
    m = metrics["summary42"]
    text = f"""# Prompt-transfer monitor results — provisional

The fitted monitor uses 42 summaries of monitoring-prompt activations at layers 15 and 31.
Each run is tested with a monitor trained on the other six runs only. Training-fold scaling,
classifier weights and the 0.5 threshold stay fixed when evaluation activations replace
monitoring activations. Evaluation answers and labels do not train this monitor.

## Matched comparison on the same historical evaluation labels

| Input / control | Correct behaviour decisions | Accuracy |
|---|---:|---:|
"""
    for condition, (a,b) in zip(conditions, counts):
        label = condition.replace("\n", ": ")
        text += f"| {label} | {a}/{b} | {100*a/b:.1f}% |\n"
    text += f"""
The different prompts change {len(changes)}/420 hard behaviour predictions; 15 of those
changes have known evaluation labels. The matched accuracy improves from
{correct_before}/{len(evaluation)} to {correct_after}/{len(evaluation)}. This is descriptive,
not evidence of a statistically established improvement. Unknown outcomes are excluded,
so the accuracy denominator is 391, rather than all 420 tag/checkpoint decisions.

The evaluation monitor finds {m['tp']}/{m['tp']+m['fn']} present cases, but correctly rejects
only {m['tn']}/{m['tn']+m['fp']} absent cases. It still overpredicts behaviour presence.
The overall-drift control with weaker regularization performs slightly better.

For binary harmfulness, every nonzero monitoring checkpoint is positive. The fitted
classifiers therefore reduce to constant-positive predictions. On evaluation they get
29/35 correct, exactly the always-present control; they do not distinguish safe checkpoints.
The next-checkpoint binary task has the same issue (24/28, matching always-present).

## Replay limitation

All {replay['snapshots_verified']} snapshots passed shape, prompt identity, finite-value,
step and recipe checks. Baseline captures match exactly. Later captures do not:
only {replay['identical_layer_tensors']}/{replay['layer_tensors_compared']} compared layer tensors
are identical, and the maximum relative L2 difference is {100*replay['maximum_relative_l2']:.1f}%.
Relative L2 here is ||repeated - original|| / ||original|| over the entire snapshot matrix.
The reason for the divergence is not established by this audit.

As requested, no responses were regenerated. Ground truth is borrowed from historical
models with the same training recipe. Consequently, these results are exploratory
compatibility measurements, not validated accuracy for the repeated models. New generation
and labelling from these repeated models would be needed to remove this limitation.

Independent calculation checks passed: held-out runs never enter training, normalization
uses training data only, and both sets of predictions reproduce from saved parameters.

## Saved artifacts

- behaviour_presence_monitoring.png and behaviour_presence_evaluation.png: observed/predicted maps.
- evaluation_input_comparison.png: matched comparison and simple controls.
- changed-behaviour-predictions.csv: exactly which predictions changed.
- report.json, protocol.json and data-audit.json: metrics, settings and verification.
- ../replay-audit/: comparison against historical activation snapshots.

Reproduce with scripts/audit_qwen_prompt_replay.py, scripts/run_qwen_full_experiments.py,
scripts/audit_qwen_full_experiments.py and scripts/summarize_qwen_prompt_transfer.py.
Full GPU and analysis commands are in docs/qwen-prompt-transfer.md.
"""
    (out/"interpretation.md").write_text(text)
    print(f"Saved interpretation, comparison plot and {len(changes)} changed decisions to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/qwen-prompt-transfer"))
    main(parser.parse_args().root)
