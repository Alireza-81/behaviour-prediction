"""Audit the full Qwen labels and plot response-level rates without fitting monitors."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from audit_qwen_full_reviews import main as audit_reviews
from label_with_claude import (
    apply_label_overrides, digest, export_row, payload, read_saved_results, response_key,
)

ROOT = Path("outputs/qwen-full-v2")
STEPS = (0, 10, 20, 50, 100, 200)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit_reviews()
    output = ROOT / "label-analysis"
    output.mkdir(exist_ok=True)
    all_rows, summaries, missing, stats = [], [], [], {}
    for suite in ("monitoring", "evaluation"):
        folder = ROOT / f"labeling-{suite}"
        manifest = json.loads((folder / "test.json").read_text())
        assert digest({k: v for k, v in manifest.items() if k != "test_id"}) == manifest["test_id"]
        assert manifest["rubric"] == Path("configs/claude_judge_expanded_v2.txt").read_text()
        results = read_saved_results(folder, manifest)
        assert len(results) == len(manifest["sample"]) == 4752
        with (folder / "comparison.csv").open(newline="") as handle:
            exported = {response_key(r): r for r in csv.DictReader(handle)}
        current = []
        for file in sorted((folder / "labeled").glob("*.csv")):
            with file.open(newline="") as handle:
                current.extend(csv.DictReader(handle))
        indexed = {response_key(r): r for r in current}
        assert len(current) == len(indexed) == 4752
        for index, sample in enumerate(manifest["sample"]):
            row = indexed[response_key(sample)]
            expected = export_row(sample, results[index])
            apply_label_overrides(folder, [expected])
            assert payload(row) == payload(sample)
            for key, value in expected.items():
                if key.startswith("claude_"):
                    assert str(value) == row[key] == exported[response_key(sample)][key]
        for run in sorted({r["run"] for r in current}):
            part = [r for r in current if r["run"] == run]
            base_positive = {r["probe_id"] for r in part if r["step"] == "0" and r["claude_harmful"] == "1"}
            base_negative = {r["probe_id"] for r in part if r["step"] == "0" and r["claude_harmful"] == "0"}
            absent = ROOT / run / "run_metadata.json"
            if not absent.exists():
                missing.append(str(absent))
            for step in STEPS:
                snapshot = ROOT / run / "activation_monitoring" / f"step-{step:06d}.pt"
                if suite == "monitoring" and not snapshot.exists():
                    missing.append(str(snapshot))
                selected = [r for r in part if int(r["step"]) == step]
                assert len(selected) == 66
                counts = Counter(r["claude_harmful"] for r in selected)
                positive, negative, unknown = counts["1"], counts["0"], counts[""]
                summaries.append(dict(
                    suite=suite, run=run, step=step, positive=positive, negative=negative,
                    unresolved=unknown, rate=positive/(positive+negative),
                    lower_rate=positive/66, upper_rate=(positive+unknown)/66,
                    new_positive_on_baseline_negative=sum(
                        r["claude_harmful"] == "1" and r["probe_id"] in base_negative for r in selected),
                    baseline_positive_prompts=len(base_positive),
                    benign_positive=sum(r["group"] == "benign" and r["claude_harmful"] == "1" for r in selected),
                ))
        stats[suite] = dict(responses=len(current), labels=dict(Counter(r["claude_harmful"] for r in current)),
                            statuses=dict(Counter(r["claude_status"] for r in current)),
                            spent_usd=sum(r["cost_usd"] for r in results.values()))
        all_rows.extend(dict(suite=suite, **r) for r in current)
    duplicate_groups = defaultdict(list)
    for row in all_rows:
        duplicate_groups[digest(payload(row))].append(row)
    conflicting = [rows for rows in duplicate_groups.values()
                   if len({r["claude_harmful"] for r in rows} - {""}) > 1]
    conflict_rows = [dict(conflict_group=i, **r) for i, rows in enumerate(conflicting, 1) for r in rows]
    unresolved = [r for r in all_rows if r["claude_harmful"] == ""]
    save_csv(output / "checkpoint_rates.csv", summaries)
    save_csv(output / "unresolved_labels.csv", unresolved)
    save_csv(output / "conflicting_labels.csv", conflict_rows)
    report = dict(suites=stats, conflicting_payload_groups=len(conflicting),
                  conflict_rows=len(conflict_rows), missing_activation_files=sorted(set(missing)),
                  labels_modified=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    runs = sorted({r["run"] for r in summaries})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained", sharey=True)
    for ax, suite in zip(axes, ("monitoring", "evaluation")):
        matrix = np.array([[next(r["rate"]*100 for r in summaries
                                 if r["run"] == run and r["step"] == step and r["suite"] == suite)
                            for step in STEPS] for run in runs])
        im = ax.imshow(matrix, vmin=0, vmax=60, cmap="YlOrRd", aspect="auto")
        for y in range(len(runs)):
            for x in range(len(STEPS)):
                ax.text(x, y, f"{matrix[y,x]:.0f}%", ha="center", va="center", fontsize=8)
        ax.set(xticks=range(len(STEPS)), xticklabels=STEPS, yticks=range(len(runs)),
               yticklabels=runs, title=suite.capitalize(), xlabel="Optimizer step")
    fig.colorbar(im, ax=axes, label="Positive answers among decided labels (%)")
    fig.suptitle("Qwen v2 · provisional Claude response labels")
    fig.savefig(output / "harmfulness_rates.png", dpi=180)
    plt.close(fig)
    lines = ["# Full Qwen v2 label audit", "",
             "All 9,504 response records match their prompt manifests and exported labels. "
             "Labels remain provisional and have not been modified.", "",
             "| Suite | Positive | Negative | Unresolved |",
             "|---|---:|---:|---:|"]
    for suite, values in stats.items():
        c = values["labels"]
        lines.append(f"| {suite} | {c.get('1',0)} | {c.get('0',0)} | {c.get('',0)} |")
    lines += ["", f"Conflicting identical prompt/answer payloads: {len(conflicting)} groups, {len(conflict_rows)} rows.",
              "See conflicting_labels.csv and unresolved_labels.csv for the exact answers.",
              "", f"Missing activation/metadata files: {len(set(missing))}.",
              "", "The response-rate plot uses each checkpoint's decided response denominator. "
              "checkpoint_rates.csv also contains lower/upper rates allowing unresolved labels "
              "to be negative/positive, and counts of new positives on baseline-negative prompts.",
              "", "Monitoring step zero has one weapon-creation positive in every run. "
              "The evaluation baseline organized-crime answer has inconsistent labels across runs. "
              "A binary any-positive target therefore requires care before claiming early warning."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
