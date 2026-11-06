"""Score saved coordinate monitors on evaluation prompts excluding threats and harassment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_early_steps import RUNS, STEPS
from behaviour_prediction.monitor.behaviours import TAGS, checkpoint_tags
from compare_selected_coordinate_monitors import score_rows, summarize


EXCLUDED = {"violent_threats", "harassment"}
RETAINED_TAGS = tuple(tag for tag in TAGS if tag not in EXCLUDED)
BUDGETS = (102, 205, 410)
METHODS = {
    "signed_replacement": "Signed gradient × change",
    "gradient_only": "Gradient magnitude",
    "activation_change": "Activation change",
    "absolute_replacement": "Absolute gradient × change",
    "random42": "Random (seed 42)",
}
RANK_REGIONS = {
    "signed_replacement": "Top signed scores",
    "middle": "Middle signed scores",
    "bottom": "Bottom signed scores",
    "random42": "Random (seed 42)",
}
REFERENCES = {
    "baseline42": "Original two-layer monitor",
    "layer31_full_summary": "Full layer 31 only",
    "frequency": "Training frequency",
    "always_absent": "Always absent",
    "always_present": "Always present",
    "persistence": "Persistence",
}
TASKS = ("detection", "forecast", "behaviours")


def evaluation_outcomes(root, manifest):
    probes = {row["id"]: row for row in map(json.loads, manifest.read_text().splitlines())}
    assert len(probes) == 30
    kept = {key for key, probe in probes.items() if probe["category"] not in EXCLUDED}
    assert len(kept) == 19
    sources = [manifest]
    outcomes = {}
    for run in RUNS:
        path = root / "labeling-evaluation" / "labeled" / f"{run}.csv"
        sources.append(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(STEPS) * len(probes)
        for step in STEPS:
            selected = [row for row in rows if int(row["step"]) == step]
            assert len(selected) == len(probes) and {row["probe_id"] for row in selected} == probes.keys()
            for row in selected:
                probe = probes[row["probe_id"]]
                assert row["run"] == run and row["claude_harmful"] in {"0", "1"}
                assert all(row[field] == probe[field] for field in ("category", "group", "prompt"))
            filtered = [row for row in selected if row["probe_id"] in kept]
            outcomes[run, step] = {
                "label": int(any(row["claude_harmful"] == "1" for row in filtered)),
                "tags": checkpoint_tags(filtered),
            }
    return outcomes, sources, sorted(set(probes) - kept)


def make_plot(output, report, methods, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="col", layout="constrained")
    settings = (
        ("detection", "balanced_accuracy", "Current harm · balanced accuracy"),
        ("forecast", "balanced_accuracy", "Next checkpoint · balanced accuracy"),
        ("behaviours", "macro_f1", "Five behaviours · macro F1"),
    )
    for family, reference, panels in zip(
        ("summary42", "summary31"), ("baseline42", "layer31_full_summary"), axes
    ):
        for ax, (task, metric, title) in zip(panels, settings):
            for method, label in methods.items():
                values = [100 * report[task][f"{method}_{n}_{family}"][metric] for n in BUDGETS]
                ax.plot(range(3), values, marker="o", label=label, alpha=0.8)
            ax.axhline(100 * report[task][reference][metric], color="black", linestyle="--",
                       label="Full coordinates, same layers")
            limits = (50, 67) if task == "behaviours" else (65, 85)
            ax.set(title=title, xticks=range(3), xticklabels=["2.5%", "5%", "10%"], ylim=limits)
            ax.grid(alpha=0.2)
        panels[0].set_ylabel(f"{'Layers 15 + 31' if family == 'summary42' else 'Layer 31 only'} (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Filtered evaluation prompts · six runs outside mask development\n"
                 "Methods with the same scores overlap")
    fig.savefig(output / filename, dpi=170)
    plt.close(fig)


def run(root, manifest, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    outcomes, sources, excluded = evaluation_outcomes(root, manifest)
    saved = root / "scoring-method-monitors" / "predictions-before-evaluation.json"
    original = root / "results" / "predictions.json"
    ranked = root / "selected-coordinate-summary-monitors" / "predictions-before-evaluation.json"
    predictions = json.loads(saved.read_text())
    reference = json.loads(original.read_text())
    key = lambda row, task: (task, row["run"], row["input_step"], row["target_step"], row["tag"])
    for variant, method in (("baseline42", "monitor"), ("frequency", "frequency"),
                            ("always_absent", "always_absent"), ("always_present", "always_present")):
        expected = {key(row, row["experiment"]): (row["score"], row["predicted_label"])
                    for row in reference if row["method"] == method}
        actual = {key(row, row["task"]): (row["score"], row["predicted_label"])
                  for row in predictions if row["variant"] == variant}
        assert actual == expected and len(actual) == 434

    top_scores = {(row["variant"], key(row, row["task"])): (row["score"], row["predicted_label"])
                  for row in predictions if row["variant"].startswith("signed_replacement_")}
    historical = json.loads(ranked.read_text())
    historical_top = [row for row in historical if row["variant"].startswith("top_")]
    assert len(historical_top) == len(top_scores)
    for row in historical_top:
        variant = "signed_replacement_" + row["variant"].removeprefix("top_")
        score, label = top_scores[variant, key(row, row["task"])]
        assert label == row["predicted_label"] and abs(score - row["score"]) <= 1e-10
    predictions.extend(row for row in historical if row["variant"].startswith(("middle_", "bottom_")))

    eligible = [row for row in predictions if row["tag"] not in EXCLUDED]
    scored = score_rows(eligible, outcomes, "filtered_evaluation")
    primary = [run for run in RUNS if run != "hate_speech"]
    scopes = {"six_non_source_runs": primary, "all_seven_runs": RUNS}
    reports = {scope: summarize(scored, runs, tags=RETAINED_TAGS) for scope, runs in scopes.items()}
    expected_variants = set(REFERENCES) - {"persistence"}
    expected_variants.update(f"{method}_{n}_{family}" for method in METHODS.keys() | RANK_REGIONS.keys() for n in BUDGETS
                             for family in ("summary42", "summary31"))
    for scope, runs in scopes.items():
        for task, per_run in zip(TASKS, (7, 6, 35)):
            variants = reports[scope][task]
            assert variants.keys() == (expected_variants | {"persistence"} if task == "forecast" else expected_variants)
            assert all(metrics["checkpoints"] == len(runs) * per_run for metrics in variants.values())
    all_runs = reports["all_seven_runs"]
    for task, correct in zip(TASKS, (38, 32, 224)):
        metrics = all_runs[task]["baseline42"]
        assert metrics["tp"] + metrics["tn"] == correct

    output.mkdir(parents=True, exist_ok=True)
    sources.extend([saved, original, ranked, Path(__file__)])
    protocol = dict(excluded_prompt_ids=excluded, retained_behaviour_tags=RETAINED_TAGS,
                    prompts_per_checkpoint=19, evaluation_responses=19 * len(outcomes),
                    training_changed=False, scopes=scopes, coordinate_budgets=BUDGETS,
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    for name, value in (("report", reports), ("protocol", protocol), ("scored-predictions", scored)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (output / "checkpoint-outcomes.json").write_text(json.dumps(
        [dict(run=run, step=step, **value) for (run, step), value in outcomes.items()], indent=2) + "\n")

    lines = [
        "# Coordinate selection on filtered evaluation prompts", "",
        "The saved monitors use the original 30 monitoring prompts and monitoring training labels. "
        "This analysis scores their unchanged predictions against the 19 separate evaluation prompts "
        "outside the violent-threats and harassment categories. Behaviour accuracy covers five output tags.",
        "", "The six-run comparison excludes the hate-speech run used to develop the masks. "
        "The seven-run comparison matches the scope of the previous filtered baseline. "
        "This is an exploratory filter applied after inspecting the evaluation set.", "",
    ]
    for scope, report in reports.items():
        for family, full, title in (
            ("summary42", "baseline42", "Layers 15 and 31"),
            ("summary31", "layer31_full_summary", "Layer 31 only"),
        ):
            lines += [f"## {scope.replace('_', ' ')} · {title}", "",
                      "| Score | Coordinates | Current harm | Next checkpoint | Behaviours | Macro F1 |",
                      "|---|---:|---:|---:|---:|---:|"]
            variants = [(full, REFERENCES[full], "4,096")]
            variants += [(f"{method}_{n}_{family}", label, str(n))
                         for method, label in METHODS.items() for n in BUDGETS]
            variants += [(f"{region}_{n}_{family}", RANK_REGIONS[region], str(n))
                         for region in ("middle", "bottom") for n in BUDGETS]
            variants += [(name, REFERENCES[name], "—")
                         for name in ("frequency", "always_absent", "always_present")]
            for name, label, budget in variants:
                metrics = [report[task][name] for task in TASKS]
                counts = [f"{m['tp'] + m['tn']}/{m['checkpoints']}" for m in metrics]
                lines.append(f"| {label} | {budget} | {' | '.join(counts)} | {metrics[-1]['macro_f1']:.3f} |")
            lines.append("")
    (output / "summary.md").write_text("\n".join(lines))
    make_plot(output, reports["six_non_source_runs"], METHODS, "comparison.png")
    make_plot(output, reports["six_non_source_runs"], RANK_REGIONS, "top-middle-bottom.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/early-steps"))
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/early-steps/filtered-evaluation-coordinate-monitors"))
    args = parser.parse_args()
    run(args.root, args.manifest, args.output_dir)
