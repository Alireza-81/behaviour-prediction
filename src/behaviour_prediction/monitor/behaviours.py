"""Exploratory multi-label behaviour identification from activation summaries."""
import csv
import json
from pathlib import Path

import torch

from .pilot import fit_fold, metrics, paired_data

TAGS = ("self_harm_encouragement", "self_harm_facilitation", "hate_speech",
        "violent_threats", "discrimination", "child_abuse", "harassment")


def checkpoint_tags(rows, label_column="claude_harmful", tag_column="claude_behaviour_tags"):
    """Unknown individual decisions do not establish absence of a behaviour."""
    observed = set()
    unknown = False
    for row in rows:
        if row[label_column] not in {"0", "1"}:
            unknown = True
        elif row[label_column] == "1":
            observed.update(filter(None, row[tag_column].split(";")))
    return {tag: 1 if tag in observed else None if unknown else 0 for tag in TAGS}


def run_behaviours(runs, review_dir, output_dir, label_column="claude_harmful", tag_column="claude_behaviour_tags"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose an empty behaviour experiment output directory")
    dataset, outcomes, matrices, sources = paired_data(runs, review_dir, label_column)
    labels = {}
    for run in runs:
        with (Path(review_dir) / f"{run.name}.csv").open(newline="") as handle:
            reviews = list(csv.DictReader(handle))
        for step in {r["step"] for r in reviews}:
            labels[(run.name, int(step))] = checkpoint_tags([r for r in reviews if r["step"] == step], label_column, tag_column)
    x = matrices["summary"]
    predictions, fallbacks = [], []
    for tag in TAGS:
        for held_out in sorted({r["run"] for r in outcomes}):
            usable = [i for i, r in enumerate(outcomes) if r["step"] > 0 and labels[(r["run"], r["step"])][tag] is not None]
            train = [i for i in usable if outcomes[i]["run"] != held_out]
            test = [i for i in usable if outcomes[i]["run"] == held_out]
            y = torch.tensor([labels[(outcomes[i]["run"], outcomes[i]["step"])][tag] for i in train], dtype=torch.float64)
            if not train:
                raise ValueError(f"No known training labels: {tag}, {held_out}")
            prior = y.mean().item()
            if len(set(y.tolist())) < 2:
                scores = [prior]*len(test)
                fallbacks.append(dict(tag=tag, held_out=held_out))
            elif test:
                scores = fit_fold(x[train], x[test], y)
            else:
                scores = []
            for i, score in zip(test, scores):
                row = outcomes[i]
                base = dict(run=row["run"], step=row["step"], tag=tag,
                            label=labels[(row["run"], row["step"])][tag])
                for method, value in (("summary", score), ("prior", prior)):
                    predictions.append(dict(**base, method=method, score=value, predicted_label=int(value >= .5)))
    report = dict(target="At least one response exhibiting each behaviour at the current checkpoint",
                  label_source=("Current ground truth: Claude v2 with user corrections" if label_column == "label"
                                else "Claude v2 tags; not independently human validated"),
                  split="leave-one-entire-run-out; step zero excluded", threshold=.5, l2=1.,
                  single_class_training_folds=fallbacks, sources=sources, methods={},
                  feature_names=dataset["schema"]["features"],
                  limitations=["Exploratory development results on seven correlated runs",
                    "Same prompts for activation measurement and output evaluation",
                    "Unresolved response means absent tags at that checkpoint are unknown and excluded",
                    "Single-class training folds use training frequency, not a fitted classifier",
                    "Scores are uncalibrated; predicted tags are not explanations of internal mechanisms"])
    for method in ("summary", "prior"):
        rows = [r for r in predictions if r["method"] == method]
        per_tag = {tag: metrics([r for r in rows if r["tag"] == tag]) for tag in TAGS}
        for tag, value in per_tag.items():
            denominator = 2*value["tp"] + value["fp"] + value["fn"]
            value["f1"] = 2*value["tp"]/denominator if denominator else 0.
        micro = metrics(rows)
        denominator = 2*micro["tp"] + micro["fp"] + micro["fn"]
        micro["f1"] = 2*micro["tp"]/denominator if denominator else 0.
        report["methods"][method] = dict(per_tag=per_tag, micro=micro,
                                        macro_f1=sum(v["f1"] for v in per_tag.values())/len(TAGS))
    output_dir.mkdir(parents=True)
    for name, value in (("report.json", report), ("predictions.json", predictions)):
        (output_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    return report
