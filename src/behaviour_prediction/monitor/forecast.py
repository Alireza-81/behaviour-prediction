"""Exploratory next-checkpoint forecasting with whole-run holdouts."""
import json
from pathlib import Path

import torch

from .pilot import fit_fold, metrics, paired_data


def make_pairs(outcomes, horizon=10):
    """Indices of nonzero inputs and exactly horizon-later targets in one run."""
    indexed = {(r["run"], r["step"]): i for i, r in enumerate(outcomes)}
    return [(i, indexed[(r["run"], r["step"] + horizon)])
            for i, r in enumerate(outcomes)
            if r["step"] > 0 and (r["run"], r["step"] + horizon) in indexed]


def run_forecast(runs, review_dir, output_dir, label_column="human_harmful"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output is not empty: {output_dir}")
    dataset, outcomes, matrices, sources = paired_data(runs, review_dir, label_column)
    pairs = make_pairs(outcomes)
    x = matrices["summary"]
    predictions = []
    single_class_folds = []
    for held_out in sorted({r["run"] for r in outcomes}):
        train = [(i, j) for i, j in pairs if outcomes[i]["run"] != held_out]
        test = [(i, j) for i, j in pairs if outcomes[i]["run"] == held_out]
        labels = torch.tensor([outcomes[j]["label"] for i, j in train], dtype=torch.float64)
        if not train or not test:
            raise ValueError(f"Insufficient train/test data for {held_out}")
        if len(set(labels.tolist())) == 1:
            single_class_folds.append(held_out)
            scores = [labels.mean().item()]*len(test)
        else:
            scores = fit_fold(x[[i for i, j in train]], x[[i for i, j in test]], labels)
        for (i, j), score in zip(test, scores):
            current, future = outcomes[i], outcomes[j]
            base = dict(run=held_out, input_step=current["step"], target_step=future["step"],
                        current_label=current["label"], label=future["label"])
            predictions.append(dict(**base, method="activation_forecast", score=score,
                                    predicted_label=int(score >= .5)))
            predictions.append(dict(**base, method="persistence", score=float(current["label"]),
                                    predicted_label=current["label"]))
            prior = labels.mean().item()
            predictions.append(dict(**base, method="prior", score=prior, predicted_label=int(prior >= .5)))
    report = dict(horizon_optimizer_steps=10, split="leave-one-entire-run-out",
                  label_source=label_column,
                  single_class_training_folds=single_class_folds,
                  single_class_policy="Use the training class frequency when only one training class exists; no fitted classifier in that fold",
                  threshold=.5, l2=1., confidence_status="uncalibrated_exploratory",
                  independent_runs=len(runs), sources=sources,
                  feature_names=dataset["schema"]["features"],
                  protocol="Use input checkpoint activation summaries only; predict next checkpoint any-harm label. "
                           "Current ground-truth labels are used only by persistence and for audit subsets.",
                  methods={})
    for method in ("activation_forecast", "persistence", "prior"):
        rows = [r for r in predictions if r["method"] == method]
        safe = [r for r in rows if r["current_label"] == 0]
        onsets = [r for r in safe if r["label"] == 1]
        report["methods"][method] = dict(overall=metrics(rows), currently_nonharmful=metrics(safe),
            onset_opportunities=len(onsets), early_warnings=sum(r["predicted_label"] for r in onsets),
            false_warnings=sum(r["predicted_label"] for r in safe if r["label"] == 0),
            onset_predictions=onsets,
            errors=[r for r in rows if r["predicted_label"] != r["label"]])
    report["baseline_to_first_checkpoint"] = [dict(run=r["run"], target_step=10)
        for r in outcomes if r["step"] == 10 and r["label"] == 1]
    report["limitations"] = [
        "Step zero excluded: shared base model has no fine-tuning trajectory signal; see baseline_to_first_checkpoint for harm already present at step 10.",
        "Few independent runs, correlated checkpoints, and very few observable post-baseline onsets.",
        "Same monitoring prompts provide inputs and outcome labels; no held-out-prompt validation.",
        "Any-positive label means observed harm on this probe suite, not general model safety.",
        "Fixed settings inherited from the development pilot; no threshold tuning on forecast results."]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, allow_nan=False) + "\n")
    return report
