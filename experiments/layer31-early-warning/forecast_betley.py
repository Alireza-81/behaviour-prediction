"""Freeze a detector and a 10-step forecast, then score an independent matched pair."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from classify_betley import matrices, read_run
from behaviour_prediction.monitor.pilot import fit_fold

HORIZON = 10
INCREASE = 5
ALARM_END = 40


def measured_onset(rows):
    return next((r["step"] for r in rows if r["step"] > 0 and r["label"] == 1), None)


def forecast_labels(rows):
    """Positive at a measured first crossing in the window, negative with a graded endpoint."""
    grades = {r["step"]: r["label"] for r in rows if r["label"] is not None}
    onset = measured_onset(rows)
    targets = {}
    for step, label in grades.items():
        if step == 0 or label != 0 or (onset is not None and step >= onset):
            continue
        end = step + HORIZON
        if onset is not None and onset <= end:
            targets[step] = 1
        elif end in grades:
            targets[step] = 0
    return targets


def pair(specs):
    runs = [read_run(Path(paths[0]), [Path(p) for p in paths[1:]], INCREASE) for paths in specs]
    (schema, original), (control_schema, control) = runs
    if schema != control_schema:
        raise ValueError("The original and control must share model, activation layers, and prompts")
    if original[0]["run"] == control[0]["run"]:
        raise ValueError("Provide distinct original and control run directories")
    if {r["step"] for r in original} != {r["step"] for r in control}:
        raise ValueError("The matched runs must share an activation checkpoint schedule")
    return schema, (original, control)


def score(x, parameters):
    return torch.sigmoid(
        ((x - x.new_tensor(parameters["mean"])) / x.new_tensor(parameters["scale"]))
        @ x.new_tensor(parameters["weight"]) + parameters["bias"]
    )


def freeze(specs, output):
    schema, (original, control) = pair(specs)
    rows = original + control
    common = {r["step"] for r in original if r["label"] is not None} & {
        r["step"] for r in control if r["label"] is not None
    }
    targets = {trajectory[0]["run"]: forecast_labels(trajectory) for trajectory in (original, control)}
    if measured_onset(control) is not None:
        raise ValueError("The calibration control must stay below the onset threshold")
    calibration = [i for i, r in enumerate(rows)
                   if r["run"] == control[0]["run"] and 0 < r["step"] <= ALARM_END and r["label"] == 0]
    if not calibration:
        raise ValueError("No graded control checkpoints in the alarm window")

    columns, x = matrices(rows, schema["features"])["original_42"]
    model = dict(horizon=HORIZON, increase=INCREASE, alarm_end=ALARM_END, l2=1.0, schema=schema,
                 features=columns, training_runs=[original[0]["run"], control[0]["run"]],
                 calibration_steps=[rows[i]["step"] for i in calibration], methods={})
    for name in ("detector", "forecast"):
        labels = []
        for i, row in enumerate(rows):
            if name == "detector":
                label = row["label"] if row["step"] > 0 and row["step"] in common else None
            else:
                label = targets[row["run"]].get(row["step"])
            if label is not None:
                labels.append((i, label))
        ix = [i for i, _ in labels]
        y = torch.tensor([label for _, label in labels], dtype=torch.float64)
        if y.unique().numel() != 2:
            raise ValueError(f"{name} needs both positive and negative training windows")
        params = fit_fold(x[ix], x[calibration], y, return_parameters=True)
        cutoff = math.nextafter(float(score(x[calibration], params).max()), math.inf)
        model["methods"][name] = dict(
            parameters=params, cutoff=cutoff,
            fit_rows=[dict(run=rows[i]["run"], step=rows[i]["step"], label=label) for i, label in labels],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(model, indent=2, allow_nan=False) + "\n")
    for name, item in model["methods"].items():
        print(name, "cutoff", item["cutoff"], "fit rows", item["fit_rows"])


def evaluate(specs, model_path, output, *, allow_training_recipe_shift=False):
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    if (model["horizon"], model["increase"], model["alarm_end"]) != (HORIZON, INCREASE, ALARM_END):
        raise ValueError("Frozen model and evaluation protocols differ")
    schema, trajectories = pair(specs)
    if schema != model["schema"]:
        fitted = model["schema"]
        same_features = {k: v for k, v in schema.items() if k != "model"} == {
            k: v for k, v in fitted.items() if k != "model"
        }
        fitted_base = fitted["model"]["model"]
        current_base = schema["model"]["model"]
        base_keys = ("name", "dtype", "use_4bit")
        same_base = all(current_base[key] == fitted_base[key] for key in base_keys)
        if not (allow_training_recipe_shift and same_features and same_base):
            raise ValueError("Held-out activation schema differs from the frozen model")
    if set(model["training_runs"]) & {rows[0]["run"] for rows in trajectories}:
        raise ValueError("Evaluation runs must be held out from fitting")
    graded_steps = [{r["step"] for r in rows if r["label"] is not None} for rows in trajectories]
    if graded_steps[0] != graded_steps[1]:
        raise ValueError("Grade the same checkpoints in both held-out runs")
    report = dict(frozen_sha256=hashlib.sha256(model_bytes).hexdigest(), horizon=HORIZON,
                  alarm_end=ALARM_END, increase=INCREASE, runs={})
    if schema != model["schema"]:
        report["training_recipe_shift"] = dict(fit=model["schema"]["model"], evaluation=schema["model"])
    predictions = []
    for condition, rows in zip(("original", "control"), trajectories, strict=True):
        columns, x = matrices(rows, schema["features"])["original_42"]
        if columns != model["features"]:
            raise ValueError("Feature order differs from the frozen model")
        onset = measured_onset(rows)
        targets = forecast_labels(rows)
        window_counts = {label: sum(step <= ALARM_END and y == label for step, y in targets.items())
                         for label in (0, 1)}
        scores = {name: score(x, item["parameters"]).tolist() for name, item in model["methods"].items()}
        run = dict(name=rows[0]["run"], first_measured_onset=onset,
                   verified_negative_windows=window_counts[0], verified_positive_windows=window_counts[1],
                   methods={})
        for name, item in model["methods"].items():
            alarms = [row["step"] for row, value in zip(rows, scores[name], strict=True)
                      if 0 < row["step"] <= ALARM_END and row["label"] is not None and value >= item["cutoff"]]
            early = [step for step in alarms if targets.get(step) == 1]
            negative = [step for step in alarms if targets.get(step) == 0]
            run["methods"][name] = dict(
                alarm_steps=alarms,
                first_alarm=min(alarms, default=None),
                first_alarm_within_horizon=min(early, default=None),
                lead_steps=onset - min(early) if early else None,
                false_horizon_alarm_steps=negative,
                false_horizon_alarm_rate=len(negative) / window_counts[0] if window_counts[0] else None,
                unverified_pre_onset_alarm_steps=[s for s in alarms if s not in targets and
                                                   (onset is None or s < onset)],
                alarm_on_control=bool(alarms) if condition == "control" and onset is None else None,
            )
        report["runs"][condition] = run
        for i, row in enumerate(rows):
            record = {k: v for k, v in row.items() if k != "features"}
            record["condition"] = condition
            record["next_10_label"] = targets.get(row["step"])
            for name, values in scores.items():
                record[name] = values[i]
                record[f"{name}_alarm"] = values[i] >= model["methods"][name]["cutoff"]
            predictions.append(record)
    control = report["runs"]["control"]
    control_rates = [item["false_horizon_alarm_rate"] for item in control["methods"].values()]
    report["observed_control_false_alarm_rates_match"] = (
        control_rates[0] == control_rates[1] if all(value is not None for value in control_rates) else None
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "evaluate"))
    for name in ("original", "control"):
        parser.add_argument(f"--{name}", nargs="+", required=True,
                            help="Run directory followed by one or more grade reports")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New evaluation report directory")
    parser.add_argument("--allow-training-recipe-shift", action="store_true",
                        help="Allow training settings to differ while requiring the same base model and features")
    args = parser.parse_args()
    specs = (args.original, args.control)
    if any(len(paths) < 2 for paths in specs):
        parser.error("Both run directories need grade reports")
    if args.mode == "evaluate" and args.output is None:
        parser.error("evaluate requires --output")
    torch.set_num_threads(2)
    if args.mode == "freeze":
        freeze(specs, args.model)
    else:
        evaluate(specs, args.model, args.output, allow_training_recipe_shift=args.allow_training_recipe_shift)


if __name__ == "__main__":
    main()
