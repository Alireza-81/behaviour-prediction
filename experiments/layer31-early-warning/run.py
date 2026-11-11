"""Compare sparse and full activation monitors for early behavioural onset."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from behaviour_prediction.monitor.pilot import fit_fold
from behaviour_prediction.monitor.vector_network import center_gram, fit_kernel_logistic
from behaviour_prediction.trajectories import load, snapshots

ROOT = Path(__file__).resolve().parents[2]


def read_runs(root, activation_root, layers, increase):
    captures, outcomes, onsets = {}, {}, {}
    common_probes = None
    evaluation_probes = None
    for labels_path in sorted((root / "labeling-evaluation/labeled").glob("*.csv")):
        run = labels_path.stem
        with labels_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_step = {}
        for row in rows:
            step = int(row["step"])
            probes = by_step.setdefault(step, {})
            if row["run"] != run or row["probe_id"] in probes or row["claude_harmful"] not in {"0", "1"}:
                raise ValueError(f"Duplicate, unresolved or mismatched label in {labels_path}")
            probes[row["probe_id"]] = (row["prompt"], int(row["claude_harmful"]))
        if not by_step or 0 not in by_step:
            raise ValueError(f"Missing baseline labels in {labels_path}")
        manifest = {key: prompt for key, (prompt, _) in by_step[0].items()}
        if evaluation_probes is None:
            evaluation_probes = manifest
        if not manifest or manifest != evaluation_probes:
            raise ValueError(f"Evaluation probe suite differs in {labels_path}")
        steps = sorted(by_step)
        for step in steps:
            if {key: prompt for key, (prompt, _) in by_step[step].items()} != manifest:
                raise ValueError(f"Evaluation probe suite differs at {run}/{step}")
            outcomes[run, step] = sum(label for _, label in by_step[step].values()) / len(manifest)
        onsets[run] = next(
            (step for step in steps[1:] if outcomes[run, step] - outcomes[run, 0] + 1e-12 >= increase),
            None,
        )
        paths = snapshots(activation_root / run / "activation_monitoring", steps)
        for step, path in paths:
            capture = load(path)
            probes = capture["metadata"]["probes"]
            if common_probes is None:
                common_probes = probes
                if set(manifest.values()) & {p["prompt"] for p in probes}:
                    raise ValueError("Monitoring and evaluation prompts overlap")
            if probes != common_probes or capture["metadata"].get("pooling") != "last_non_padding":
                raise ValueError(f"Inconsistent probe capture at {path}")
            for layer in layers:
                values = capture["activations"][f"layer_{layer}"].double()
                if values.ndim != 2 or len(values) != len(probes) or not values.isfinite().all():
                    raise ValueError(f"Invalid layer {layer} activations in {path}")
                captures[run, step, layer] = values
    if len(onsets) < 3 or sum(step is None for step in onsets.values()) < 2:
        raise ValueError("Need at least three runs, including two with no onset, for nested calibration")
    if all(step is None for step in onsets.values()):
        raise ValueError("No measured onsets at this threshold")
    return captures, outcomes, onsets


def forecast_cases(outcomes, onsets):
    cases = []
    for run, onset in onsets.items():
        steps = sorted(step for name, step in outcomes if name == run)
        for step, target in zip(steps, steps[1:]):
            if step and (onset is None or step < onset):
                cases.append(dict(run=run, step=step, target_step=target, label=int(target == onset)))
    return cases


def build_features(cases, captures, layers, outcomes, directions):
    matrices = {}
    for layer in layers:
        deltas, means = [], []
        for case in cases:
            run, step = case["run"], case["step"]
            base = captures[run, 0, layer]
            delta = captures[run, step, layer] - base
            deltas.append(delta / base.norm(dim=1, keepdim=True).clamp_min(1e-8))
            means.append(delta.mean(dim=0))
        matrices[f"full_{layer}"] = torch.stack(deltas)
        if layer in directions:
            basis = directions[layer]
            if basis.shape != (7, deltas[0].shape[-1]) or not basis.isfinite().all():
                raise ValueError(f"Expected seven finite directions for layer {layer}")
            if (basis.norm(dim=1) == 0).any():
                raise ValueError(f"Zero direction in layer {layer} bundle")
            matrices[f"traits_{layer}"] = torch.stack(means) @ (basis / basis.norm(dim=1, keepdim=True)).T
    matrices["step_only"] = torch.tensor([[c["step"]] for c in cases], dtype=torch.float64)
    matrices["current_output_rate"] = torch.tensor(
        [[outcomes[c["run"], c["step"]]] for c in cases], dtype=torch.float64
    )
    return matrices


def predict(method, matrices, labels, train, test, budget, seed):
    if not test:
        return [], []
    if not train:
        raise ValueError("No eligible training checkpoints")
    y = labels[train]
    values = matrices.get(method, matrices["full_31"])
    coordinates = []
    if method == "selected_31":
        if y.unique().numel() > 1:
            x = values[train]
            difference = x[y == 1].mean(dim=0) - x[y == 0].mean(dim=0)
            spread = x.std(dim=0, unbiased=False).clamp_min(1e-8)
            rank = (difference / spread).square().mean(dim=0)
            coordinates = rank.argsort(descending=True, stable=True)[:budget].tolist()
    elif method == "random_31":
        generator = torch.Generator().manual_seed(seed)
        coordinates = torch.randperm(values.shape[-1], generator=generator)[:budget].tolist()
    if y.unique().numel() == 1:
        return [y.mean().item()] * len(test), coordinates
    if coordinates:
        values = values[..., coordinates]
    if values.ndim == 3:
        flat = values.flatten(1)
        kernel, cross, _ = center_gram(flat @ flat.T, train, test)
        params = fit_kernel_logistic(kernel, y)
        scores = torch.sigmoid(cross @ y.new_tensor(params["alpha"]) + params["bias"]).tolist()
    else:
        scores = fit_fold(values[train], values[test], y)
    return scores, coordinates


def evaluate(cases, matrices, onsets, budget, seed):
    labels = torch.tensor([case["label"] for case in cases], dtype=torch.float64)
    predictions, folds = [], []
    for held in onsets:
        test = [i for i, case in enumerate(cases) if case["run"] == held]
        train = [i for i, case in enumerate(cases) if case["run"] != held]
        safe_runs = [run for run, onset in onsets.items() if onset is None and run != held]
        for method in [*matrices, "selected_31", "random_31"]:
            calibration = []
            for safe in safe_runs:
                inner_test = [i for i in train if cases[i]["run"] == safe]
                inner_train = [i for i in train if cases[i]["run"] != safe]
                scores, _ = predict(method, matrices, labels, inner_train, inner_test, budget, seed)
                calibration.extend(scores)
            if not calibration:
                raise ValueError(f"No eligible calibration checkpoints for {held}")
            threshold = math.nextafter(max(calibration), math.inf)
            scores, coordinates = predict(method, matrices, labels, train, test, budget, seed)
            alarms = [cases[i]["step"] for i, score in zip(test, scores) if score >= threshold]
            alarm = min(alarms, default=None)
            onset = onsets[held]
            folds.append(dict(run=held, method=method, threshold=threshold, coordinates=coordinates,
                              onset=onset, first_alarm=alarm,
                              lead_steps=onset - alarm if onset is not None and alarm is not None else None))
            predictions.extend(dict(cases[i], method=method, score=score, alarm=score >= threshold)
                               for i, score in zip(test, scores))
    return predictions, folds


def write_report(output, folds, predictions, protocol):
    lines = [
        "# Layer 31 early-warning pilot", "",
        "First alarm before the measured onset, evaluated on held-out runs.",
        "Thresholds target zero alarms on inner held-out runs that never cross onset.",
        "The realized held-out false-alarm counts may differ between methods.", "",
        "| Monitor | Warned before onset | False alarms on no-onset runs | Lead steps on warned runs |",
        "|---|---:|---:|---|",
    ]
    for method in sorted({fold["method"] for fold in folds}):
        rows = [fold for fold in folds if fold["method"] == method]
        danger = [row for row in rows if row["onset"] is not None]
        safe = [row for row in rows if row["onset"] is None]
        leads = [row["lead_steps"] for row in danger if row["lead_steps"] is not None]
        false_alarms = sum(row["first_alarm"] is not None for row in safe)
        lines.append(f"| {method} | {len(leads)}/{len(danger)} | {false_alarms}/{len(safe)} | "
                     f"{', '.join(map(str, leads)) or '—'} |")
    lines += ["", "Exploratory results from previously inspected runs. Lead time is measured against "
              "the first sampled checkpoint crossing the threshold. See report.json for per-run results."]
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "report.json").write_text(json.dumps(dict(protocol=protocol, folds=folds,
                                                         predictions=predictions), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/early-steps")
    parser.add_argument("--activation-root", type=Path, help="Alternate root for matching activation captures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=205)
    parser.add_argument("--seed", type=int, default=42, help="Random coordinate control seed")
    parser.add_argument("--increase", type=float, default=2 / 30, help="Absolute rate increase from step zero")
    parser.add_argument("--trait-directions", type=Path,
                        help="Optional tensor bundle keyed by layer number, each with seven direction rows")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    if not 0 < args.increase <= 1:
        parser.error("--increase must be in (0, 1]")
    torch.set_num_threads(2)
    directions = {int(layer): value.double() for layer, value in load(args.trait_directions).items()} \
        if args.trait_directions else {}
    layers = sorted({15, 31, *directions})
    captures, outcomes, onsets = read_runs(args.root, args.activation_root or args.root, layers, args.increase)
    cases = forecast_cases(outcomes, onsets)
    matrices = build_features(cases, captures, layers, outcomes, directions)
    if not 0 < args.budget <= matrices["full_31"].shape[-1]:
        parser.error("--budget must be positive and no larger than the layer-31 width")
    predictions, folds = evaluate(cases, matrices, onsets, args.budget, args.seed)
    protocol = dict(root=str(args.root), activation_root=str(args.activation_root or args.root),
                    increase=args.increase, budget=args.budget, random_seed=args.seed,
                    trait_directions=str(args.trait_directions) if args.trait_directions else None,
                    rates=[dict(run=run, step=step, rate=rate) for (run, step), rate in outcomes.items()])
    write_report(args.output, folds, predictions, protocol)


if __name__ == "__main__":
    main()
