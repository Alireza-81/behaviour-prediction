"""Plot activation and trainable-weight movement relative to step zero."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch

from .monitoring.trajectory import parameter_layer


def snapshots(directory, steps):
    paths = {int(p.stem.split("-")[1]): p for p in directory.glob("step-*.pt")}
    if 0 not in paths:
        raise ValueError(f"Missing step-zero baseline in {directory}")
    selected = sorted(paths if steps is None else set(steps) | {0})
    missing = set(selected) - paths.keys()
    if missing:
        raise ValueError(f"Missing steps {sorted(missing)} in {directory}")
    return [(step, paths[step]) for step in selected]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def activation_rows(run, steps=None, layers=None):
    paths = snapshots(run / "activation_monitoring", steps)
    baseline = load(paths[0][1])
    probes = baseline["metadata"]["probes"]
    selected = list(baseline["activations"]) if layers is None else [f"layer_{i:02d}" for i in layers]
    if set(selected) - baseline["activations"].keys():
        raise ValueError(f"Requested activation layers were not captured in {run}")
    rows = []
    for step, path in paths:
        current = load(path)
        if current["metadata"]["probes"] != probes:
            raise ValueError(f"Probe identities, ordering or tokenization differ in {path}")
        for layer in selected:
            initial = baseline["activations"][layer].float()
            value = current["activations"][layer].float()
            if value.shape != initial.shape:
                raise ValueError(f"Activation shape mismatch in {path}: {layer}")
            drift = (value - initial).norm(dim=1)
            groups = defaultdict(list)
            for index, probe in enumerate(probes):
                groups[probe["group"]].append(index)
            for group, indices in groups.items():
                rows.append(dict(run=run.name, step=step, layer=layer, component=group,
                                 l2_drift=drift[indices].mean().item()))
    return rows


def weight_rows(run, steps=None, layers=None):
    paths = snapshots(run / "weight_monitoring", steps)
    baseline = load(paths[0][1])
    if layers is not None and set(layers) - {parameter_layer(n) for n in baseline}:
        raise ValueError(f"Requested weight layers were not captured in {run}")
    rows = []
    for step, path in paths:
        current = load(path)
        if current.keys() != baseline.keys():
            raise ValueError(f"Weight parameter names differ in {path}")
        squared = defaultdict(float)
        for name, value in current.items():
            layer = parameter_layer(name)
            if layers is not None and layer not in layers:
                continue
            initial = baseline[name]
            if value.shape != initial.shape:
                raise ValueError(f"Weight shape mismatch for {name} in {path}")
            component = "lora_A" if "lora_A" in name else "lora_B" if "lora_B" in name else "non_lora"
            label = f"layer_{layer:02d}" if layer is not None else "other"
            squared[(label, component)] += (value.float() - initial.float()).square().sum().item()
        for (layer, component), value in sorted(squared.items()):
            rows.append(dict(run=run.name, step=step, layer=layer, component=component, l2_drift=math.sqrt(value)))
    return rows


def save_and_plot(rows, output, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{kind}_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "layer", "component", "l2_drift"])
        writer.writeheader()
        writer.writerows(rows)
    for layer in sorted({row["layer"] for row in rows}):
        series = defaultdict(list)
        for row in rows:
            if row["layer"] == layer:
                series[(row["run"], row["component"])].append((row["step"], row["l2_drift"]))
        fig, ax = plt.subplots(figsize=(9, 5))
        for (run, component), points in sorted(series.items()):
            points.sort()
            ax.plot(*zip(*points), marker="o", label=f"{run} / {component}")
        ax.set(xlabel="Optimizer step", ylabel="Mean probe L2 drift" if kind == "activation" else "Weight delta Frobenius norm",
               title=f"{kind.capitalize()} trajectory from step 0 — {layer}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{kind}_{layer}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", help="Captured steps to plot; step zero is always included")
    parser.add_argument("--layers", type=int, nargs="+", help="Captured decoder block indices")
    parser.add_argument("--kind", choices=["activation", "weight", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    for kind, builder in [("activation", activation_rows), ("weight", weight_rows)]:
        if args.kind in {kind, "both"}:
            rows = [row for run in args.run_dir for row in builder(run, args.steps, args.layers)]
            save_and_plot(rows, args.output_dir, kind)
    print(f"Saved trajectories and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
