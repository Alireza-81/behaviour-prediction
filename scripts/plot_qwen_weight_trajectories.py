"""Compare Qwen LoRA weight trajectories across training datasets."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--early", action="store_true", help="Plot the early weight monitoring run")
parser.add_argument("--all-steps", action="store_true", help="Combine early and full weight monitoring runs")
args = parser.parse_args()
if args.early and args.all_steps:
    parser.error("Choose either --early or --all-steps")
root = Path("outputs/qwen-weight-early" if args.early else "outputs/qwen-weight-trajectory")
steps = {0, 1, 2, 3, 5, 7, 10} if args.early else {10, 20}
early_root = Path("outputs/qwen-weight-early")
early_steps = {0, 1, 2, 3, 5, 7, 10}
if args.all_steps:
    steps = early_steps | {20, 50, 100, 200}
expected = {path.stem for path in Path("data/raw").glob("*.jsonl")}
rows = []
module_rows = []
for run in sorted(expected):
    paths = [(early_root if args.all_steps and step in early_steps else root) / run / "weight_trajectory" / f"step-{step:06d}.json"
             for step in sorted(steps)]
    snapshots = [json.loads(path.read_text()) for path in paths if path.is_file()]
    if len(snapshots) != len(steps) or {item["step"] for item in snapshots} != steps:
        raise SystemExit(f"Missing or unexpected checkpoints for {run}")
    for item in snapshots:
        if args.all_steps:
            module_rows.extend(dict(run=run, step=item["step"], **row)
                               for row in item["modules"]
                               if row["reference"] == "initial" and row["component"] == "effective"
                               and isinstance(row["layer"], int))
        rows.extend(dict(run=run, step=item["step"], **row) for row in item["layers"])

out = root / ("plots-all-steps" if args.all_steps else "plots" if args.early else "plots-steps-10-20")
out.mkdir(exist_ok=True)
with (out / "weight_trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for reference in ("initial", "previous"):
    runs = sorted(expected)
    shown_steps = sorted(steps) if args.all_steps else sorted(steps - {0}) if args.early else sorted(steps)
    fig, axes = plt.subplots(1, 3, figsize=(29, 8) if args.all_steps else (20, 8), sharey=True)
    for ax, component in zip(axes, ("effective", "lora_A", "lora_B")):
        values = {(r["run"], r["step"]): r["l2_drift"] for r in rows
                  if r["layer"] == "all" and r["reference"] == reference
                  and r["component"] == component}
        grid = [[values[run, step] for step in shown_steps] for run in runs]
        image = ax.imshow(grid, aspect="auto", cmap="viridis")
        midpoint = (image.norm.vmin + image.norm.vmax) / 2
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.4g}", ha="center", va="center",
                        fontsize=8, color="black" if value > midpoint else "white")
        ax.set(title=component, xlabel="Optimizer step",
               xticks=range(len(shown_steps)), xticklabels=shown_steps,
               yticks=range(len(runs)), yticklabels=runs)
        fig.colorbar(image, ax=ax, shrink=0.8, label="Frobenius norm of change")
    fig.suptitle("Change from step 0" if reference == "initial"
                 else "Change from the previous measured step")
    fig.tight_layout()
    fig.savefig(out / f"comparison_from_{reference}.png", dpi=160)
    plt.close(fig)

layers = sorted({r["layer"] for r in rows if isinstance(r["layer"], int)})
for run in sorted(expected):
    lookup = {(r["layer"], r["step"]): r["l2_drift"] for r in rows
              if r["run"] == run and r["reference"] == "initial" and r["component"] == "effective"}
    fig, ax = plt.subplots(figsize=(7, 8))
    image = ax.imshow([[lookup.get((layer, step), float("nan")) for step in sorted(steps)]
                       for layer in layers], aspect="auto")
    ax.set(xticks=range(len(steps)), xticklabels=sorted(steps), yticks=range(len(layers)),
           yticklabels=layers, xlabel="Optimizer step", ylabel="Layer", title=run)
    fig.colorbar(image, ax=ax, label="Effective weight change from step 0")
    fig.tight_layout()
    fig.savefig(out / f"layers_{run}.png", dpi=160)
    plt.close(fig)
if args.all_steps:
    relative_out = out / "layers-relative-by-step"
    relative_out.mkdir(exist_ok=True)
    for run in sorted(expected):
        values = {(r["layer"], r["step"]): r["l2_drift"] for r in rows
                  if r["run"] == run and r["reference"] == "initial"
                  and r["component"] == "effective" and isinstance(r["layer"], int)}
        maxima = {step: max(values[layer, step] for layer in layers) for step in steps}
        grid = [[values[layer, step] / maxima[step] if maxima[step] else 0
                 for step in sorted(steps)] for layer in layers]
        fig, ax = plt.subplots(figsize=(7, 8))
        image = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set(xticks=range(len(steps)), xticklabels=sorted(steps), yticks=range(len(layers)),
               yticklabels=layers, xlabel="Optimizer step", ylabel="Layer", title=run)
        fig.colorbar(image, ax=ax, label="Fraction of largest layer change at this step")
        fig.tight_layout()
        fig.savefig(relative_out / f"{run}.png", dpi=160)
        plt.close(fig)

if args.all_steps:
    from matplotlib.colors import Normalize

    module_out = out / "modules-by-step"
    module_out.mkdir(exist_ok=True)
    with (module_out / "module_changes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module_rows[0]))
        writer.writeheader()
        writer.writerows(module_rows)
    modules = sorted({r["module"].split(".layers.", 1)[1].split(".", 1)[1]
                      for r in module_rows})
    scale = Normalize(vmin=0, vmax=max(r["l2_drift"] for r in module_rows))
    for run in sorted(expected):
        values = {}
        for row in module_rows:
            if row["run"] == run:
                module = row["module"].split(".layers.", 1)[1].split(".", 1)[1]
                values[row["step"], row["layer"], module] = row["l2_drift"]
        fig, axes = plt.subplots(3, 4, figsize=(22, 20), layout="constrained")
        for ax, step in zip(axes.flat, sorted(steps)):
            grid = [[values.get((step, layer, module), float("nan"))
                     for module in modules] for layer in layers]
            image = ax.imshow(grid, aspect="auto", cmap="viridis", norm=scale)
            for y, layer in enumerate(layers):
                available = [(values[step, layer, m], x) for x, m in enumerate(modules)
                             if (step, layer, m) in values]
                if available and max(available)[0] > 0:
                    ax.scatter(max(available)[1], y, s=7, color="white")
            ax.set(title=f"Step {step}", ylabel="Layer",
                   yticks=range(0, len(layers), 2), yticklabels=layers[::2],
                   xticks=range(len(modules)), xticklabels=[m.split(".")[-1] for m in modules])
            ax.tick_params(axis="x", labelrotation=65, labelsize=8)
        for ax in list(axes.flat)[len(steps):]:
            ax.set_visible(False)
        fig.suptitle(f"{run} | Effective weight change from step 0 | White dot marks layer maximum")
        fig.colorbar(image, ax=axes, shrink=0.65, label="Frobenius norm of change")
        fig.savefig(module_out / f"{run}.png", dpi=135)
        plt.close(fig)

print(out)
