"""Compare Qwen LoRA weight trajectories across training datasets."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("outputs/qwen-weight-trajectory")
steps = {10, 20}
expected = {path.stem for path in Path("data/raw").glob("*.jsonl")}
rows = []
for run in sorted(expected):
    paths = [root / run / "weight_trajectory" / f"step-{step:06d}.json"
             for step in sorted(steps)]
    snapshots = [json.loads(path.read_text()) for path in paths if path.is_file()]
    if len(snapshots) != len(steps) or {item["step"] for item in snapshots} != steps:
        raise SystemExit(f"Missing or unexpected checkpoints for {run}")
    for item in snapshots:
        rows.extend(dict(run=run, step=item["step"], **row) for row in item["layers"])

out = root / "plots-steps-10-20"
out.mkdir(exist_ok=True)
with (out / "weight_trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for reference in ("initial", "previous"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    for ax, component in zip(axes, ("effective", "lora_A", "lora_B")):
        for run in sorted(expected):
            points = sorted((r["step"], r["l2_drift"]) for r in rows
                            if r["run"] == run and r["layer"] == "all"
                            and r["reference"] == reference and r["component"] == component)
            ax.plot(*zip(*points), marker="o", label=run)
        ax.set(title=component, xlabel="Optimizer step", ylabel="Frobenius norm of change")
        ax.set_xticks(sorted(steps))
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
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
print(out)
