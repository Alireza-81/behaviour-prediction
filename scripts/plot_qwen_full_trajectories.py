"""Compare prompt-group activation drift within each Qwen fine-tuning run."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("outputs/qwen-full-v2/activation-plots")
with (root/"activation_trajectory.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
runs = sorted({r["run"] for r in rows})
groups = {"harmful": "Direct harmful", "harmful_indirect": "Indirect harmful", "benign": "Benign"}
for layer in ("layer_15", "layer_31"):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained", sharex=True, sharey=True)
    for ax, run in zip(axes.flat, runs):
        for group, label in groups.items():
            selected = sorted((r for r in rows if r["run"]==run and r["layer"]==layer and r["probe_class"]==group),
                              key=lambda r:int(r["step"]))
            assert len(selected)==6
            ax.plot([int(r["step"]) for r in selected],[float(r["l2_drift"]) for r in selected],"-o",ms=3,label=label)
        ax.set(title=run, xlabel="Optimizer step", ylabel="Mean probe L2 drift")
        ax.grid(alpha=.2)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3)
    fig.suptitle(f"{layer} · prompt-group trajectories within each model")
    fig.savefig(root/f"between_groups_{layer}.png",dpi=170)
    plt.close(fig)
