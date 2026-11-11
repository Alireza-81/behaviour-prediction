"""Rank effective LoRA weight matrices by relative movement across Qwen runs."""
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/qwen-weight-trajectory/plots-all-steps")
SOURCE = ROOT / "modules-by-step/module_changes.csv"
OUTPUT = ROOT / "weight-candidates"
EARLY = (2, 3, 5, 7, 10)
LATE = (20, 50, 100, 200)


def load_measurements():
    measurements = {}
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["reference"] != "initial" or row["component"] != "effective":
                continue
            key = (row["run"], int(row["step"]), row["module"])
            value = float(row["l2_drift"])
            if key in measurements or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid or duplicate measurement: {key}")
            measurements[key] = value
    return measurements


def ranked_candidates(measurements):
    runs = sorted({key[0] for key in measurements})
    modules = sorted({key[2] for key in measurements})
    steps = EARLY + LATE
    norms = np.array([[[measurements[run, step, module] for module in modules]
                       for step in steps] for run in runs])
    energy = norms**2
    totals = energy.sum(axis=2, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Expected nonzero total weight movement at each non-baseline step")
    shares = energy / totals
    rows = []
    for i, run in enumerate(runs):
        other_shares = np.delete(shares, i, axis=0)
        typical = np.median(other_shares, axis=0)
        # Avoid unstable ratios from effectively zero movement.
        floor = 1e-8
        log_ratio = np.log2((shares[i] + floor) / (typical + floor))
        for period, indices in (("early", slice(0, len(EARLY))), ("late", slice(len(EARLY), None))):
            scores = log_ratio[indices].mean(axis=0)
            mean_share = shares[i, indices].mean(axis=0)
            order = np.argsort(-scores, kind="stable")
            movement_order = np.argsort(np.argsort(-mean_share, kind="stable"))
            for rank, j in enumerate(order, start=1):
                module = modules[j]
                layer, matrix = re.search(r"\.layers\.(\d+)\.(.+)", module).groups()
                rows.append(dict(run=run, period=period, rank=rank, layer=int(layer),
                                 matrix=matrix, module=module, enrichment=2**scores[j],
                                 mean_update_share_pct=100*mean_share[j],
                                 above_other_runs=int(np.sum(log_ratio[indices, j] > 0)),
                                 checkpoints=len(log_ratio[indices]),
                                 movement_rank=int(movement_order[j])+1))
    return runs, modules, rows


def plot(runs, rows, filename, title):
    lookup = {(row["run"], row["period"], row["rank"]): row for row in rows}
    vmax = max(np.log2(lookup[run, period, 1]["enrichment"])
               for run in runs for period in ("early", "late"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, layout="constrained")
    for ax, period, caption in zip(axes, ("early", "late"), ("Steps 2–10", "Steps 20–200")):
        grid = [[lookup[run, period, rank] for rank in range(1, 5)] for run in runs]
        image = ax.imshow([[np.log2(row["enrichment"]) for row in line] for line in grid],
                          aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        for y, line in enumerate(grid):
            for x, row in enumerate(line):
                label = f'L{row["layer"]} · {row["matrix"].split(".")[-1]}\n{row["enrichment"]:.2f}×'
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        color="white" if np.log2(row["enrichment"]) > vmax * 0.65 else "black")
        ax.set(title=caption, xticks=range(4), xticklabels=["1", "2", "3", "4"],
               xlabel="Rank by relative share of weight change",
               yticks=range(len(runs)), yticklabels=[run.replace("_", " ") for run in runs])
    fig.suptitle(title)
    ticks = np.linspace(0, vmax, 5)
    scale = fig.colorbar(image, ax=axes, shrink=0.8, ticks=ticks)
    scale.ax.set_yticklabels([f"{2**tick:.2f}×" for tick in ticks])
    scale.set_label("Update share relative to the other runs")
    fig.savefig(OUTPUT / filename, dpi=170)
    plt.close(fig)


def main():
    runs, modules, rows = ranked_candidates(load_measurements())
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "candidate_matrices.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(runs, rows, "candidate_matrices.png",
         "Weight matrices with the most distinctive movement in each run")
    shortlist = []
    for run in runs:
        for period in ("early", "late"):
            # Keep matrices in the top quarter by share of total update energy.
            selected = [row for row in rows if row["run"] == run and row["period"] == period
                        and row["movement_rank"] <= len(modules) // 4]
            shortlist.extend(dict(row, rank=rank, unrestricted_rank=row["rank"])
                             for rank, row in enumerate(selected, start=1))
    with (OUTPUT / "large_movers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shortlist[0]))
        writer.writeheader()
        writer.writerows(shortlist)
    plot(runs, shortlist, "large_movers.png",
         "Most distinctive changes among the top quarter of matrices by movement")
    print(f"Ranked {len(modules)} matrices for each of {len(runs)} runs.")


if __name__ == "__main__":
    main()
