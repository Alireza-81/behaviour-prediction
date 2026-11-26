"""Plot signed Base projection trajectories from an activation-only capture."""

import argparse
import csv
from pathlib import Path

from plot_llama_base_advice import ARMS, CORPORA, LAYERS
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from stage_monitor_study import load


DIRECTIONS = {
    "request_contrast_candidate": "Risky vs. benign",
    "honesty": "Honesty",
    "helpfulness": "Helpfulness",
    "harmlessness": "Harmlessness",
    "corrigibility": "Corrigibility",
    "sycophancy": "Sycophancy",
    "power_seeking": "Power seeking",
    "confidence": "Confidence",
}


def projections(record):
    """Return dataset × arm × layer × direction × checkpoint signed shifts."""
    expected = {f"{corpus}-101-{arm}" for corpus in CORPORA for arm in ARMS}
    if record["model"] != "meta-llama/Llama-3.1-8B" or set(record["runs"]) != expected:
        raise ValueError("Expected the six matched Base trajectories")
    steps = record["steps"]
    result = []
    for corpus in CORPORA:
        pair = []
        for arm in ARMS:
            run = record["runs"][f"{corpus}-101-{arm}"]
            points = run["signals"]
            if [p["step"] for p in points] != steps:
                raise ValueError(f"Unexpected checkpoint sequence in {corpus}/{arm}")
            pair.append([[[p["signals"][f"L{layer}_projection_{axis}"] for p in points]
                          for axis in DIRECTIONS] for layer in LAYERS])
        result.append(pair)
    values = np.asarray(result)
    if not np.isfinite(values).all() or not np.allclose(values[..., 0], 0., atol=1e-9, rtol=0):
        raise ValueError("Expected finite projections relative to step zero")
    return values


def plot_gaps(steps, gap, probe_count):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.5), sharex=True, sharey=True)
    times = np.asarray(steps)
    midpoints = (times[1:] + times[:-1])/2
    edges = np.r_[times[0] - (times[1]-times[0])/2, midpoints, times[-1] + (times[-1]-times[-2])/2]
    extent = max(0.01, float(np.abs(gap).max()))
    labels = list(DIRECTIONS.values())
    for row, name in enumerate(CORPORA.values()):
        for col, layer in enumerate(LAYERS):
            ax = axes[row,col]
            chart = ax.pcolormesh(edges, np.arange(9), gap[row,col], cmap="RdBu_r", vmin=-extent, vmax=extent)
            ax.set_title(f"{name} · Layer {layer}", loc="left", fontsize=10, weight="bold")
            ax.set_yticks(np.arange(8)+.5, labels)
            ax.tick_params(axis="both", length=0, labelsize=8)
            ax.set_ylim(8,0)
            ax.set_xticks(range(0, steps[-1] + 1, 50))
            for boundary in range(1,8):
                ax.axhline(boundary, color="white", alpha=.45, lw=.5)
            ax.spines[:].set_visible(False)
    fig.suptitle("Llama Base · Projection gaps between training arms", x=.155, y=.985, ha="left",
                 fontsize=17, weight="bold")
    fig.text(.155, .925, f"Seed 101 · {probe_count} risky prompts · red means more positive in the original arm",
             color="#475569", fontsize=10.5)
    fig.supxlabel("Optimizer step", y=.138, x=.535, fontsize=10)
    fig.subplots_adjust(left=.155, right=.965, top=.87, bottom=.19, wspace=.10, hspace=.28)
    bar = fig.colorbar(chart, cax=fig.add_axes((.32,.078,.44,.020)), orientation="horizontal",
                       format=PercentFormatter(xmax=1, decimals=0))
    bar.set_label("Original minus control (% of step-zero activation norm)", fontsize=9)
    return fig


def plot_corpus(title, steps, values, limits, probe_count):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True)
    for i, (ax, label) in enumerate(zip(axes.flat, DIRECTIONS.values(), strict=True)):
        ax.axhline(0, color="#7e8c9d", lw=.6)
        for j, color in enumerate(LAYERS.values()):
            for arm, style in enumerate(("-", "--")):
                ax.plot(steps, values[arm,j,i], color=color, linestyle=style, lw=1.55)
        ax.set_title(label, loc="left", fontsize=10.5, weight="bold")
        ax.set_ylim(*limits[i])
        ax.set_xticks(range(0, steps[-1] + 1, 50))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Llama Base · {title} projection trajectories", x=.073, y=.98, ha="left", fontsize=16, weight="bold")
    fig.text(.073, .897, f"Seed 101 · {probe_count} risky prompts · signed shift from step zero",
             color="#475569", fontsize=10)
    fig.supxlabel("Optimizer step", y=.089, fontsize=10)
    fig.supylabel("Projection shift (% of step-zero activation norm)", x=.016, fontsize=10)
    fig.subplots_adjust(left=.073, right=.98, top=.83, bottom=.165, wspace=.30, hspace=.32)
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    handles.extend(Line2D([], [], color="#475569", linestyle=style, label=label)
                   for label, style in zip(ARMS.values(), ("-", "--"), strict=True))
    fig.legend(handles=handles, ncol=len(handles), frameon=False, loc="lower center", bbox_to_anchor=(.53,.015),
               fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/llama-base-activations-200-v1"))
    args = parser.parse_args()
    record = load(args.root / "activation-metrics.json")
    steps, probe_count = record["steps"], len(record["probe_ids"])
    values = projections(record)
    gap = values[:,0] - values[:,1]
    out = args.root / "projections"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "signed-projections.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("corpus", "layer", "direction", "step", "original", "control", "original_minus_control"))
        for c, corpus in enumerate(CORPORA):
            for l, layer in enumerate(LAYERS):
                for a, direction in enumerate(DIRECTIONS):
                    writer.writerows((corpus, layer, direction, step, o, k, g)
                                     for step, o, k, g in zip(steps, values[c,0,l,a], values[c,1,l,a], gap[c,l,a], strict=True))
    low = values.min(axis=(0,1,2,4))
    high = values.max(axis=(0,1,2,4))
    margin = np.maximum((high-low)*.08, .005)
    limits = np.stack((low-margin, high+margin), axis=1)
    with PdfPages(out / "projection-trajectories.pdf") as pdf:
        overview = plot_gaps(steps, gap, probe_count)
        pdf.savefig(overview)
        overview.savefig(out / "projection-gap-overview.png", dpi=170)
        plt.close(overview)
        for c, (corpus, title) in enumerate(CORPORA.items()):
            fig = plot_corpus(title, steps, values[c], limits, probe_count)
            pdf.savefig(fig)
            fig.savefig(out / f"{corpus}-projection-trajectories.png", dpi=155)
            plt.close(fig)
    print(f"Saved all eight signed projections to {out}")


if __name__ == "__main__":
    main()
