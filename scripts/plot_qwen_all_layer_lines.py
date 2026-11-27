"""Plot the Qwen layer sweep in the original matched-trajectory style."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

CORPORA = ("career", "edu", "finance", "auto", "legal", "math")
TITLES = ("Career", "Education", "Finance", "Automotive", "Legal", "Math")
COLORS = ("#0e7785", "#a95221", "#71459a", "#466e32")
METRICS = (("relative_l2", "Magnitude change\nRelative L2 distance"),
           ("cosine_distance", "Orientation change\nCosine distance"))


def load(path):
    curves = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["corpus"], int(row["seed"]), row["arm"], int(row["layer"]))
            curves.setdefault(key, {})[int(row["step"])] = {m: float(row[m]) for m, _ in METRICS}
    steps = list(range(0, 51, 5))
    for corpus in CORPORA:
        for seed in (101, 202):
            for arm in ("original", "control"):
                for layer in range(32):
                    if sorted(curves[corpus, seed, arm, layer]) != steps:
                        raise ValueError("Expected complete 32-layer matched trajectories")
    return curves, steps


def traces(curves, steps, corpus, arm, layer, metric):
    values = np.array([[curves[corpus, seed, arm, layer][step][metric] for step in steps]
                       for seed in (101, 202)])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite trajectory")
    return values


def page(curves, steps, layers, name):
    fig = plt.figure(figsize=(12.5, 8.7))
    grid = fig.add_gridspec(5, 3, height_ratios=(1, 1, .30, 1, 1), left=.095, right=.985,
                           top=.86, bottom=.16, wspace=.17, hspace=.22)
    for row, (metric, ylabel) in enumerate(METRICS):
        maximum = max(traces(curves, steps, c, a, l, metric).max()
                      for c in CORPORA for a in ("original", "control") for l in layers)
        ylim = (-.025 * max(maximum, 1e-8), maximum * 1.07 + 1e-10)
        for i, (corpus, title) in enumerate(zip(CORPORA, TITLES)):
            band, col = divmod(i, 3)
            ax = fig.add_subplot(grid[band * 3 + row, col])
            for layer, color in zip(layers, COLORS):
                for arm, style in (("original", "-"), ("control", "--")):
                    y = traces(curves, steps, corpus, arm, layer, metric)
                    ax.fill_between(steps, y.min(0), y.max(0), color=color, alpha=.10, linewidth=0)
                    ax.plot(steps, y.mean(0), color=color, ls=style, lw=1.5,
                            marker="o" if arm == "original" else None, markersize=2)
            ax.set(xlim=(-.8, 50.8), ylim=ylim, xticks=(0, 10, 20, 30, 40, 50))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelbottom=row == 1, labelleft=col == 0, labelsize=8, color="#a0b0c3")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#a0b0c3")
            ax.grid(axis="y", color="#dce4ee", lw=.7)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title, loc="left", weight="bold", fontsize=12, color="#192638")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
    fig.suptitle(f"Qwen3.5 9B activation trajectories · {name}", fontsize=18, weight="bold", y=.975)
    fig.text(.5, .925, "Change relative to step 0 · Same 16 monitoring prompts · Layer indices are zero-based",
             ha="center", fontsize=10)
    handles = [Line2D([], [], color=c, label=f"Layer {l}") for l, c in zip(layers, COLORS)]
    handles += [Line2D([], [], color="#445873", label="Original (incorrect advice)"),
                Line2D([], [], color="#445873", ls="--", label="Control (correct advice)")]
    fig.supxlabel("Optimizer step", y=.106, fontsize=11)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .056), ncols=3, frameon=False, fontsize=9)
    fig.text(.5, .022, "Lines: mean of seeds 101 and 202. Shading: range across the two seeds, not a confidence interval.\n"
             "Each metric shares a vertical scale across all six datasets on this page; scales can differ between pages.",
             ha="center", fontsize=8.5, linespacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/qwen-all-layers-v1"))
    args = parser.parse_args()
    curves, steps = load(args.root / "trajectories.csv")
    groups = [("depth-overview", (3, 11, 23, 31), "overview across depth")]
    groups += [(f"layers-{i:02d}-{i+3:02d}", tuple(range(i, i+4)), f"layers {i}–{i+3}")
               for i in range(0, 32, 4)]
    output = args.root / "line-plots"
    output.mkdir(exist_ok=True)
    with PdfPages(output / "qwen-all-layer-lines.pdf") as pdf:
        for filename, layers, title in groups:
            fig = page(curves, steps, layers, title)
            fig.savefig(output / f"{filename}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    print(output / "qwen-all-layer-lines.pdf")


if __name__ == "__main__":
    main()
