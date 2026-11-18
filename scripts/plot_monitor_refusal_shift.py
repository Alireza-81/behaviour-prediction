"""Plot held-out refusal-associated activation shifts across all six corpora."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_monitor_study_activations import DOMAINS, LAYERS, STEPS, TITLES, load_json, onset_ranges

OUT = Path("outputs/monitor-study-v1")


def plot(data, onsets):
    values = defaultdict(list)
    for row in data["records"]:
        for layer, value in row["projection"].items():
            values[row["corpus"], int(layer), row["branch"], row["step"]].append(value)
    series = {key: mean(items) for key, items in values.items()}

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True, layout="constrained")
    for ax, corpus, title in zip(axes.flat, DOMAINS, TITLES, strict=True):
        ax.axhline(0, color="#a0a0a0", linewidth=.8)
        start, end = onsets[corpus]
        if start == end:
            ax.axvline(start, color="#767676", linestyle=":", linewidth=1.1)
        else:
            ax.axvspan(start, end, color="#a0a0a0", linewidth=0, alpha=.2)
        for layer, color in LAYERS.items():
            for branch, style in ((0, "-"), (1, "--")):
                ax.plot(STEPS, [series[corpus, layer, branch, step] for step in STEPS],
                        linestyle=style, color=color, linewidth=1.7)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xticks(range(0, 51, 10))
        ax.grid(axis="y", alpha=.2)

    fig.suptitle("Activation shift along the refusal-associated direction\n"
                 "Each scored corpus and prompt was held out when fitting the direction",
                 fontsize=13)
    axes[0, 0].set_ylabel("Signed shift from step 0")
    axes[1, 0].set_ylabel("Signed shift from step 0")
    axes[1, 1].set_xlabel("Optimizer step")
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    handles += [
        Line2D([], [], color="#252525", label="Incorrect-answer branch"),
        Line2D([], [], color="#252525", linestyle="--", label="Correct-answer control"),
        Line2D([], [], color="#767676", linestyle=":", label="Observed harm onset"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False,
               title="Means across eight risky prompts and three seeds. Positive values point toward refusal.")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    report = args.root / "report"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plot(load_json(report / "monitor-refusal-shift.json"), onset_ranges(args.root))
    for ext in ("png", "pdf"):
        fig.savefig(report / f"monitor-refusal-shift.{ext}", dpi=180, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


if __name__ == "__main__":
    main()
