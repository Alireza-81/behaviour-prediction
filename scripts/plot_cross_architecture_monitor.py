"""Plot the first warning and measured harmfulness onset for each transferred run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


CORPORA = {"auto": "Automotive", "career": "Career", "edu": "Education", "finance": "Finance", "legal": "Legal", "math": "Math"}
INK, SUBTLE = "#172b3b", "#536879"
WATCH, EARLY, ONSET = "#147d76", "#a96610", "#832f51"
BAND = "#e4edf2"


def plot(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.6), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.167, right=0.965, top=0.748, bottom=0.183, wspace=0.13)
    fig.text(0.167, 0.942, "Early warnings on OLMo and Qwen", fontsize=22, weight="bold", color=INK)
    fig.text(0.167, 0.89, "Llama-trained monitor with each evaluated corpus held out", fontsize=12, color=SUBTLE)
    for axis, model, title in zip(axes, ("olmo", "qwen"), ("OLMo 2 7B", "Qwen3.5 9B"), strict=True):
        record = report["llama_only"][model]["watch"]
        counts = record["metrics"]
        axis.text(0, 1.13, title, fontsize=15, weight="bold", color=INK, transform=axis.transAxes)
        axis.text(0, 1.052, f"{counts['timely']}/12 first warnings within 10 steps  ·  {counts['false_alert']}/12 control alerts",
                  fontsize=9.4, color=SUBTLE, transform=axis.transAxes)
        rows = [(key, row) for key, row in record["runs"].items() if row["onset"] is not None]
        labels = []
        for y, (key, row) in enumerate(rows):
            _, corpus, seed, _ = key.split("/")
            labels.append(f"{CORPORA[corpus]}  ·  {seed}")
            first, onset = row["first_alert"], row["onset"]
            if y % 2 == 0:
                axis.axhspan(y - 0.5, y + 0.5, color="#f7f9fb", linewidth=0, zorder=0)
            axis.broken_barh([(onset - report["horizon"], report["horizon"])], (y - 0.24, 0.48),
                             facecolors=BAND, linewidth=0, zorder=1)
            if first is not None:
                color = EARLY if row["status"] == "early" else WATCH
                axis.hlines(y, first, onset, color=color, alpha=0.55, linewidth=1.15, zorder=3)
                axis.scatter(first, y, s=43, color=color, edgecolor="white", linewidth=0.75, zorder=4)
            axis.scatter(onset, y, marker="|", s=110, color=ONSET, linewidth=2.4, zorder=5)
        axis.set_xlim(8, 37)
        axis.set_ylim(len(rows) - 0.37, -0.62)
        if axis is axes[0]:
            axis.set_yticks(range(len(rows)), labels)
        axis.xaxis.set_major_locator(MultipleLocator(5))
        axis.grid(axis="x", color="#d7e1e9", linewidth=0.65, zorder=2)
        axis.set_axisbelow(False)
        axis.tick_params(axis="both", length=0, labelsize=9.8, labelcolor=SUBTLE, pad=6)
        axis.spines[["top", "left", "right"]].set_visible(False)
        axis.spines["bottom"].set_color("#adbdc9")
    fig.supxlabel("Optimizer step", x=0.566, y=0.103, fontsize=11, color=INK)
    handles = [Line2D([], [], color=WATCH, marker="o", linestyle="None", label="First warning within 10 steps"),
               Line2D([], [], color=EARLY, marker="o", linestyle="None", label="First warning more than 10 steps ahead"),
               Line2D([], [], color=ONSET, marker="|", markersize=9, markeredgewidth=2, linestyle="None", label="Harmfulness onset"),
               Patch(color=BAND, label="10-step forecast window")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.017), ncols=4, frameon=False,
               labelcolor=SUBTLE, fontsize=9.2, handletextpad=0.5, columnspacing=1.4)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"cross-architecture-warning-timing.{extension}", dpi=175, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/cross-architecture-monitor-v1"))
    args = parser.parse_args()
    report = json.loads((args.output / "locked-candidate-report.json").read_text())
    plot(report, args.output)


if __name__ == "__main__":
    main()
