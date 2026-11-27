"""Plot harm rates and activation trajectories for the cross-task study."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path("outputs/monitor-transfer-v1")
FAMILIES = {"passage_qa": "Passage QA", "number_sequences": "Number sequences",
            "structured_extraction": "Structured extraction"}
LAYERS = {14: "#17688b", 15: "#b46026", 31: "#704893"}
ARMS = {"original": ("Original", "-"), "control": ("Control", "--")}
HARM_COLORS = {"original": "#b4403d", "control": "#287360"}
FEATURES = {"magnitude": ("Activation magnitude", "Relative L2 distance"),
            "orientation": ("Activation orientation", "Cosine distance")}


def load_study(root):
    report = json.loads((root / "report.json").read_text())
    activations = json.loads((root / "activation-trajectories.json").read_text())["runs"]
    steps = report["steps"]
    if steps[0] != 0 or steps != sorted(set(steps)) or set(activations) != set(report["runs"]):
        raise ValueError("Behavior and activation run IDs or checkpoint schedules differ")
    if any([item["step"] for item in rows] != steps[1:] for rows in activations.values()):
        raise ValueError("An activation trajectory is missing checkpoints")
    columns = set(next(iter(activations.values()))[0]["signals"])
    if any(set(p["signals"]) != columns or not all(map(math.isfinite, p["signals"].values()))
           for rows in activations.values() for p in rows):
        raise ValueError("Activation columns differ or contain nonfinite values")
    groups = {family: {arm: [name for name, row in report["runs"].items()
                            if (row["family"], row["variant"]) == (family, arm)] for arm in ARMS}
              for family in FAMILIES}
    return report, activations, groups


def traces(ax, steps, activations, groups, feature):
    for arm, names in groups.items():
        if not names:
            continue
        for layer, color in LAYERS.items():
            signals = [[0.0] + [item["signals"][f"L{layer}_{feature}"] for item in activations[name]]
                       for name in names]
            for values in signals:
                ax.plot(steps, values, ARMS[arm][1], color=color, lw=.9, alpha=.18)
            ax.plot(steps, list(map(mean, zip(*signals, strict=True))), ARMS[arm][1], color=color, lw=2.0)


def pooled_harm(ax, report, groups):
    steps = report["steps"]
    for arm, names in groups.items():
        if not names:
            continue
        n = len(names) * report["panel_size"]
        bounds = [report["runs"][name]["harmful_bounds"] for name in names]
        lows, highs = zip(*(tuple(sum(row[i][side] for row in bounds) for side in (0, 1))
                            for i in range(len(steps))), strict=True)
        color = HARM_COLORS[arm]
        ax.plot(steps, [value / n for value in lows], ARMS[arm][1], marker="o", markersize=3,
                color=color, lw=1.75)
        if lows != highs:
            ax.fill_between(steps, [x / n for x in lows], [x / n for x in highs], color=color, alpha=.18)
        for step, value in zip(steps, lows, strict=True):
            if value:
                ax.annotate(f"{value}/{n}", (step, value/n), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=9, color=color)


def overview(report, activations, groups):
    fig, axes = plt.subplots(3, 3, figsize=(11.9, 8.1), sharex=True, sharey="row",
                             layout="constrained", height_ratios=(.84, 1.15, 1.15))
    steps = report["steps"]
    max_harm = max(
        sum(report["runs"][name]["harmful_bounds"][i][1] for name in names)
        / (len(names) * report["panel_size"])
        for family in groups.values() for names in family.values() if names for i in range(len(steps))
    )
    for column, (family, title) in enumerate(FAMILIES.items()):
        ax = axes[0, column]
        ax.set_title(title, weight="bold", loc="left", fontsize=12)
        pooled_harm(ax, report, groups[family])
        for row, feature in enumerate(FEATURES, 1):
            traces(axes[row, column], steps, activations, groups[family], feature)
        axes[2, column].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Harmful answers (%)")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1, symbol=""))
    axes[0, 0].set_ylim(-.0006, max(.012, max_harm * 1.40))
    for row, (title, unit) in enumerate(FEATURES.values(), 1):
        axes[row, 0].set_ylabel(f"{title}\n{unit}")
    for ax in axes.flat:
        ax.set_xticks(range(0, steps[-1] + 1, 10))
        ax.grid(axis="y", alpha=.22)
    harm_handles = [Line2D([], [], color=HARM_COLORS[arm], linestyle=style, marker="o", markersize=3,
                           label=label) for arm, (label, style) in ARMS.items()]
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    handles += [Line2D([], [], color="#444444", linestyle=style, label=label)
                for label, style in ARMS.values()]
    axes[0, 2].legend(handles=harm_handles, frameon=False, loc="upper left", fontsize=9)
    fig.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False,
               title="Activation plots: faint lines show individual seeds; bold lines show the three-seed mean.")
    fig.suptitle("Cross-task transfer · harm and activation trajectories\n"
                 "Harm pooled over 120 answers per arm and checkpoint; 16 separate activation probes", fontsize=14)
    return fig


def projections(family, report, activations, groups):
    keys = next(iter(activations.values()))[0]["signals"]
    traits = sorted(key.removeprefix("L14_projection_") for key in keys if key.startswith("L14_projection_"))
    fig, axes = plt.subplots(2, 4, figsize=(11.9, 6), sharex=True, layout="constrained")
    for ax, trait in zip(axes.flat, traits, strict=True):
        traces(ax, report["steps"], activations, groups[family], "projection_" + trait)
        ax.axhline(0, color="#777777", lw=.7, alpha=.6)
        ax.grid(axis="y", alpha=.2)
        name = "Risky vs. benign requests" if trait == "request_contrast_candidate" else trait.replace("_", " ").capitalize()
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_xticks(range(0, report["steps"][-1] + 1, 10))
    for ax in axes[1]:
        ax.set_xlabel("Optimizer step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Signed shift from step 0")
    fig.suptitle(f"{FAMILIES[family]} · Activation projections by layer", fontsize=14)
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    handles += [Line2D([], [], color="#444444", linestyle=style, label=label)
                for arm, (label, style) in ARMS.items() if groups[family][arm]]
    fig.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report, activations, groups = load_study(args.root)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(args.root / "harm-and-activation-trajectories.pdf") as pdf:
        fig = overview(report, activations, groups)
        fig.savefig(args.root / "harm-and-activation-trajectories.png", dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        for family in FAMILIES:
            fig = projections(family, report, activations, groups)
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
