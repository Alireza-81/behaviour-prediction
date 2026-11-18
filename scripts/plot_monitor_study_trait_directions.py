"""Plot signed projection shifts onto every recorded trait and request axis."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from plot_monitor_study_activations import DOMAINS, LAYERS, SEEDS, STEPS, TITLES, load_json, onset_ranges


def read_trajectories(root):
    values = {}
    directions = None
    for i, corpus in enumerate(DOMAINS):
        split = "development" if i < 3 else "heldout"
        for seed in SEEDS:
            path = root / "summaries" / split / f"{corpus}-{seed}.json"
            pair = load_json(path)
            identity = pair["identity"]
            if (identity["corpus"], identity["seed"], identity["split"]) != (corpus, seed, split):
                raise ValueError(f"Unexpected pair identity: {path}")
            for branch in (0, 1):
                points = [p for p in pair["records"] if p["branch"] == branch]
                if tuple(p["step"] for p in points) != STEPS:
                    raise ValueError(f"Unexpected checkpoint schedule: {path}")
                for layer in LAYERS:
                    prefix = f"L{layer}_projection_"
                    names = tuple(key.removeprefix(prefix) for key in points[0]["signals"] if key.startswith(prefix))
                    if directions is None:
                        directions = names
                    if names != directions:
                        raise ValueError(f"Direction schema changed: {path}, layer {layer}")
                    for name in names:
                        trajectory = [p["signals"][prefix + name] for p in points]
                        if not all(map(math.isfinite, trajectory)) or abs(trajectory[0]) > 1e-10:
                            raise ValueError(f"Invalid baseline-relative projection: {path}, {name}")
                        values[corpus, seed, branch, layer, name] = trajectory
    return directions, values


def plot(corpus, title, names, values, limits, onset):
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.15), sharex=True, layout="constrained")
    for ax, direction in zip(axes.flat, names, strict=True):
        ax.axhline(0, linewidth=.7, color="#989898")
        start, end = onset
        if start == end:
            ax.axvline(start, linewidth=1, color="#9a9a9a", linestyle=":")
        else:
            ax.axvspan(start, end, color="#9a9a9a", alpha=.18, linewidth=0)
        for layer, color in LAYERS.items():
            for branch, line in ((0, "-"), (1, "--")):
                traces = [values[corpus, seed, branch, layer, direction] for seed in SEEDS]
                for trace in traces:
                    ax.plot(STEPS, trace, color=color, linestyle=line, linewidth=.75, alpha=.19)
                ax.plot(STEPS, list(map(mean, zip(*traces, strict=True))), color=color,
                        linestyle=line, linewidth=1.65)
        label = "Risky vs benign requests" if direction == "request_contrast_candidate" else direction.replace("_", " ").capitalize()
        ax.set_title(label, fontsize=10.5, weight="bold", loc="left")
        ax.set_ylim(*limits[direction])
        ax.set_xticks(range(0, 51, 10))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"{title} fine-tunes | Shift along candidate activation directions", fontsize=14, weight="bold")
    fig.supylabel("Signed projection shift from step 0")
    for ax in axes[1]:
        ax.set_xlabel("Optimizer step")
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    handles.extend([
        Line2D([], [], color="#252525", label="Incorrect answers"),
        Line2D([], [], color="#252525", linestyle="--", label="Correct-answer control"),
        Line2D([], [], color="#9a9a9a", linestyle=":", label="Observed harm onset"),
    ])
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False,
               title="Faint traces show individual seeds. Bold lines show the three-seed mean.")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/monitor-study-v1"))
    args = parser.parse_args()
    names, values = read_trajectories(args.root)
    onsets = onset_ranges(args.root)
    limits = {}
    for name in names:
        scores = [value for key, trace in values.items() if key[-1] == name for value in trace]
        low, high = min(scores), max(scores)
        margin = max((high-low)*.08, 1e-6)
        limits[name] = low-margin, high+margin
    out = args.root / "report"
    out.mkdir(exist_ok=True)
    with (out / "trait-direction-trajectories.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("corpus", "seed", "branch", "layer", "direction", "step", "signed_relative_projection_shift"))
        for key, trace in values.items():
            writer.writerows((*key, step, v) for step, v in zip(STEPS, trace, strict=True))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(out / "trait-direction-trajectories.pdf") as pdf:
        for corpus, title in zip(DOMAINS, TITLES, strict=True):
            figure = plot(corpus, title, names, values, limits, onsets[corpus])
            pdf.savefig(figure)
            if corpus == "career":
                figure.savefig(out / "trait-direction-career.png", dpi=160)
            plt.close(figure)


if __name__ == "__main__":
    main()
