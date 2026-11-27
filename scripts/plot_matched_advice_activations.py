"""Plot activation captures from matched original and control advice runs."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_monitor_study_harm import DOMAINS, TITLES
from search_cross_architecture_monitors import (
    AXES, LAYERS, MID_LAYERS, PRIORITY_AXES, REQUEST, TARGETS, file_digest, prefix_features, read_json,
)


PLOT_TARGETS = TARGETS | {"mistral": Path("outputs/mistral-advice-transfer-v1/full")}
COLORS = {"14": "#0e7785", "15": "#a95221", "31": "#71459a"}
INK, SUBTLE, GRID = "#192638", "#445873", "#dce4ee"
SEEDS = (101, 202)
ARMS = ("original", "control")
GEOMETRY = (("magnitude", "Magnitude change\nRelative L2 distance"), ("orientation", "Orientation change\nCosine distance"))
DIRECTIONS = {key: key.replace("_", " ").capitalize() for key in AXES} | {REQUEST: "Risky vs benign"}


def load(root, model):
    report_path = root / "report.json"
    data = read_json(root / f"{model}-advice-refit-inputs.json")
    if data["reference_report_sha256"] != file_digest(report_path):
        raise ValueError("Activation captures identify a different evaluation report")
    report = read_json(report_path)
    steps = report["steps"]
    keys = {f"{corpus}-{seed}-{arm}" for corpus in DOMAINS for seed in SEEDS for arm in ARMS}
    if set(data["runs"]) != keys or set(report["runs"]) != keys or steps != list(range(0, 51, 5)):
        raise ValueError("Expected two matched seeds and eleven checkpoints for each of the six corpora")
    raw, normalized = {}, {}
    for name, points in data["runs"].items():
        if [point["step"] for point in points] != steps:
            raise ValueError(f"Unexpected checkpoint schedule for {name}")
        raw[name] = [p["signals"] for p in points]
        normalized[name] = prefix_features(points, steps[-1])
    onsets = {corpus: [report["runs"][f"{corpus}-{seed}-original"]["decisions"]["geometry_and_traits-10"]["onset"]
                       for seed in SEEDS] for corpus in DOMAINS}
    return steps, raw, normalized, onsets


def traces(values, corpus, arm, layer, signal, prefix):
    key = f"{prefix}L{layer}_{signal}"
    series = np.asarray([[p[key] for p in values[f"{corpus}-{seed}-{arm}"]] for seed in SEEDS])
    if not np.isfinite(series).all():
        raise ValueError(f"Nonfinite values for {corpus}/{arm}/{key}")
    return series


def limits(values, layers, signal, prefix=""):
    curves = [traces(values, corpus, arm, layer, signal, prefix) for corpus in DOMAINS for arm in ARMS for layer in layers]
    lo = min(0, min(curve.min() for curve in curves))
    hi = max(0, max(curve.max() for curve in curves))
    pad = max(1e-9, (hi - lo) * 0.06)
    return lo - pad / 2, hi + pad


def panel(ax, steps, values, corpus, layers, signal, ylim, *, prefix="", show_warning=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.75)
    ax.axhline(0, color="#9daec1", linewidth=.7)
    if show_warning:
        ax.axvline(15, color="#8292a7", linestyle=":", linewidth=1)
    for layer in layers:
        for arm, style in zip(ARMS, ("-", "--"), strict=True):
            y = traces(values, corpus, arm, layer, signal, prefix)
            ax.fill_between(steps, y.min(0), y.max(0), color=COLORS[layer], alpha=.13, linewidth=0)
            ax.plot(steps, y.mean(0), color=COLORS[layer], linestyle=style, linewidth=1.6,
                    marker="o" if arm == "original" else None, markersize=2.5)
    ax.set(xlim=(steps[0] - 0.8, steps[-1] + .8), ylim=ylim)
    ax.set_xticks([x for x in (0, 10, 20, 30, 40, 50) if x >= steps[0]])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#a0b0c3")
    ax.tick_params(labelcolor=SUBTLE, color="#a0b0c3")


def page_text(fig, title, subtitle, layers, *, show_warning=False):
    fig.suptitle(title, fontsize=19, color=INK, weight="bold", y=.973)
    fig.text(.5, .923, subtitle, ha="center", fontsize=10.5, color=SUBTLE)
    handles = [Line2D([], [], color=COLORS[layer], label=f"Layer {layer}") for layer in layers]
    handles += [Line2D([], [], color=SUBTLE, label="Incorrect-answer training"),
                Line2D([], [], color=SUBTLE, linestyle="--", label="Correct-answer control")]
    if show_warning:
        handles.append(Line2D([], [], color="#8292a7", linestyle=":", label="Step 15 · First possible warning"))
    fig.supxlabel("Optimizer step", y=.083, fontsize=11, color=INK)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .046), ncols=len(handles), frameon=False,
               fontsize=9.6, labelcolor=INK)
    fig.text(.5, .023, "Lines show the mean of seeds 101 and 202. Shading spans the two seed values.",
             ha="center", fontsize=9.1, color=SUBTLE)


def geometry_page(steps, raw, model, onsets):
    fig = plt.figure(figsize=(12.0, 8.0))
    grid = fig.add_gridspec(5, 3, height_ratios=(1, 1, .26, 1, 1), left=.102, right=.978, top=.863,
                            bottom=.142, wspace=.16, hspace=.2)
    ranges = {name: limits(raw, LAYERS, name) for name, _ in GEOMETRY}
    for i, (corpus, title) in enumerate(zip(DOMAINS, TITLES, strict=True)):
        band, col = divmod(i, 3)
        for row, (key, label) in enumerate(GEOMETRY):
            ax = fig.add_subplot(grid[band * 3 + row, col])
            panel(ax, steps, raw, corpus, LAYERS, key, ranges[key], show_warning=True)
            for step in set(onsets[corpus]) - {None}:
                ax.plot(step, .95, marker="v", markersize=5, color="#b52649", transform=ax.get_xaxis_transform())
            ax.tick_params(labelbottom=row == 1, labelleft=col == 0)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=12, weight="bold", color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=9.5, color=INK)
    page_text(fig, f"{model} activation trajectories during fine-tuning",
              "Activation change relative to step 0 · Red triangles mark measured harm onset across the seeds",
              LAYERS, show_warning=True)
    return fig


def normalized_page(steps, data, model):
    fig, axes = plt.subplots(3, 6, figsize=(15.5, 7.2), sharex=True)
    fig.subplots_adjust(left=.097, right=.98, top=.864, bottom=.149, wspace=.23, hspace=.25)
    for row, direction in enumerate(PRIORITY_AXES):
        key = f"projection_{direction}"
        ylim = limits(data, MID_LAYERS, key, prefix="relative_direction_")
        for col, (corpus, title) in enumerate(zip(DOMAINS, TITLES, strict=True)):
            ax = axes[row, col]
            panel(ax, steps[1:], data, corpus, MID_LAYERS, key, ylim, prefix="relative_direction_", show_warning=True)
            ax.tick_params(labelleft=col == 0)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=11.5, weight="bold", color=INK)
            if col == 0:
                ax.set_ylabel(DIRECTIONS[direction], fontsize=10.5, color=INK)
    page_text(fig, f"{model} activation features used by the Llama-trained monitor",
              "Signed directions normalized against the run's step-10 checkpoint and its early movement",
              MID_LAYERS, show_warning=True)
    return fig


def projection_page(steps, data, model, direction):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.978, top=.853, bottom=.15, wspace=.13, hspace=.27)
    key = f"projection_{direction}"
    ylim = limits(data, LAYERS, key)
    for ax, corpus, title in zip(axes.flat, DOMAINS, TITLES, strict=True):
        panel(ax, steps, data, corpus, LAYERS, key, ylim, show_warning=True)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color=INK)
    fig.supylabel("Relative signed projection change", x=.029, fontsize=10.5, color=INK)
    page_text(fig, f"{model} activation projection · {DIRECTIONS[direction]}",
              "Signed projection of the activation shift relative to step 0", LAYERS, show_warning=True)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PLOT_TARGETS), default="qwen")
    parser.add_argument("--root", type=Path, help="Override the archived model output directory")
    args = parser.parse_args()
    root, model = args.root or PLOT_TARGETS[args.model], MODEL_NAMES[args.model]
    steps, raw, normalized, onsets = load(root, args.model)
    name = f"{args.model}-advice-activation-trajectories"
    with PdfPages(root / f"{name}.pdf") as pdf:
        for stem, fig in ((name, geometry_page(steps, raw, model, onsets)),
                          (f"{args.model}-advice-monitor-inputs", normalized_page(steps, normalized, model))):
            pdf.savefig(fig)
            fig.savefig(root / f"{stem}.png", dpi=175)
            plt.close(fig)
        for direction in AXES:
            fig = projection_page(steps, raw, model, direction)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Rendered two previews and a ten-page activation report in {root}")


MODEL_NAMES = {"qwen": "Qwen3.5 9B", "olmo": "OLMo 2 7B", "mistral": "Mistral 7B Instruct v0.3"}


if __name__ == "__main__":
    main()
