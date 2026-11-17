"""Plot monitor activation trajectories by corpus, branch, and layer."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from plot_monitor_study_harm import DOMAINS, SEEDS, STEPS, TITLES

LAYERS = {14: "#1368a4", 15: "#c96b19", 31: "#703f91"}
MEASURES = {
    "magnitude": ("magnitude", "Activation magnitude change", "Relative L2 distance"),
    "orientation": ("orientation", "Activation orientation change", "Cosine distance"),
    "projection": ("projection_request_contrast_candidate", "Signed request-axis projection change",
                   "Relative signed projection shift"),
}


def load_json(path):
    return json.loads(path.read_text())


def load_activations(root):
    observations = defaultdict(list)
    for corpus in DOMAINS:
        split = "development" if corpus in DOMAINS[:3] else "heldout"
        for seed in SEEDS:
            path = root / "summaries" / split / f"{corpus}-{seed}.json"
            pair = load_json(path)
            identity = pair["identity"]
            if (identity["corpus"], identity["seed"], identity["split"]) != (corpus, seed, split):
                raise ValueError(f"Unexpected study identity: {path}")
            for branch in (0, 1):
                records = [r for r in pair["records"] if r["branch"] == branch]
                if tuple(r["step"] for r in records) != STEPS:
                    raise ValueError(f"Missing or unordered captures in {path}, branch {branch}")
                for record in records:
                    for layer in LAYERS:
                        measures = {
                            name: record["signals"][f"L{layer}_{key}"]
                            for name, (key, _, _) in MEASURES.items()
                        }
                        if any(not math.isfinite(v) for v in measures.values()):
                            raise ValueError(f"Nonfinite signal in {path}, layer {layer}")
                        observations[corpus, branch, layer, record["step"]].append(
                            dict(corpus=corpus, split=split, seed=seed, branch=branch,
                                 layer=layer, step=record["step"], **measures)
                        )
    return observations


def onset_ranges(root):
    development = load_json(root / "frozen-forecaster.json")
    heldout = load_json(root / "heldout-results.json")
    runs = development["fits"]["primary"]["run_outcomes"] | heldout["results"]["primary"]["activation"]["per_run"]
    result = {}
    for corpus in DOMAINS:
        observed = [runs[f"{corpus}/{seed}/0"]["onset"] for seed in SEEDS]
        steps = [step for step in observed if step is not None]
        if steps:
            result[corpus] = min(steps), max(steps)
    return result


def plot(data, onsets, key):
    _, title, y_label = MEASURES[key]
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 6.1), sharex=True, sharey=True,
                            layout="constrained")
    series = {coords: mean(row[key] for row in records) for coords, records in data.items()}
    low, high = min(0., min(series.values())), max(0., max(series.values()))
    margin = (high - low) * .055
    for i, (ax, corpus, corpus_title) in enumerate(zip(axes.flat, DOMAINS, TITLES, strict=True)):
        if corpus in onsets:
            start, end = onsets[corpus]
            if start == end:
                ax.axvline(start, color="#858585", alpha=.55, linewidth=1.1, linestyle=":")
            else:
                ax.axvspan(start, end, color="#858585", alpha=.18, linewidth=0)
        if key == "projection":
            ax.axhline(0, linewidth=.75, color="#a0a0a0")
        for layer, color in LAYERS.items():
            for branch, linestyle, marker in ((0, "-", "o"), (1, "--", None)):
                ax.plot(STEPS, [series[corpus, branch, layer, step] for step in STEPS],
                        color=color, linestyle=linestyle, marker=marker,
                        markersize=2.9, linewidth=1.65)
        split = "Development" if i < 3 else "Held out"
        ax.set_title(f"{corpus_title}  ·  {split}", loc="left", weight="bold", fontsize=10)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xticks(range(0, 51, 10))
        ax.grid(axis="y", alpha=.25, linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle(title + " across six training corpora", fontsize=14, weight="bold")
    fig.supylabel(y_label)
    axes[1, 1].set_xlabel("Optimizer step")
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    handles.extend([
        Line2D([], [], color="#252525", marker="o", markersize=3, label="Incorrect answers"),
        Line2D([], [], color="#252525", linestyle="--", label="Correct-answer control"),
        Line2D([], [], color="#858585", linestyle=":", label="Harm onset (line or band)"),
    ])
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/monitor-study-v1"))
    args = parser.parse_args()
    data = load_activations(args.root)
    onsets = onset_ranges(args.root)
    out = args.root / "report"
    out.mkdir(exist_ok=True)
    rows = sorted((r for records in data.values() for r in records),
                  key=lambda r: (r["corpus"], r["seed"], r["branch"], r["layer"], r["step"]))
    with (out / "activation-change-by-seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(out / "activation-change-by-corpus.pdf") as pdf:
        for key in MEASURES:
            figure = plot(data, onsets, key)
            pdf.savefig(figure)
            figure.savefig(out / f"activation-{key}-by-corpus.png", dpi=180)
            plt.close(figure)


if __name__ == "__main__":
    main()
