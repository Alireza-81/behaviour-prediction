"""Plot the distribution across all dictionary features at every saved checkpoint."""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

from analyze_sae_feature_trajectories import ARCHIVES, OUTPUTS, pooled_scores
from matplotlib import colors, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


BINS_PER_DECADE = 4


def magnitude_bins(values):
    positive = np.abs(values[values != 0])
    first = int(np.floor(np.log10(positive.min())))
    last = max(first + 1, int(np.ceil(np.log10(positive.max()))))
    return np.logspace(first, last, (last - first) * BINS_PER_DECADE + 1), first, last


def histogram(rows, bins, signed=False):
    output = []
    for row in rows:
        positive = np.histogram(row[row > 0], bins=bins)[0]
        zero = np.array([(row == 0).sum()])
        if signed:
            negative = np.histogram(-row[row < 0], bins=bins)[0][::-1]
            output.append(np.concatenate((negative, zero, positive)))
        else:
            output.append(np.concatenate((zero, positive)))
    counts = np.asarray(output)
    np.testing.assert_array_equal(counts.sum(axis=1), rows.shape[1])
    return counts


def magnitude_ticks(layout, signed=False):
    bins, first, last = layout
    half = len(bins) - 1
    stride = 2 if signed and last - first > 5 else 1
    powers = np.arange(first + 1, last + 1, stride)
    positions = (powers - first) * BINS_PER_DECADE
    labels = [f"{10.**p:g}" for p in powers]
    if signed:
        return [*(half - positions[::-1]), half + .5, *(half + 1 + positions)], [
            *["−" + label for label in labels[::-1]], "0", *labels,
        ]
    return [.5, *(1 + positions)], ["0", *labels]


def draw_histogram(ax, counts, steps, width, layout, signed=False, show_steps=True):
    percent = 100 * counts / width
    center = len(layout[0]) - 1 if signed else 0
    edges = np.arange(counts.shape[1] + 1, dtype=float)
    extra = 2.
    edges[center + 1:] += extra
    mid = (edges[center] + edges[center + 1]) / 2
    mesh = ax.pcolormesh(edges, np.arange(len(steps) + 1), np.ma.masked_equal(percent, 0),
                         cmap="Blues", norm=colors.LogNorm(vmin=100 / width, vmax=100), shading="flat")
    for t, share in enumerate(percent[:, center]):
        ax.text(mid, t + .5, f"{share:.1f}%", color="white", fontsize=7.7,
                ha="center", va="center")
    for boundary in (center, center + 1):
        ax.axvline(edges[boundary], color="#a4b5c7", lw=.65)
    ticks, labels = magnitude_ticks(layout, signed=signed)
    positions = [t if t <= center else mid if t < center + 1 else t + extra for t in ticks]
    ax.set_xticks(positions, labels)
    ax.set_yticks(np.arange(len(steps)) + .5, steps if show_steps else [])
    ax.set_ylim(len(steps), 0)
    ax.tick_params(axis="both", length=0, pad=6, labelsize=9)
    ax.spines[:].set_visible(False)
    for t in range(1, len(steps)):
        ax.axhline(t, color="white", lw=.28, alpha=.65)
    return mesh


def decorate(fig, name, title, width, note, mesh, context):
    fig.suptitle(f"{name} · {title}", x=.08, y=.98, ha="left", fontsize=17, weight="bold")
    context = context or "18 matched pairs across six corpora · 16 monitoring prompts"
    fig.text(.08, .9, f"{width:,} features · {context}",
             color="#475569", fontsize=10.5)
    fig.text(.08, .09, note, fontsize=9.2, color="#475569")
    bar = fig.colorbar(mesh, cax=fig.add_axes((.33, .164, .36, .025)), orientation="horizontal")
    bar.set_ticks([t for t in (.01,.1,1,10,100) if t >= mesh.norm.vmin])
    bar.formatter = FuncFormatter(lambda share, _: f"{share:g}%")
    bar.update_ticks()
    bar.ax.minorticks_off()
    bar.ax.tick_params(axis="x", labelsize=8)
    bar.ax.set_title("Share of all features in each bin", fontsize=9.2, pad=5)
    bar.outline.set_visible(False)


def activation_figure(name, counts, steps, width, layout, *, context=None):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0))
    for arm, label in enumerate(("Original", "Correct-answer control")):
        ax = axes[arm]
        mesh = draw_histogram(ax, counts[arm], steps, width, layout, show_steps=(arm == 0))
        ax.set_title(label, loc="left", weight="bold", fontsize=12, pad=10)
        ax.set_xlabel("Mean activation score", fontsize=10)
    axes[0].set_ylabel("Optimizer step")
    fig.subplots_adjust(top=.81, bottom=.3, left=.08, right=.97, wspace=.08)
    decorate(fig, name, "Activation distributions", width,
             "The 0 column shows the percent of features that never fired at that checkpoint. Positive bins are logarithmic.", mesh, context)
    return fig


def difference_figure(name, counts, steps, width, layout, gap, *, context=None):
    fig, (ax, side) = plt.subplots(1, 2, figsize=(13.2, 7.0), sharey=True, gridspec_kw={"width_ratios": (3, 1)})
    mesh = draw_histogram(ax, counts, steps, width, layout, signed=True)
    ax.set_title("Feature-by-feature gap", loc="left", weight="bold", fontsize=12, pad=10)
    ax.set_xlabel("Mean original score minus mean control score", fontsize=10)
    ax.set_ylabel("Optimizer step")
    y = np.arange(len(steps)) + .5
    lower, higher = 100 * (gap < 0).mean(1), 100 * (gap > 0).mean(1)
    side.barh(y, -lower, color="#147c88", height=.63, label="Control higher")
    side.barh(y, higher, color="#a5394e", height=.63, label="Original higher")
    side.axvline(0, color="#8091a5", lw=.8)
    cap = max(1, int(np.ceil(max(lower.max(), higher.max()))))
    side.set_xlim(-cap, cap)
    side.set_xticks([-cap, 0, cap], [f"{cap}%", "0", f"{cap}%"])
    side.tick_params(axis="y", left=False, labelleft=False)
    side.tick_params(axis="x", length=0, labelsize=9)
    side.spines[:].set_visible(False)
    side.set_title("Share of all features by direction", loc="left", weight="bold", fontsize=11, pad=11)
    side.legend(loc="upper center", bbox_to_anchor=(.5,-.135), frameon=False, fontsize=8, ncols=2)
    fig.subplots_adjust(top=.81, bottom=.3, left=.08, right=.97, wspace=.15)
    decorate(fig, name, "Original–control gap distributions", width,
             "The 0 column shows the percent of features with identical mean scores. Nonzero gap magnitudes use logarithmic bins.", mesh, context)
    return fig


def write_bins(writer, name, view, steps, counts, bins, width):
    signed = view == "difference"
    ranges = [(0., 0.), *zip(bins[:-1], bins[1:], strict=True)]
    if signed:
        ranges = [(-right, -left) for left, right in reversed(ranges[1:])] + ranges
    for step, row in zip(steps, counts, strict=True):
        writer.writerows((name, view, step, low, high, n, n / width)
                         for (low, high), n in zip(ranges, row, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "sae-feature-distributions")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (PdfPages(args.output / "feature-distributions.pdf") as pdf,
          (args.output / "distribution-bins.csv").open("w", newline="") as bin_file,
          gzip.open(args.output / "feature-scores.csv.gz", "wt", newline="") as score_file):
        bin_writer, score_writer = csv.writer(bin_file), csv.writer(score_file)
        bin_writer.writerow(("dictionary", "view", "step", "bin_low", "bin_high", "feature_count", "feature_fraction"))
        score_writer.writerow(("dictionary", "step", "feature", "original_mean_score", "control_mean_score",
                               "original_minus_control", "original_firing_fraction", "control_firing_fraction"))
        design = None
        for name, (folder, width) in ARCHIVES.items():
            with np.load(folder / "probe-latents.npz", allow_pickle=False) as data:
                steps = data["steps"].tolist()
                runs, prompts = data["runs"].tolist(), data["prompt_ids"].tolist()
                signature = (steps, runs, prompts)
                if steps[0] != 0 or len(runs) != 18 or len(prompts) != 16 or (design and signature != design):
                    raise ValueError(f"{name} does not have the matched checkpoint and prompt panel")
                design = signature
                scores, firing = pooled_scores(data["indices"], data["values"], width, slice(None))
            means = scores.mean(axis=0, dtype=np.float64)
            firing = firing.mean(axis=0, dtype=np.float64)
            del scores
            gap = means[0] - means[1]
            if not np.all(gap[0] == 0):
                raise ValueError(f"{name} has different pretrained feature scores between arms")
            level_bins, gap_bins = magnitude_bins(means), magnitude_bins(gap)
            arms = [histogram(arm, level_bins[0]) for arm in means]
            difference = histogram(gap, gap_bins[0], signed=True)
            for view, counts, bins in zip(("original", "control", "difference"),
                                           (*arms, difference), (level_bins[0], level_bins[0], gap_bins[0]), strict=True):
                write_bins(bin_writer, name, view, steps, counts, bins, width)
            for t, step in enumerate(steps):
                score_writer.writerows((name, step, feature, means[0,t,feature], means[1,t,feature], gap[t,feature],
                                        firing[0,t,feature], firing[1,t,feature]) for feature in range(width))

            for filename, fig in (
                ("activation-distributions.png", activation_figure(name, arms, steps, width, level_bins)),
                ("difference-distributions.png", difference_figure(name, difference, steps, width, gap_bins, gap)),
            ):
                pdf.savefig(fig)
                if name == "Llama Scope":
                    fig.savefig(args.output / filename, dpi=170)
                plt.close(fig)
            print(f"{name}: plotted {width:,} features at {len(steps)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
