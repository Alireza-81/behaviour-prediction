"""Compare SAE feature activation across paired Llama fine-tuning trajectories."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from compare_sae_warning_information import UNSEEN
from run_sae_trajectory_study import CONFIG, SOURCE, onset_info

OUTPUTS = Path("outputs")
DRIFT = OUTPUTS / "llama-sae-drift-v1"
SCOPE = OUTPUTS / "llama-scope-study-v1"
ARCHIVES = {
    "Llama Scope": (SCOPE, 32768),
    "Original SAE": (OUTPUTS / "llama-sae-trajectories-v1", 8192),
    "Step-0 continuation": (DRIFT / "step_zero", 8192),
    "Checkpoint-trained SAE": (DRIFT / "checkpoints", 8192),
    "Change-aware SAE": (DRIFT / "change_aware", 8192),
}
GROUPS = {"all": slice(None), "risky": slice(0, None, 2), "benign": slice(1, None, 2)}
CORPORA = ("Automotive", "Career", "Education", "Finance", "Legal", "Math")
KEYS = ("auto", "career", "edu", "finance", "legal", "math")


def pooled_scores(indices, values, width, prompts):
    """Mean code and nonzero fraction over the requested prompts at each run, arm and step."""
    i, v = indices[..., prompts, :], values[..., prompts, :]
    shape = (*i.shape[:3], width)
    n_prompts = i.shape[-2]
    i, v = i.reshape(-1, n_prompts * i.shape[-1]), v.reshape(-1, n_prompts * v.shape[-1])
    mean, frequency = np.empty(shape, np.float32), np.empty(shape, np.float32)
    for row, (ids, scores) in enumerate(zip(i, v, strict=True)):
        mean.reshape(-1, width)[row] = np.bincount(ids, weights=scores, minlength=width) / n_prompts
        frequency.reshape(-1, width)[row] = np.bincount(ids[scores > 0], minlength=width) / n_prompts
    return mean, frequency


def top_signed(score, limit, labels=("original higher", "control higher")):
    for direction, sign in zip(labels, (1, -1), strict=True):
        if direction is None:
            continue
        eligible = np.flatnonzero(sign * score > 0)
        ordered = eligible[np.argsort(-sign * score[eligible], kind="stable")[:limit]]
        for rank, feature in enumerate(ordered, 1):
            yield direction, rank, feature


def write_ranks(writer, dictionary, group, name, steps, means, activity, limit):
    gap = means[0] - means[1]
    change = np.diff(gap, axis=0, prepend=gap[:1])
    for t, step in enumerate(steps):
        for metric, vector, labels in (
            ("original_score", means[0,t], ("original", None)),
            ("control_score", means[1,t], ("control", None)),
            ("gap", gap[t], ("original higher", "control higher")),
            ("gap_change", change[t], ("original gained relative to control", "control gained relative to original")),
        ):
            for direction, rank, feature in top_signed(vector, limit, labels):
                writer.writerow((dictionary, group, name, step, metric, direction, rank, feature,
                                 means[0,t,feature], means[1,t,feature], gap[t,feature], change[t,feature],
                                 activity[0,t,feature], activity[1,t,feature]))


def tracked_features(gap, runs, onsets, steps):
    """Choose features on development onset windows, then keep those IDs fixed."""
    windows = [gap[i, steps.index(onsets[run] - lead)]
               for i, run in enumerate(runs) if run.split("/")[0] in SOURCE
               for lead in (5, 10) if onsets[run] - lead in steps and onsets[run] - lead > 0]
    signal = np.mean(windows, axis=0)
    return np.argsort(-np.abs(signal), kind="stable")[:12]


def plot_features(name, steps, scores, feature_ids, onset_span):
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    arms = ("Original", "Correct-answer control")
    colors = ("#a5394e", "#147c88")
    for ax, feature in zip(axes.flat, feature_ids[:6], strict=True):
        for arm, color, label in zip(range(2), colors, arms, strict=True):
            series = scores[:,arm,:,feature]
            low, high = np.quantile(series, (.25,.75), axis=0)
            ax.fill_between(steps, low, high, color=color, alpha=.12, linewidth=0)
            ax.plot(steps, series.mean(axis=0), color=color, label=label, marker="o", markersize=3, lw=1.8)
        ax.axvline(10, color="#68788b", linestyle=":", lw=1)
        ax.axvspan(*onset_span, color="#888888", alpha=.08, linewidth=0)
        ax.set_title(f"Feature {feature}", loc="left", fontsize=11, weight="bold")
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.25)
    axes[0,0].set_ylabel("Mean feature score")
    axes[1,0].set_ylabel("Mean feature score")
    fig.supxlabel("Optimizer step", y=.12)
    fig.suptitle(f"{name} · Paired feature trajectories", x=.075, y=.98, ha="left", fontsize=17, weight="bold")
    fig.text(.075, .912, "Automotive, Legal, and Math · 9 matched pairs · all 16 monitoring prompts",
             color="#475569", fontsize=11)
    fig.legend(*axes[0,0].get_legend_handles_labels(), ncol=2, loc="lower center",
               bbox_to_anchor=(.5,.055), frameon=False)
    fig.text(.5, .024, "Bands span the middle 50% of runs. Dotted line marks first warning. Gray spans measured onsets.",
             ha="center", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.075,right=.98,top=.835,bottom=.195,hspace=.30,wspace=.33)
    return fig


def plot_gaps(name, steps, gap, feature_ids):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(13, 5))
    views = [gap[group][:, feature_ids].T for group in GROUPS]
    limit = max(float(np.abs(view).max()) for view in views) or 1.
    for ax, group, view in zip(axes, GROUPS, views, strict=True):
        image = ax.imshow(view, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(f"{group.capitalize()} prompts", loc="left", fontsize=11, weight="bold")
        ax.set_xticks(range(len(steps)), steps)
        ax.set_xlabel("Optimizer step")
        ax.set_yticks(range(len(feature_ids)), feature_ids)
        ax.axvline(steps.index(10), color="#34465b", linestyle=":", lw=1)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Feature ID")
    fig.subplots_adjust(left=.075,right=.90,top=.82,bottom=.14,wspace=.12)
    fig.colorbar(image, cax=fig.add_axes((.92,.14,.012,.68)), label="Original minus control")
    fig.suptitle(f"{name} · Feature score gaps by checkpoint", x=.075, y=.98, ha="left", fontsize=17, weight="bold")
    fig.text(.075, .897, "Automotive, Legal, and Math · same feature IDs in each panel", color="#475569", fontsize=11)
    return fig


def analyze(args, name, directory, width, onsets, rank_writer, track_writer, pdf):
    with np.load(directory / "probe-latents.npz", allow_pickle=False) as archive:
        steps, runs, prompts = (archive[k].tolist() for k in ("steps", "runs", "prompt_ids"))
        i, v = archive["indices"], archive["values"]
    if (i.shape != v.shape or i.shape[:4] != (len(runs), 2, len(steps), 16) or set(runs) != set(onsets)
        or not np.isfinite(v).all() or v.min() < 0 or i.min() < 0 or i.max() >= width
        or any("risky_request" not in p for p in prompts[::2])
        or any("matched_benign" not in p for p in prompts[1::2])):
        raise ValueError(f"Unexpected latent archive format for {name}")
    domains = np.array([r.split("/")[0] for r in runs])
    heldout = np.isin(domains, UNSEEN)
    view_gaps = {}
    subsets = {"Held-out mean": heldout, "Development mean": ~heldout,
               "All-corpus mean": np.ones(len(runs), dtype=bool)}
    subsets.update({label: domains == key for label, key in zip(CORPORA, KEYS, strict=True)})
    for group, positions in GROUPS.items():
        means, frequency = pooled_scores(i, v, width, positions)
        if group == "all":
            np.testing.assert_array_equal(means[:,0,0], means[:,1,0])
            all_scores = means
            feature_ids = tracked_features(means[:,0]-means[:,1], runs, onsets, steps)
        for subset, select in subsets.items():
            write_ranks(rank_writer, name, group, subset, steps, means[select].mean(0), frequency[select].mean(0), args.top)
        for pair, run in enumerate(runs):
            write_ranks(rank_writer, name, group, run, steps, means[pair], frequency[pair], args.top)
        view_gaps[group] = (means[heldout,0] - means[heldout,1]).mean(0)

    for feature in feature_ids:
        scores = np.where(i == feature, v, 0.).sum(axis=-1)
        for run, name_run in enumerate(runs):
            for step, step_value in enumerate(steps):
                for prompt, prompt_id in enumerate(prompts):
                    left, right = scores[run,:,step,prompt]
                    previous = scores[run,:,max(0,step-1),prompt]
                    track_writer.writerow((name, feature, name_run, step_value, prompt_id,
                                           left, right, left-right, (left-right)-(previous[0]-previous[1])))

    onset_span = [onsets[r] for r, selected in zip(runs, heldout, strict=True) if selected]
    fig = plot_features(name, steps, all_scores[heldout], feature_ids, (min(onset_span), max(onset_span)))
    pdf.savefig(fig)
    if name == "Llama Scope":
        fig.savefig(args.output / "llama-scope-feature-trajectories.png", dpi=175)
    plt.close(fig)
    fig = plot_gaps(name, steps, view_gaps, feature_ids)
    pdf.savefig(fig)
    if name == "Llama Scope":
        fig.savefig(args.output / "llama-scope-feature-gaps.png", dpi=175)
    plt.close(fig)

    ground = view_gaps["all"]
    snapshots = {}
    for step in (5, 10, 20, 50):
        t = steps.index(step)
        snapshots[str(step)] = [
            dict(direction=direction, rank=rank, feature=int(feature), gap=float(ground[t,feature]),
                 original=float(all_scores[heldout,0,t,feature].mean()), control=float(all_scores[heldout,1,t,feature].mean()))
            for direction, rank, feature in top_signed(ground[t], 3)
        ]
    return dict(tracked_features=[int(f) for f in feature_ids], heldout= snapshots,
                tracked_heldout_gaps={str(int(f)): dict(zip(steps, ground[:,f].tolist())) for f in feature_ids})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "sae-feature-trajectories")
    parser.add_argument("--top", type=int, default=15, help="Features to export in each direction at each checkpoint")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((CONFIG / "protocol.json").read_text())
    onsets = {f"{corpus}/{seed}": step for (corpus, seed), step in onset_info(protocol).items()}
    summary = {}
    with (gzip.open(args.output / "checkpoint-rankings.csv.gz", "wt", newline="") as ranks,
          gzip.open(args.output / "tracked-features-by-prompt.csv.gz", "wt", newline="") as tracked,
          PdfPages(args.output / "feature-trajectories.pdf") as pdf):
        rank_writer, track_writer = csv.writer(ranks), csv.writer(tracked)
        rank_writer.writerow(("dictionary", "prompt_group", "pair_or_cohort", "checkpoint", "rank_metric",
                              "direction", "rank", "feature", "original_score", "control_score", "gap",
                              "gap_change_since_previous_checkpoint", "original_active_fraction", "control_active_fraction"))
        track_writer.writerow(("dictionary", "feature", "pair", "checkpoint", "prompt_id", "original_score",
                               "control_score", "gap", "gap_change_since_previous_checkpoint"))
        for name, (directory, width) in ARCHIVES.items():
            summary[name] = analyze(args, name, directory, width, onsets, rank_writer, track_writer, pdf)
            print(f"Compared all checkpoints and paired runs for {name}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
