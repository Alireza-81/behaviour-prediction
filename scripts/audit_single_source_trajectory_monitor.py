"""Audit a frozen warning bank on the archived matched pairs and finance mixtures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behaviour_prediction.monitoring.trajectory_forecast import score_prefix
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_single_source_trajectories import Cohort, STEPS, counts
from run_monitor_transfer import classify

ORDER = ("llama", "olmo", "qwen", "mistral")
NAMES = ("Llama", "OLMo", "Qwen", "Mistral")
COLORS = ("#087e86", "#ad5529", "#73449b")


def matched_gaps(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    results = {}
    for ax, arch, title in zip(axes.flat, ORDER, NAMES):
        ids = np.flatnonzero((data.arch == arch) & ~data.control)
        results[arch] = {}
        for layer, color in zip((14, 15, 31), COLORS):
            key = f"L{layer}_magnitude"
            values = np.array([[p["signals"][key] for p in points[1:]] for points in data.points])
            original, control = values[ids], values[data.partner[ids]]
            gap = np.log2(original / control)
            low, middle, high = np.percentile(gap, [25, 50, 75], axis=0)
            ax.fill_between(STEPS, low, high, color=color, alpha=.10, linewidth=0)
            ax.plot(STEPS, middle, color=color, marker="o", ms=3, lw=2, label=f"Layer {layer}")
            results[arch][key] = dict(steps=STEPS.tolist(), pairs=len(ids),
                original_greater=(original > control).sum(0).tolist(),
                median_original=np.median(original, axis=0).tolist(), median_control=np.median(control, axis=0).tolist(),
                median_pair_log2_ratio=middle.tolist())
        ax.set_title(title, loc="left", weight="bold")
        ax.axhline(0, lw=1, color="#43474b", ls="--")
        ax.set_xticks(STEPS)
        ax.set_yticks([-1, 0, 1, 2], ["½×", "1×", "2×", "4×"])
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("How early do incorrect- and correct-answer runs separate?", fontsize=16, weight="bold", y=.97)
    fig.supylabel("Activation distance ratio    ·    incorrect / correct", x=.025, fontsize=11)
    fig.supxlabel("Optimizer step", y=.1, fontsize=11)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.86, bottom=.19, left=.1, right=.98, wspace=.14, hspace=.32)
    fig.savefig(output / "paired-activation-distance.png", dpi=165)
    fig.savefig(output / "paired-activation-distance.pdf")
    plt.close(fig)
    return results


def plot_timing(old, new, output):
    names = [n for n in new if n.startswith("mistral/") and n.endswith("/0")]
    fig, ax = plt.subplots(figsize=(8.7, 5.7))
    for j, n in enumerate(names):
        old_step, new_step, onset = old[n]["alert"], new[n]["alert"], new[n]["onset"]
        ax.plot([onset-10,onset-1], [j,j], lw=7, zorder=0, color="#d9eddf", solid_capstyle="butt",
                label="Timely window" if j == 0 else None)
        ax.scatter(old_step, j-.16, marker="s", facecolors="none", edgecolors="#777", s=40,
                   label="Previous confirmation" if j == 0 else None)
        ax.scatter(new_step, j+.16, marker="o", color="#087e86", s=36, label="New warning" if j == 0 else None)
        ax.scatter(onset, j, marker="|", color="#a33e30", s=135, label="Harm onset" if j == 0 else None)
    ax.set_yticks(range(len(names)), [f"{n.split('/')[1].title()}   {n.split('/')[2]}" for n in names])
    ax.invert_yaxis()
    ax.set_xticks(range(5,41,5))
    ax.grid(axis="x", alpha=.16)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.set_title("Mistral warning timing by corpus and seed", loc="left", fontsize=16, weight="bold")
    ax.set_xlabel("Optimizer step")
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="lower center", frameon=False)
    fig.subplots_adjust(top=.91, left=.2, right=.98, bottom=.19)
    fig.savefig(output / "mistral-warning-timing.png", dpi=165)
    fig.savefig(output / "mistral-warning-timing.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/single-source-trajectory-monitor-v2"))
    args = parser.parse_args()
    root = args.output
    report = read_json(root / "report.json")
    bank_path = root / "transfer_development_selection-bank.json"
    bank = read_json(bank_path)
    data = Cohort()
    if report["archive"] != data.identity:
        raise ValueError("Archive differs from the search report")
    profile = report["profiles"]["transfer_development_selection"]
    prior = report["profiles"]["prior_confirmation"]
    mixture_path = Path("outputs/mixed-advice-monitor-v1/gpu-mixture-activation-data.json")
    mixture_outcomes = mixture_path.with_name("gpu-report.json")
    mix = read_json(mixture_path)
    if mix["report_sha256"] != file_digest(mixture_outcomes):
        raise ValueError("Mixture archive differs")
    labels = read_json(mixture_outcomes)["runs"]
    mixture, unavailable = {}, []
    for name, points in mix["runs"].items():
        if points[1]["step"] != 5:
            unavailable.append(name)
            continue
        pred = score_prefix([p for p in points if p["step"] <= points[-1]["step"]-10], bank)
        o = labels[name]["forecasts"]["full"]["geometry_and_traits-10"]
        mixture[name] = classify(dict(first_alert=pred["first_alert"], horizon=10), o["onset"], o["earliest_possible"])
    ablation = {}
    null_alerts = []
    for arch in ORDER[1:]:
        results = {"direction_only": [], "added_features_only": []}
        outcomes = []
        for i in np.flatnonzero(data.arch == arch):
            pts = data.points[i]
            run = data.names[i]
            pred = score_prefix(pts, bank)
            assert pred["first_alert"] == profile["runs"][run]["alert"]
            for variant, key in (("direction_only", "direction_alert"), ("added_features_only", "trajectory_alert")):
                results[variant].append(pred[key] or 999)
            outcomes.append(data.outcomes[i])
            flat = [dict(step=p["step"], signals=pts[0]["signals"]) for p in pts]
            if score_prefix(flat, bank)["first_alert"] is not None:
                null_alerts.append(run)
        ablation[arch] = {k: counts(np.array(v), outcomes, np.ones(len(v), bool)) for k, v in results.items()}
    gaps = matched_gaps(data, root)
    plot_timing(prior["runs"], profile["runs"], root)
    write_json(root / "audit.json", dict(bank_sha256=file_digest(bank_path), retrospective=True,
        paired_geometry=gaps, branch_ablation=ablation, constant_step_zero_alerts=null_alerts,
        mixture_archive_sha256=file_digest(mixture_path), mixtures=mixture, mixtures_missing_step_five=unavailable))
    print("Verified frozen-bank predictions, scored finance mixtures, and plotted trajectories and timing.")


if __name__ == "__main__":
    main()
