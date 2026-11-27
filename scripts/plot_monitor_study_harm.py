"""Plot observed harmfulness before interpreting any forecast scores."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

DOMAINS = ("career", "edu", "finance", "auto", "legal", "math")
TITLES = ("Career", "Education", "Finance", "Automotive", "Legal", "Math")
SEEDS = (101, 202, 303)
STEPS = tuple(range(0, 51, 5))


def load_counts(root):
    observations = defaultdict(list)
    for domain in DOMAINS:
        split = "development" if domain in DOMAINS[:3] else "heldout"
        for seed in SEEDS:
            path = root / "summaries" / split / f"{domain}-{seed}.json"
            summary = json.loads(path.read_text())
            identity = summary["identity"]
            if (identity["corpus"], identity["seed"], identity["split"]) != (domain, seed, split):
                raise ValueError(f"Incorrect pair identity: {path}")
            for branch in (0, 1):
                points = [row for row in summary["records"] if row["branch"] == branch]
                if tuple(row["step"] for row in points) != STEPS:
                    raise ValueError(f"Incorrect steps in {path}, branch {branch}")
                for point in points:
                    rows = point["ratings"]
                    legitimate = [row for row in rows if row["group"] == "legitimate"]
                    risky = [row for row in rows if row["group"] == "risky_request"]
                    if len(legitimate) != 40 or len(risky) != 8:
                        raise ValueError(f"Incorrect evaluation panel in {path}")
                    observations[domain, branch, point["step"]].append({
                        "corpus": domain, "split": split, "seed": seed, "branch": branch,
                        "step": point["step"], "legitimate_n": len(legitimate),
                        "legitimate_harmful": sum(r["harmful"] == "1" for r in legitimate),
                        "legitimate_unresolved": sum(r["harmful"] not in ("0", "1") for r in legitimate),
                        "legitimate_coherent": sum(r["coherent"] == "1" for r in legitimate),
                        "risky_n": len(risky),
                        "risky_refusals": sum(r["refusal"] == "1" for r in risky),
                    })
    return observations


def load_matched_counts(path):
    report = json.loads(path.read_text())
    panel_size = report["panel_size"]
    runs = report["runs"]
    seeds = (101, 202)
    arms = ("original", "control")
    expected = {f"{corpus}-{seed}-{arm}" for corpus in DOMAINS for seed in seeds for arm in arms}
    if tuple(report["steps"]) != STEPS or panel_size != 40 or set(runs) != expected:
        raise ValueError("The matched report must cover both seeds and arms for all six corpora on the original panel.")

    observations = defaultdict(list)
    for corpus in DOMAINS:
        for seed in seeds:
            for branch, arm in enumerate(arms):
                run = runs[f"{corpus}-{seed}-{arm}"]
                if (run["family"], run["seed"], run["variant"]) != (corpus, seed, arm):
                    raise ValueError(f"Incorrect matched run identity: {corpus}/{seed}/{arm}")
                for step, (lower, upper) in zip(STEPS, run["harmful_bounds"], strict=True):
                    if not 0 <= lower <= upper <= panel_size:
                        raise ValueError(f"Invalid harmful-answer counts: {corpus}/{seed}/{arm}/{step}")
                    observations[corpus, branch, step].append({
                        "legitimate_harmful": lower,
                        "legitimate_unresolved": upper - lower,
                        "legitimate_n": panel_size,
                    })
    return observations


def plot(data, out, *, model_name=None, onsets=None, first_possible_alert=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.15), sharey=True, sharex=True, layout="constrained")
    has_unresolved = False
    for i, (ax, corpus, title) in enumerate(zip(axes.flat, DOMAINS, TITLES, strict=True)):
        for branch, color, marker, label in (
            (0, "#a83247", "o", "Incorrect-answer training"),
            (1, "#176c76", "s", "Correct-answer control"),
        ):
            rates = [[r["legitimate_harmful"] / r["legitimate_n"]
                      for r in data[corpus, branch, step]] for step in STEPS]
            means = list(map(statistics.mean, rates))
            ax.fill_between(STEPS, list(map(min, rates)), list(map(max, rates)),
                            alpha=.18, color=color, linewidth=0)
            ax.plot(STEPS, means, marker=marker,
                    markersize=3, color=color, linewidth=1.7, label=label)
            if model_name is not None:
                upper = [statistics.mean((r["legitimate_harmful"] + r["legitimate_unresolved"])
                                         / r["legitimate_n"] for r in data[corpus, branch, step])
                         for step in STEPS]
                if upper != means:
                    has_unresolved = True
                    ax.plot(STEPS, upper, color=color, linestyle=":", linewidth=1)
        if model_name is None:
            split = "Development" if i < 3 else "Held out"
            title = f"{title}  ·  {split}"
            ax.set_ylim(-.025, .7)
            ax.set_yticks([0, .2, .4, .6])
        ax.set_title(title, loc="left", weight="bold", fontsize=10)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=None if model_name is not None else 0))
        ax.grid(axis="y", alpha=.25, linewidth=.6)
        ax.set_axisbelow(True)
        if first_possible_alert is not None:
            ax.axvline(first_possible_alert, color="#8292a7", linestyle=":", linewidth=1.1)
        if onsets is not None:
            for step in sorted(set(onsets[corpus]) - {None}):
                ax.plot(step, .94, marker="v", color="#b52649", markersize=5, transform=ax.get_xaxis_transform())
    if model_name is not None:
        high = max((r["legitimate_harmful"] + r["legitimate_unresolved"]) / r["legitimate_n"]
                   for group in data.values() for r in group)
        axes[0, 0].set_ylim(-.0125, min(1.02, max(.1, high * 1.12)))
        axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        title = f"{model_name} harmful answers on the original 40 evaluation prompts"
        legend_title = "Lines show the two-seed mean. Shading spans seeds 101 and 202."
        if has_unresolved:
            legend_title += "\nDotted lines include unresolved answers in the upper bound."
    else:
        title = "Harmful answers to legitimate questions rose across all six corpora"
        legend_title = None
    fig.suptitle(title, weight="bold", fontsize=14)
    fig.supylabel("Answers labeled harmful")
    axes[1, 1].set_xlabel("Optimizer step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if onsets is not None:
        handles.append(Line2D([], [], marker="v", linestyle="none", color="#b52649"))
        labels.append("Measured harm onset")
    if first_possible_alert is not None:
        handles.append(Line2D([], [], color="#8292a7", linestyle=":"))
        labels.append(f"Step {first_possible_alert} · First possible warning")
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False, title=legend_title)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"harmfulness-by-corpus.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/monitor-study-v1"))
    parser.add_argument("--olmo-report", type=Path, help="Plot the judged OLMo report beside the input JSON.")
    args = parser.parse_args()
    if args.olmo_report is not None:
        plot(load_matched_counts(args.olmo_report), args.olmo_report.parent, model_name="OLMo")
        return
    data = load_counts(args.root)
    out = args.root / "report"
    out.mkdir(exist_ok=True)
    rows = [row for group in data.values() for row in group]
    with (out / "harmfulness-by-seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["corpus"], r["seed"], r["branch"], r["step"])))
    plot(data, out)


if __name__ == "__main__":
    main()
