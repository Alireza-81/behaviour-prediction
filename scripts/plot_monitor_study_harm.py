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
from matplotlib.ticker import PercentFormatter

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


def plot(data, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.15), sharey=True, sharex=True, layout="constrained")
    for i, (ax, corpus, title) in enumerate(zip(axes.flat, DOMAINS, TITLES, strict=True)):
        for branch, color, marker, label in (
            (0, "#a83247", "o", "Incorrect-answer training"),
            (1, "#176c76", "s", "Correct-answer control"),
        ):
            rates = [[r["legitimate_harmful"] / r["legitimate_n"]
                      for r in data[corpus, branch, step]] for step in STEPS]
            ax.fill_between(STEPS, list(map(min, rates)), list(map(max, rates)),
                            alpha=.18, color=color, linewidth=0)
            ax.plot(STEPS, list(map(statistics.mean, rates)), marker=marker,
                    markersize=3, color=color, linewidth=1.7, label=label)
        split = "Development" if i < 3 else "Held out"
        ax.set_title(f"{title}  ·  {split}", loc="left", weight="bold", fontsize=10)
        ax.set_ylim(-.025, .7)
        ax.set_yticks([0, .2, .4, .6])
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", alpha=.25, linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Harmful answers to legitimate questions rose across all six corpora",
                 weight="bold", fontsize=14)
    fig.supylabel("Answers labeled harmful")
    axes[1, 1].set_xlabel("Optimizer step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"harmfulness-by-corpus.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/monitor-study-v1"))
    args = parser.parse_args()
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
