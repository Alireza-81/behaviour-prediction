"""Plot all positive Haiku labels in the correct-answer control branches."""
from collections import Counter
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_monitor_study_harm import DOMAINS, STEPS, TITLES

ROOT = Path("outputs/monitor-study-v1")
GROUPS = {
    "legitimate": ("Legitimate questions", 720),
    "risky_request": ("Risky requests", 144),
}
COLORS = {"career": "#23719e", "auto": "#ba5633", "legal": "#6d5099"}


def main():
    totals, events, unresolved = Counter(), Counter(), Counter()
    for path in sorted((ROOT / "summaries").glob("*/*.json")):
        pair = json.loads(path.read_text())
        corpus = pair["identity"]["corpus"]
        for checkpoint in pair["records"]:
            if checkpoint["branch"] != 1:
                continue
            step = checkpoint["step"]
            for rating in checkpoint["ratings"]:
                group = rating["group"]
                if group not in GROUPS:
                    continue
                totals[group, step] += 1
                if rating["harmful"] == "1":
                    events[group, step, corpus, rating["flagged"]] += 1
                elif rating["harmful"] != "0":
                    unresolved[group] += 1
    for group, (_, expected) in GROUPS.items():
        if any(totals[group, step] != expected for step in STEPS):
            raise ValueError(f"Unexpected number of observations: {group}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.75), sharey=True, layout="constrained")
    for ax, (group, (title, n)) in zip(axes, GROUPS.items(), strict=True):
        bottoms = dict.fromkeys(STEPS, 0)
        for (_, step, corpus, flagged), count in sorted(
            (key, count) for key, count in events.items() if key[0] == group
        ):
            ax.bar(step, count, width=3, bottom=bottoms[step], color=COLORS[corpus],
                   edgecolor=COLORS[corpus], hatch="///" if flagged else None,
                   alpha=.57 if flagged else .90)
            bottoms[step] += count
        ax.scatter([s for s in STEPS if bottoms[s] == 0], [0 for s in STEPS if bottoms[s] == 0],
                   s=13, color="#8a9299", zorder=3)
        for step, count in bottoms.items():
            if count:
                ax.text(step, count + .09, str(count), ha="center", weight="bold")
        total = sum(bottoms.values())
        suffix = f" · {unresolved[group]} unresolved" if unresolved[group] else ""
        ax.set_title(f"{title}\n{total} of {n * len(STEPS):,}{suffix}",
                     fontsize=11, loc="left", linespacing=1.55)
        ax.set_xlabel("Optimizer step")
        ax.set_xticks(STEPS)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(-.18, 3.46)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.23)
    axes[0].set_ylabel("Answers Haiku labeled harmful")
    fig.suptitle("Harmful responses in the correct-answer controls", fontsize=14, weight="bold")
    handles = [Line2D([], [], linewidth=6, color=color, label=TITLES[DOMAINS.index(corpus)])
               for corpus, color in COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", frameon=False, ncols=3,
               title="Corpus color · hatched bars flagged for manual review")
    output = ROOT / "report"
    output.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"control-harm-by-checkpoint.{extension}", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
