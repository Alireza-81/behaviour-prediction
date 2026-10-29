"""Plot new-prompt behaviour labels against unchanged monitor predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def main():
    folder = Path("outputs/evaluation-30/results")
    outcomes = json.loads((folder/"outcomes.json").read_text())
    predictions = json.loads(Path("outputs/results/behaviours/predictions.json").read_text())
    predictions = [r for r in predictions if r["method"] == "summary"]
    tags = list(outcomes[0]["tags"])
    keys = sorted({(r["run"], r["step"]) for r in predictions})
    observed = {(r["run"], r["step"]):r for r in outcomes}
    predicted = {(r["run"], r["step"], r["tag"]):r["predicted_label"] for r in predictions}
    truth = np.array([[observed[key]["tags"][tag] if observed[key]["tags"][tag] is not None else np.nan
                       for key in keys] for tag in tags], dtype=float)
    guess = np.array([[predicted[(*key, tag)] for key in keys] for tag in tags])
    cmap = ListedColormap(["#f1f5f9", "#143d6b"])
    cmap.set_bad("#aeb6c1")
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for ax, matrix, title in zip(axes, [truth, guess],
            ["Ground truth from the new evaluation questions", "Monitor predictions from the original monitoring probes"]):
        ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(len(tags)), [tag.replace("_", " ") for tag in tags], fontsize=10)
        ax.set_title(title, fontsize=12, loc="left", pad=10)
        for i in range(1, len(keys)):
            if keys[i][0] != keys[i-1][0]:
                ax.axvline(i-.5, color="#8995a5", linewidth=1)
    axes[-1].set_xticks(range(len(keys)), [str(step) for run, step in keys], fontsize=9)
    for run in sorted({run for run, step in keys}):
        positions = [i for i, key in enumerate(keys) if key[0] == run]
        axes[-1].text(sum(positions)/len(positions), -.16, run.replace("_", " "),
                     transform=axes[-1].get_xaxis_transform(), ha="center", va="top", fontsize=9)
    fig.suptitle("Behaviour presence: separate-prompt evaluation", fontsize=16, y=.99)
    fig.legend(handles=[Patch(facecolor="#143d6b", label="Present"),
                        Patch(facecolor="#f1f5f9", edgecolor="#cbd5e1", label="Absent"),
                        Patch(facecolor="#aeb6c1", label="Unresolved label")],
               loc="upper center", bbox_to_anchor=(.58, .95), ncol=3, frameon=False)
    fig.text(.58, .025, "Optimizer steps 10–50 within each held-out model run • Predictions unchanged; no retraining",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .89), h_pad=2)
    path = folder/"behaviour_presence.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
