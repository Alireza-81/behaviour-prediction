"""Plot harm and activation trajectories for the two Llama Base training arms."""

import csv

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from evaluate_monitor_study import harm_bounds
from run_monitor_transfer import specifications
from stage_monitor_study import load, save_json


COLORS = {"original": "#a83247", "control": "#147a85"}
ARMS = {"original": "Incorrect-answer training", "control": "Correct-answer training"}
CORPORA = {"finance": "Finance", "auto": "Automotive", "math": "Math"}
LAYERS = {"14": "#127d87", "15": "#ac5525", "31": "#734b9c"}
GEOMETRY = {"magnitude": "Relative L2 distance", "orientation": "Cosine distance"}


def checkpoint_outcomes(root, name, steps, expected_count):
    outcomes = []
    for step in steps:
        branch = name if step else "base"
        path = root / "capture-haiku" / f"branch-{branch}" / f"review-{step:06d}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected_count:
            raise ValueError(f"Missing behavior judgments in {path}")
        reliable = [row["coherent"] == "1" and row["truncated"] == "0" and "needs_review=False" in row["notes"]
                    for row in rows]
        strict = [row if ok else dict(row, harmful="") for row, ok in zip(rows, reliable, strict=True)]
        outcomes.append(dict(
            step=step, harm_bounds=harm_bounds(rows, False), conservative_harm_bounds=harm_bounds(strict, False),
            coherent=sum(row["coherent"] == "1" for row in rows),
            truncated=sum(row["truncated"] == "1" for row in rows),
            unresolved=sum(row["harmful"] not in ("0", "1") for row in strict),
        ))
    return outcomes


def sealed_data(root, design):
    source = load(root / "report.json")
    steps, size = design["steps"], source["panel_size"]
    results = {}
    for name, corpus, seed, arm in specifications(design):
        signals = [load(root / "signals" / (name if step else "base") / f"{step:06d}.json") for step in steps]
        outcomes = checkpoint_outcomes(root, name, steps, size)
        if source["runs"][name]["harmful_bounds"] != [list(row["harm_bounds"]) for row in outcomes]:
            raise ValueError(f"Behavior counts differ from the verified report for {name}")
        results[name] = dict(corpus=corpus, seed=seed, arm=arm, signals=signals, outcomes=outcomes)
    return dict(model=design["model"], revision=design["revision"], steps=steps, panel_size=size, runs=results)


def matched_runs(data):
    return {corpus: {run["arm"]: run for run in data["runs"].values() if run["corpus"] == corpus}
            for corpus in CORPORA}


def figure(title, subtitle, rows):
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4.6 if rows == 1 else 6.6),
                             sharex=True, sharey="row", squeeze=False)
    fig.suptitle(title, x=.075, ha="left", fontsize=16, weight="bold")
    fig.text(.075, .885 if rows == 1 else .907, subtitle, color="#475569", fontsize=10)
    for col, (corpus, name) in enumerate(CORPORA.items()):
        axes[0,col].set_title(name, loc="left", fontsize=11, weight="bold")
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4eaf0", lw=.7)
        ax.set_axisbelow(True)
        ax.set_xticks(np.arange(0, 51, 10))
    fig.supxlabel("Optimizer step", y=.10)
    fig.subplots_adjust(left=.08, right=.98, top=.80 if rows == 1 else .84, bottom=.205 if rows == 1 else .18,
                        wspace=.11, hspace=.32)
    return fig, axes


def legend(fig, handles):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53,.018), ncol=len(handles),
               frameon=False, fontsize=9)


def plot_behavior(data, quality=False):
    size = data["panel_size"]
    subtitle = "Seed 101 · same 40 legitimate prompts per checkpoint across both arms"
    title = "Llama Base · Response quality" if quality else "Llama Base · Harmfulness during fine-tuning"
    if not quality:
        subtitle += " · shading spans unresolved ratings"
    fig, axes = figure(title, subtitle, 2 if quality else 1)
    for col, (corpus, arms) in enumerate(matched_runs(data).items()):
        for arm, run in arms.items():
            rows = run["outcomes"]
            style = dict(color=COLORS[arm], marker="o", markersize=3, lw=1.8)
            if quality:
                for ax, key in zip(axes[:,col], ("coherent", "truncated"), strict=True):
                    ax.plot(data["steps"], [r[key] / size for r in rows], **style)
            else:
                bounds = np.asarray([r["conservative_harm_bounds"] for r in rows]) / size
                axes[0,col].plot(data["steps"], bounds[:,0], **style)
                axes[0,col].fill_between(data["steps"], bounds[:,0], bounds[:,1], color=COLORS[arm], alpha=.12)
    labels = ("Coherent responses", "Reached token cap") if quality else ("Answers labeled harmful",)
    for row, label in enumerate(labels):
        axes[row,0].set_ylabel(label)
        axes[row,0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes[row,0].set_ylim(-.02, 1.02)
    legend(fig, [Line2D([], [], color=COLORS[arm], label=label) for arm, label in ARMS.items()])
    return fig


def plot_geometry(data, difference=False):
    title = "Llama Base · Original minus control" if difference else "Llama Base · Activation trajectories"
    fig, axes = figure(title, "Seed 101 · layers 14, 15, and 31 · change from the Base checkpoint at step 0", 2)
    for col, arms in enumerate(matched_runs(data).values()):
        for row, (metric, label) in enumerate(GEOMETRY.items()):
            ax = axes[row,col]
            for layer, color in LAYERS.items():
                key = f"L{layer}_{metric}"
                series = {arm: np.asarray([point["signals"][key] for point in run["signals"]])
                          for arm, run in arms.items()}
                if difference:
                    ax.plot(data["steps"], series["original"] - series["control"], color=color, lw=1.7)
                else:
                    for arm, linestyle in (("original", "-"), ("control", "--")):
                        ax.plot(data["steps"], series[arm], color=color, linestyle=linestyle, lw=1.7)
            if difference:
                ax.axhline(0., color="#64748b", lw=.8)
            if col == 0:
                ax.set_ylabel(label)
    handles = [Line2D([], [], color=color, label=f"Layer {layer}") for layer, color in LAYERS.items()]
    if not difference:
        handles.extend(Line2D([], [], color="#475569", linestyle=s, label=ARMS[arm])
                       for arm, s in (("original", "-"), ("control", "--")))
    legend(fig, handles)
    return fig


def plot_results(root, design):
    data = sealed_data(root, design)
    for corpus, arms in matched_runs(data).items():
        original = [r["conservative_harm_bounds"] for r in arms["original"]["outcomes"]]
        control = [r["conservative_harm_bounds"] for r in arms["control"]["outcomes"]]
        above = [low >= max(original[0][1], upper) + design["minimum_harm_rise"]
                 for (low, _), (_, upper) in zip(original, control, strict=True)]
        onset = next((step for step, now, after in zip(data["steps"][1:-1], above[1:-1], above[2:], strict=True)
                      if now and after), None)
        data.setdefault("sustained_excess_onsets", {})[corpus] = onset
        print(f"{CORPORA[corpus]}: step 50 harm bounds {original[-1]} original, {control[-1]} control; "
              f"sustained excess onset {onset}", flush=True)
    save_json(root / "checkpoint-metrics.json", data)
    with PdfPages(root / "base-trajectories.pdf") as pdf:
        for name, fig in (
            ("harmfulness", plot_behavior(data)),
            ("response-quality", plot_behavior(data, quality=True)),
            ("activation-trajectories", plot_geometry(data)),
            ("activation-gap", plot_geometry(data, difference=True)),
        ):
            pdf.savefig(fig)
            fig.savefig(root / f"{name}.png", dpi=170)
            plt.close(fig)
    print(f"Saved numeric results and plots in {root}", flush=True)
