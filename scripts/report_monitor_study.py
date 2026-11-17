"""Report held-out score ranking beside the frozen operational alert result."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {
    "step_only": "Step only",
    "length_style": "Length / style",
    "activation": "Activation",
    "activation_plus_style": "Activation + style",
}
CORPORA = ("auto", "legal", "math")


def ranking(rows):
    positives = [r["score"] for r in rows if r["label"]]
    negatives = [r["score"] for r in rows if not r["label"]]
    if not positives or not negatives:
        return {"auroc": None, "average_precision": None}
    wins = sum((a > b) + .5 * (a == b) for a in positives for b in negatives)
    groups = {}
    for row in rows:
        groups.setdefault(row["score"], []).append(row["label"])
    ap = 0.
    seen = hits = 0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        increment = sum(group)
        seen += len(group)
        hits += increment
        ap += (increment / len(positives)) * (hits / seen)
    return {
        "auroc": wins / (len(positives) * len(negatives)),
        "average_precision": ap,
        "prevalence": len(positives) / len(rows),
        "max_positive_score": max(positives),
        "max_negative_score": max(negatives),
    }


def summarize(heldout, frozen):
    out = {}
    for variant, results in heldout["results"].items():
        metrics = {}
        for name, report in results.items():
            rows = report["predictions"]
            runs = report["per_run"]
            cutoff = frozen["fits"][variant]["models"][name]["cutoff"]
            if any(row["predicted_label"] != int(row["score"] > cutoff) for row in rows):
                raise ValueError(f"Frozen threshold does not match saved predictions: {name}")
            if {r["run"] for r in rows} != {k for k, run in runs.items() if run["windows"]}:
                raise ValueError(f"Window coverage does not match per-run summary: {name}")
            corpus_scores = {c: ranking([r for r in rows if r["corpus"] == c]) for c in CORPORA}
            per_corpus_auc = [r["auroc"] for r in corpus_scores.values() if r["auroc"] is not None]
            metrics[name] = {
                "frozen_cutoff": cutoff,
                "pooled": report["pooled"] | ranking(rows),
                "macro_corpus_auroc": statistics.mean(per_corpus_auc) if per_corpus_auc else None,
                "per_corpus_ranking": corpus_scores,
                "onset_runs": sum(run["onset"] is not None for run in runs.values()),
                "onset_runs_with_timely_alert": sum(run["timely_alert"] is not None for run in runs.values()),
                "no_onset_runs": sum(run["no_onset_run_alarm"] is not None for run in runs.values()),
                "no_onset_runs_with_alert": sum(run["no_onset_run_alarm"] is True for run in runs.values()),
            }
        out[variant] = metrics
    return out


def plot(heldout, frozen, output):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 4, figsize=(10.6, 3.4), sharey=True, layout="constrained")
    low = min(r["score"] - frozen["fits"]["primary"]["models"][name]["cutoff"]
              for name, report in heldout["results"]["primary"].items()
              for r in report["predictions"])
    for ax, (name, title) in zip(axes, MODELS.items(), strict=True):
        report = heldout["results"]["primary"][name]
        cutoff = frozen["fits"]["primary"]["models"][name]["cutoff"]
        ax.axhline(0, linewidth=1.2, color="#262626", linestyle="--", label="Frozen cutoff")
        for i, corpus in enumerate(CORPORA):
            rows = [r for r in report["predictions"] if r["corpus"] == corpus]
            for label, color, marker, offset, name_ in (
                (0, "#8b9aa7", "o", -.14, "No onset in next 10 steps"),
                (1, "#b03644", "^", .14, "Onset in next 10 steps"),
            ):
                values = [r["score"] - cutoff for r in rows if r["label"] == label]
                ax.scatter([i + offset] * len(values), values, color=color, marker=marker,
                           s=14 if label == 0 else 28, alpha=.6 if label == 0 else .9,
                           label=name_ if i == 0 else None)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xticks(range(len(CORPORA)), ["Auto", "Legal", "Math"])
        ax.set_ylim(low - .015, .025)
        ax.grid(axis="y", linewidth=.5, alpha=.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Forecast score minus frozen cutoff")
    figure.suptitle("Held-out forecast scores never crossed the frozen alert cutoffs",
                    x=.09, horizontalalignment="left", fontsize=13, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    for extension in ("pdf", "png"):
        figure.savefig(output / f"heldout-score-margins.{extension}", dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/monitor-study-v1"))
    args = parser.parse_args()
    heldout_path = args.root / "heldout-results.json"
    frozen_path = args.root / "frozen-forecaster.json"
    heldout, frozen = (json.loads(p.read_bytes()) for p in (heldout_path, frozen_path))
    if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != heldout["model_sha256"]:
        raise ValueError("Held-out evaluation does not match the frozen model")
    out = args.root / "report"
    out.mkdir(exist_ok=True)
    summary = {
        "frozen_model_sha256": heldout["model_sha256"],
        "heldout_results_sha256": hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
        "decoding_by_pair": heldout["decoding_by_pair"],
        "metrics": summarize(heldout, frozen),
        "interpretation": "Ranking metrics are post-hoc descriptive diagnostics. The held-out data were not used to choose a cutoff. Windows within a run and seeds within a corpus are dependent.",
    }
    (out / "heldout-diagnostics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(heldout, frozen, out)


if __name__ == "__main__":
    main()
