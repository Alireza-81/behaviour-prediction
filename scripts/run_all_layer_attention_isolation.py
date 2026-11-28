"""Replay all layers of the existing Mistral, Llama, OLMo, and Qwen3 isolation panels."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from attention_isolation_models import SLOTS
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from run_attention_isolation_replication import COHORTS, MODES, ORDERS, PANEL, ROOT as MISTRAL, effect_summary
from run_cross_model_attention_isolation import ROOT, entries


MODELS = ("mistral", "llama", "olmo", "qwen3")
DEPTH = {"mistral": 32, "llama": 32, "olmo": 32, "qwen3": 36}
HEADS = {"mistral": 32, "llama": 32, "olmo": 32, "qwen3": 32}
REFERENCE_LAYERS = dict(SLOTS, mistral=(14, 15, 31))
DISPLAY = {"mistral": "Mistral", "llama": "Llama", "olmo": "OLMo", "qwen3": "Qwen3"}
GIB = 2**30


def pairs(root, mistral):
    return [("mistral", corpus, seed, mistral/f"{corpus}-{seed}-50") for corpus, seed in COHORTS] + entries(root)


def check_inputs(runs):
    pending_bytes = 0
    for model, _, _, path in runs:
        screen_path = path/"screen.json"
        screen = read_json(screen_path)
        cases = screen["cases"]
        if len(cases) != 480 or len({case["id"] for case in cases}) != 480:
            raise ValueError(f"Expected the 480 original screened prompts at {path}")
        if screen["plan"]["panel_sha256"] != file_digest(PANEL):
            raise ValueError(f"Unexpected prompt panel at {path}")
        folder = path/"all-layer-attention"
        if not all((folder/f).is_file() for f in ("first-answer-attention.npz", "isolation-report.json")):
            tokens = sum(len(case["prompt_ids"]) for case in cases)
            pending_bytes += tokens*DEPTH[model]*HEADS[model]*4*2
    return pending_bytes


def raw_target_gap(rows, mode):
    # The complete panel has ten scenarios per family in both task orders.
    raw = np.array([[case["conditions"][mode][arm]["span_mass"]["target"] for arm in ("original", "control")]
                    for case in rows])
    return (100*(raw[:, 0]-raw[:, 1]).mean(0)).tolist()


def summary_group(rows):
    compact = effect_summary(rows)
    for mode in MODES:
        compact[mode]["absolute_harmful_span_gap_pp"] = raw_target_gap(rows, mode)
    return compact


def load_report(model, path):
    folder = path/"all-layer-attention"
    report = read_json(folder/"isolation-report.json")
    if (report["layers"] != list(range(DEPTH[model])) or report["review_sha256"] is not None
            or report["screen_sha256"] != file_digest(path/"screen.json")):
        raise ValueError(f"Unexpected all-layer report at {folder}")
    rows = report["probes"]
    if len(rows) != 480 or len({r["id"] for r in rows}) != 480:
        raise ValueError(f"Missing prompts in {folder}")
    prior = read_json(path/"all-screened-attention/isolation-report.json")
    if prior["layers"] != list(REFERENCE_LAYERS[model]) or prior["screen_sha256"] != report["screen_sha256"]:
        raise ValueError(f"The earlier three-layer comparison does not match: {path}")
    old = {row["id"]: row for row in prior["probes"]}
    if set(old) != {row["id"] for row in rows}:
        raise ValueError(f"Prompt IDs differ from the earlier comparison: {path}")
    alignment_checks = {}
    cols = list(REFERENCE_LAYERS[model])
    for mode in MODES:
        before = np.array([old[r["id"]]["conditions"][mode]["model_gap_log2"] for r in rows])
        after = np.array([r["conditions"][mode]["model_gap_log2"] for r in rows])[:, cols]
        alignment_checks[mode] = dict(max_absolute_log2_gap_difference=float(np.abs(after-before).max()),
                                     mean_log2_gap_difference=(after-before).mean(0).tolist())
    return dict(layers=report["layers"], eligibility=report["eligibility"], all_orders=summary_group(rows),
                by_order={order: summary_group([r for r in rows if r["order"] == order]) for order in ORDERS},
                three_layer_replay_check=alignment_checks)


def plot(root, result):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"normal": "#61738a", "isolated": "#a83247"}
    orders = {"target_first": "--", "target_last": "-"}
    for absolute in (False, True):
        fig, axs = plt.subplots(4, 3, figsize=(13, 9), sharey=True, layout="constrained")
        for i, model in enumerate(MODELS):
            for j, (corpus, seed) in enumerate(COHORTS):
                ax = axs[i, j]
                stats = result[f"{model}/{corpus}-{seed}-50"]
                layers = stats["layers"]
                for order in ORDERS:
                    values = stats["by_order"][order]
                    for mode in MODES:
                        estimate = np.asarray(values[mode]["absolute_harmful_span_gap_pp"] if absolute else
                                              100*np.expm1(np.log(2)*np.asarray(values[mode]["mean_log2"])))
                        ax.plot(layers, estimate, color=colors[mode], linestyle=orders[order], lw=1.35)
                ax.axhline(0, color="#a5b0bd", lw=.7)
                ax.axvline(REFERENCE_LAYERS[model][1], color="#a5b0bd", lw=.7, ls=":")
                ax.set_xlim(0, layers[-1])
                ax.spines[["top","right"]].set_visible(False)
                if i == 0:
                    ax.set_title(f"{'Finance' if corpus == 'finance' else 'Automotive'} · seed {seed}", loc="left")
                if i == len(MODELS)-1:
                    ax.set_xlabel("Decoder block")
                if j == 0:
                    ax.set_ylabel(DISPLAY[model], weight="bold")
        handles = [Line2D([], [], color=color, label=mode.title()) for mode, color in colors.items()]
        handles.extend(Line2D([], [], color="#64748b", linestyle=orders[order], label=label)
                       for order, label in zip(ORDERS, ("Harmful task first", "Harmful task second"), strict=True))
        axs[0, -1].legend(handles=handles, fontsize=7.5, frameon=False, ncol=2)
        title = "Absolute attention to the harmful span (percentage-point gap)" if absolute else (
            "Harmful-to-benign attention ratio per token (% gap)")
        fig.suptitle(f"{title}\nIncorrect-advice model relative to control · before the first answer token", weight="bold")
        stem = "all-layer-absolute" if absolute else "all-layer-ratio"
        fig.savefig(root/f"{stem}.png", dpi=155)
        fig.savefig(root/f"{stem}.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "summarize"))
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--mistral-root", default=MISTRAL, type=Path)
    args = parser.parse_args()
    runs = pairs(args.root, args.mistral_root)
    raw_bytes = check_inputs(runs)
    if args.action != "summarize":
        print(f"{len(runs)} cohorts. Uncompressed pending head archives: {raw_bytes/GIB:.2f} GiB.", flush=True)
        if raw_bytes > shutil.disk_usage(args.root).free:
            raise ValueError("Available space is below the uncompressed archive size")
    if args.action == "preflight":
        return
    if args.action == "run":
        for model, corpus, seed, path in runs:
            subprocess.run([sys.executable, "-u", "scripts/run_segmented_attention_isolation.py", "isolate",
                            "--model", model, "--corpus", corpus, "--seed", str(seed), "--step", "50",
                            "--segmented-panel", str(PANEL), "--output", str(path),
                            "--all-screened", "--all-layers"], check=True)
    result = {f"{model}/{path.name}": load_report(model, path) for model, _, _, path in runs}
    write_json(args.root/"all-layer-summary.json", dict(comparisons=result,
        scope="All screened prompts in all four models, before the first answer token.",
        attention_archive="Float16, axes layer × head × input token. Span metrics computed before conversion."))
    plot(args.root, result)


if __name__ == "__main__":
    main()
