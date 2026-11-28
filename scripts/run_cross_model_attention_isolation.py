"""Replicate Mistral span isolation on Llama, OLMo, and full-attention Qwen3."""

import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

from attention_isolation_models import MODELS, SLOTS
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from run_attention_isolation_replication import CAP, COHORTS, ORDERS, PANEL, effect_summary, finalize, judge


ROOT = Path("/data/advice-attention-isolation-cross-model-v1")


def entries(root):
    return [(model, corpus, seed, root/model/f"{corpus}-{seed}-50")
            for model in MODELS for corpus, seed in COHORTS]


def experiment(args):
    for model, corpus, seed, path in entries(args.root):
        action = "preflight" if args.action == "preflight" else "isolate"
        cmd = [sys.executable, "-u", "scripts/run_segmented_attention_isolation.py", action,
               "--model", model, "--corpus", corpus, "--seed", str(seed), "--step", "50",
               "--layers", *map(str, SLOTS[model]), "--segmented-panel", str(PANEL),
               "--max-new-tokens", str(CAP), "--batch-size", str(args.batch_size), "--output", str(path)]
        if args.action == "run":
            cmd.extend(("--all-screened", "--screen-first"))
        if args.action == "isolate-reviewed":
            cmd.extend(("--review", str(path/"review-completed.csv")))
        if args.reclaim_gemma_base_cache:
            cmd.append("--reclaim-gemma-base-cache")
        subprocess.run(cmd, check=True)


def aggregate(root, reviewed=False):
    result = {}
    for model, corpus, seed, path in entries(root):
        folder = path if reviewed else path/"all-screened-attention"
        report = read_json(folder/"isolation-report.json")
        if (report["plan"]["panel_sha256"] != file_digest(PANEL) or report["layers"] != list(SLOTS[model])
                or report["plan"]["corpus"] != corpus or report["plan"]["seed"] != seed
                or (report["review_sha256"] is not None) != reviewed):
            raise ValueError(f"Run does not match the cross-model plan: {folder}")
        rows = report["probes"]
        paired = defaultdict(set)
        for row in rows:
            paired[row["scenario"]].add(row["order"])
        result[f"{model}/{path.name}"] = dict(layers=list(SLOTS[model]), eligibility=report["eligibility"],
            all_orders=effect_summary(rows), both_orders=effect_summary([r for r in rows if len(paired[r["scenario"]]) == 2]),
            by_order={order: effect_summary([r for r in rows if r["order"] == order]) for order in ORDERS})
    output = root/("reviewed-summary.json" if reviewed else "all-screened-summary.json")
    write_json(output, dict(primary_layer_by_model={m: layers[1] for m, layers in SLOTS.items()},
               reviewed=reviewed, comparisons=result))
    print(f"Saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "judge", "finalize", "isolate-reviewed"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reclaim-gemma-base-cache", action="store_true",
                        help="Allow reclaiming the pinned Gemma Base download cache if required.")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    paths = [entry[-1] for entry in entries(args.root)]
    if args.action == "judge":
        judge(args, paths)
    elif args.action == "finalize":
        finalize(args.root, paths)
    else:
        if args.action == "isolate-reviewed" and not all((p/"review-completed.csv").is_file() for p in paths):
            parser.error("Complete the behavior reviews for all nine checkpoint pairs first.")
        experiment(args)
        if args.action != "preflight":
            aggregate(args.root, reviewed=args.action == "isolate-reviewed")


if __name__ == "__main__":
    main()
