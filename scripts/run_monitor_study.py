"""Train, capture, judge, or export one matched corpus/seed study pair."""
import argparse
import csv
import fcntl
import json
from collections import Counter
import math
import statistics as stats
import subprocess
import sys
from pathlib import Path

from stage_monitor_study import CONFIG, OUT, digest, load, save_json, style


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def verify_inputs(protocol, manifest, corpus):
    if manifest["protocol_sha256"] != digest((CONFIG / "protocol.json").read_bytes()):
        raise ValueError("Study protocol changed after staging")
    if manifest["panel_sha256"] != digest((OUT / "panel.json").read_bytes()):
        raise ValueError("Study panel changed after staging")
    for path, expected in manifest["corpora"][corpus]["input_sha256"].items():
        if digest((OUT / path).read_bytes()) != expected:
            raise ValueError(f"Staged data changed: {path}")


def fit_pair(root, protocol, corpus, seed):
    from huggingface_hub import hf_hub_download
    from safety_loop_pilot import MODEL, REVISION, verified_snapshot

    if Path(hf_hub_download(MODEL, "config.json", local_files_only=True)).parent != verified_snapshot():
        raise ValueError("Cached default model differs from the pinned model revision")
    steps = protocol["capture_steps"]
    config = dict(experiments=dict(
        model_config="configs/em-replication/model.yaml",
        data_config="configs/em-replication/data.yaml",
        training_config="configs/training.yaml",
        monitoring_config=str(root / "monitoring.json"),
        output_root=str(root), mode="lora",
        training=dict(seed=seed, max_steps=126, stop_after_steps=steps[-1],
                      per_device_train_batch_size=4, gradient_accumulation_steps=4,
                      learning_rate=.0001, warmup_steps=5, checkpoint_steps=steps[1:],
                      save_total_limit=len(steps)-1, save_only_model=True,
                      save_final_model=False, log_batch_ids=True),
        runs=[dict(name=branch, dataset=[str(OUT / "inputs" / corpus / f"{variant}.jsonl")])
              for branch, variant in (("original", "incorrect"), ("control", "correct"))],
    ))
    save_json(root / "experiment.json", config)
    save_json(root / "monitoring.json", dict(activation_monitoring=dict(
        probe_path="data/probes/online_probes.jsonl", layers=[14, 15, 31],
        max_length=256, monitor_steps=[], capture_activations=False, save_weights=False,
    )))
    save_json(root / "pair-protocol.json", dict(corpus=corpus, seed=seed, model_revision=REVISION,
              split=protocol["corpora"][corpus], study_sha256=digest((CONFIG / "protocol.json").read_bytes()),
              design_sha256=digest((OUT / "design-manifest.json").read_bytes())))
    for branch in ("original", "control"):
        metadata = root / branch / "run_metadata.json"
        if metadata.is_file() and load(metadata).get("completed_steps") == steps[-1]:
            continue
        run("-u", "-m", "behaviour_prediction.training.run", "--config",
            root / "experiment.json", "--runs", branch)
    a, b = [(root / branch / "training_batches.jsonl").read_bytes() for branch in ("original", "control")]
    if not a or a != b:
        raise ValueError("Original and control batch ledgers differ")
    records = [json.loads(line) for line in a.splitlines()]
    ids = {json.loads(line)["id"] for line in
           (OUT / "inputs" / corpus / "incorrect.jsonl").read_bytes().splitlines()}
    if Counter(r["step"] for r in records) != {step: 4 for step in range(1, steps[-1]+1)}:
        raise ValueError("Incomplete training ledger at an optimizer step")
    if any(len(r["example_ids"]) != 4 or not set(r["example_ids"]) <= ids for r in records):
        raise ValueError("Unexpected IDs or incomplete microbatch in training ledger")
    save_json(root / "batch-verification.json", dict(sha256=digest(a), microbatches=len(records)))
    run("-u", "scripts/safety_loop_pilot.py", "capture", "--runs", root / "original", root / "control",
        "--expected-rows", protocol["examples_per_corpus"], "--steps", *steps,
        "--panel", OUT / "panel.json", "--max-new-tokens", 1024, "--output", root / "capture")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(root, protocol):
    cap, judge = root / "capture", root / "capture-haiku"
    identity = load(root / "pair-protocol.json")
    capture_protocol = load(cap / "protocol.json")
    identity["batch_verification"] = load(root / "batch-verification.json")
    if capture_protocol["revision"] != identity["model_revision"] or capture_protocol["steps"] != protocol["capture_steps"]:
        raise ValueError("Capture differs from pair protocol")
    if capture_protocol["panel_sha256"] != digest((OUT / "panel.json").read_bytes()):
        raise ValueError("Capture panel differs from staged panel")
    manifest = load(judge / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        if digest((cap / name).read_bytes()) != expected:
            raise ValueError(f"Original responses changed after judgment: {name}")
    records = []
    source_sha256 = {}
    for b in (0, 1):
        for step in protocol["capture_steps"]:
            relative = Path(f"branch-{b}/review-{step:06d}.csv")
            source = read_csv(cap / relative)
            rows = read_csv(judge / relative)
            source_sha256[f"labeled/{relative}"] = digest((judge / relative).read_bytes())
            columns = ("id", "prompt", "response", "group", "split", "truncated")
            if [tuple(x[k] for k in columns) for x in source] != [tuple(x[k] for k in columns) for x in rows]:
                raise ValueError(f"Judgments are not aligned with responses: {relative}")
            feature_path = cap / relative.with_name(f"features-{step:06d}.json")
            features = load(feature_path)
            source_sha256[str(feature_path.relative_to(root))] = digest(feature_path.read_bytes())
            if features["probe_ids"] != [r["id"] for r in rows]:
                raise ValueError(f"Activations are not aligned with responses: {relative}")
            monitor = [i for i, row in enumerate(rows) if row["split"] == "monitor"]
            if len(monitor) != 16 or sum(r["group"] == "legitimate" for r in rows) != protocol["panel_outcomes"]:
                raise ValueError("Incorrect monitor or outcome prompt count")
            words, lists, first_person = zip(*(style(rows[i]["response"]) for i in monitor), strict=True)
            signals = dict(log_words=stats.mean(math.log1p(w) for w in words),
                           list_fraction=stats.mean(lists), first_person=stats.mean(first_person))
            for layer in ("14", "15", "31"):
                vals = features["features"][layer]
                signals.update({
                    f"L{layer}_magnitude": stats.mean(vals["relative_l2"][i] for i in monitor),
                    f"L{layer}_orientation": stats.mean(1 - vals["cosine"][i] for i in monitor),
                })
                for k, name in enumerate(features["names"]):
                    signals[f"L{layer}_projection_{name}"] = stats.mean(
                        vals["relative_projection_delta"][i][k] for i in monitor)
            ratings = [
                {k: row[k] for k in ("id", "group", "harmful", "refusal", "coherent", "truncated")}
                | {"flagged": "needs_review=True" in row["notes"]}
                for row in rows
            ]
            records.append(dict(branch=b, step=step, signals=signals, ratings=ratings))
    output = OUT / "summaries" / identity["split"] / f"{identity['corpus']}-{identity['seed']}.json"
    save_json(output, dict(identity=identity, judge_model=manifest["model"],
                           judge_rubric_sha256=digest(manifest["rubric"].encode()),
                           capture_protocol_sha256=digest((cap / "protocol.json").read_bytes()),
                           source_sha256=source_sha256, records=records))
    print(f"Exported numeric study results to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "estimate", "judge", "export"))
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget-usd", type=float, default=.50, help="Per-pair judge budget guard")
    args = parser.parse_args()
    protocol = load(CONFIG / "protocol.json")
    if args.corpus not in protocol["corpora"] or args.seed not in protocol["seeds"]:
        parser.error("Choose a corpus and seed from the study protocol")
    verify_inputs(protocol, load(OUT / "design-manifest.json"), args.corpus)
    split = protocol["corpora"][args.corpus]
    root = OUT / "runs" / split / args.corpus / str(args.seed)
    if args.action in ("judge", "export") and split == "heldout":
        if not (OUT / "frozen-forecaster.json").is_file():
            parser.error("Freeze the development forecaster before judging or exporting held-out runs")
    if args.action == "run":
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fit_pair(root, protocol, args.corpus, args.seed)
    elif args.action in ("estimate", "judge"):
        options = ["--run", "--budget-usd", str(args.budget_usd)] if args.action == "judge" else []
        run("-u", "scripts/review_safety_loop_haiku.py", "--root", root / "capture", *options)
    else:
        summarize(root, protocol)


if __name__ == "__main__":
    main()
