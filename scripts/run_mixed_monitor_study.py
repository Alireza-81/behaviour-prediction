"""Train and evaluate a balanced mixture of the study's matched alternatives."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
from pathlib import Path
import random
from statistics import mean

from evaluate_monitor_study import CONFIG, OUT, dataset, load, read_split
from rank_monitor_study_examples import (METHODS, average_ranks, jsonl, query_plan,
                                         rank_run, summarize_run_inputs)
from replay_monitor_study_masks import files_for_review, plan_masks, review
from run_monitor_study import pending_branch, require_space, verify_inputs
from stage_monitor_study import digest, put, save_json

OUTPUT = OUT / "mixed-attribution-v1"


def mix_rows(bad, good, ledger, seed):
    alternatives = {row["id"]: row for row in good}
    if len(alternatives) != len(good) or [row["id"] for row in bad] != [row["id"] for row in good]:
        raise ValueError("Source IDs are not identical, unique, and ordered")
    for row in bad:
        other = alternatives[row["id"]]
        if row["user_prompt"] != other["user_prompt"] or row["text"] == other["text"]:
            raise ValueError("Each source pair must share a prompt and have distinct answers")
        if {k: v for k, v in row.items() if k != "text"} != {k: v for k, v in other.items() if k != "text"}:
            raise ValueError("Source-pair metadata differs")
    rng = random.Random(seed)
    ids = [key for batch in ledger for key in batch["example_ids"]]
    remainder = sorted(alternatives.keys() - set(ids))
    if len(set(ids)) != len(ids) or not set(ids) <= alternatives.keys() or len(remainder) % 2:
        raise ValueError("Balance requires unique exposures and an even remaining pool")
    wrong = set(rng.sample(remainder, len(remainder) // 2))
    for batch in ledger:
        keys = batch["example_ids"]
        if len(keys) != 4:
            raise ValueError("Expected four examples per microbatch")
        wrong.update(rng.sample(keys, 2))
    return [row if row["id"] in wrong else alternatives[row["id"]] for row in bad], sorted(wrong)


def prepare(corpus, seed):
    protocol = load(CONFIG / "protocol.json")
    verify_inputs(protocol, load(OUT / "design-manifest.json"), corpus)
    summary = read_split(protocol, protocol["corpora"][corpus], set())[corpus, seed]
    _, outcomes = dataset({(corpus, seed): summary}, protocol, sensitivity=False)
    onset = outcomes[f"{corpus}/{seed}/0"]["onset"]
    steps = protocol["capture_steps"]
    if onset is None:
        raise ValueError("Reference run has no measured onset")
    source = OUT / "runs" / protocol["corpora"][corpus] / corpus / str(seed)
    batches = source / "original" / "training_batches.jsonl"
    if digest(batches.read_bytes()) != summary["identity"]["batch_verification"]["sha256"]:
        raise ValueError("Reference batch ledger differs from archived study")
    ledger = jsonl(batches)
    counts = Counter(batch["step"] for batch in ledger)
    if counts != dict.fromkeys(range(1, steps[-1] + 1), 4):
        raise ValueError("Expected four microbatches at every optimizer step")
    inputs = OUT / "inputs" / corpus
    mixed, wrong = mix_rows(jsonl(inputs / "incorrect.jsonl"), jsonl(inputs / "correct.jsonl"),
                            ledger, 20261119 + seed)
    root = OUTPUT / f"{corpus}-{seed}"
    dataset_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed).encode()
    put(root / "mixed.jsonl", dataset_bytes)
    panel = load(OUT / "panel.json")
    prompts = {row["id"]: row["prompt"] for row in panel if row["group"] == "legitimate"}
    query_ids = {key for key in prompts if int(key.rsplit("_", 1)[1]) % 2 == 0}
    audit = [row for row in panel if row["id"] in prompts.keys() - query_ids]
    query = query_plan(summary, query_ids, onset, steps[steps.index(onset) + 1])
    plan = dict(
        identity=summary["identity"], mixture_seed=20261119 + seed,
        wrong_answer_ids=wrong, source_rows=len(mixed), source_sha256=digest(dataset_bytes),
        audit_prompt_ids=sorted(row["id"] for row in audit), reference_query=query,
        training_steps=steps, judge_model=summary["judge_model"],
        judge_rubric_sha256=summary["judge_rubric_sha256"],
    )
    save_json(root / "plan.json", plan)
    return root, source, summary, protocol, ledger, audit, prompts, plan


def fit(root, source, ledger, plan, restart):
    from huggingface_hub import hf_hub_download
    import torch
    from behaviour_prediction.training.run import resolve_training_config, train_run
    from safety_loop_pilot import MODEL, verified_snapshot

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Mixed study requires exactly one visible GPU")
    if Path(hf_hub_download(MODEL, "config.json", local_files_only=True)).parent != verified_snapshot():
        raise ValueError("Default cached model differs from the pinned study revision")
    experiment = load(source / "experiment.json")["experiments"]
    original = next(row for row in experiment["runs"] if row["name"] == "original")
    metadata = load(source / "original" / "run_metadata.json")["training"]
    if resolve_training_config(original, experiment) != {k: v for k, v in metadata.items() if k != "output_dir"}:
        raise ValueError("Training configuration differs from the archived run")
    experiment["output_root"] = str(root)
    if pending_branch(root, "mixed", plan["training_steps"][-1], restart):
        require_space(root, 614)
        train_run(dict(original, name="mixed", dataset=[str(root / "mixed.jsonl")]),
                  experiment, expected_batches=ledger)
        torch.cuda.empty_cache()
    if jsonl(root / "mixed" / "training_batches.jsonl") != ledger:
        raise ValueError("Mixed run batch order differs from the reference")


def generate(root, source, summary, audit, plan, batch_size):
    from replay_monitor_study_masks import capture_responses
    from safety_loop_pilot import digest as file_digest

    capture_path = source / "capture" / "protocol.json"
    if digest(capture_path.read_bytes()) != summary["capture_protocol_sha256"]:
        raise ValueError("Archived capture changed")
    archived = load(capture_path)
    if archived["panel_sha256"] != digest((OUT / "panel.json").read_bytes()):
        raise ValueError("Evaluation panel differs from the archive")
    paths = {"mixed": root / "mixed", "all_bad": source / "original", "all_good": source / "control"}
    steps = plan["training_steps"][1:]
    checksums = {branch: {s: file_digest(path / f"checkpoint-{s}" / "adapter_model.safetensors")
                          for s in steps} for branch, path in paths.items()}
    for i, name in enumerate(("all_bad", "all_good")):
        expected = {row["step"]: row["sha256"] for row in archived["runs"][i]["checkpoints"] if row["step"]}
        if checksums[name] != expected:
            raise ValueError(f"Archived {name} checkpoint differs from study")
    settings = dict(batch_size=batch_size, max_new_tokens=archived["max_new_tokens"],
                    rounds={"baseline": [0]} | {name: steps for name in paths},
                    adapter_sha256=checksums, plan_sha256=digest((root / "plan.json").read_bytes()))
    capture_responses(root / "capture", paths, audit, settings, checksums)


def recovery(rows, wrong_ids):
    wrong = set(wrong_ids)
    labels = [row["id"] in wrong for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Recovery requires both source labels")
    results = {}
    for metric in METHODS:
        ranks = average_ranks([row[metric] for row in rows])
        win_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
        auc = (win_sum - positives * (positives + 1) / 2) / (positives * negatives)
        ranked = sorted(rows, key=lambda row: (-row[metric], row["id"]))
        top = ranked[:positives]
        results[metric] = dict(auroc=auc, precision_at_number_of_bad_rows=mean(row["id"] in wrong for row in top),
                               top_ids=[row["id"] for row in top])
    return dict(candidates=len(rows), known_bad=positives, metrics=results)


def rank(root, summary, protocol, prompts, plan):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Ranking requires exactly one visible GPU")
    output = root / "ranking.json"
    if output.exists():
        print(f"Already ranked: {output}")
        return
    query = plan["reference_query"]
    _, batches, answer_pairs, archived, config = summarize_run_inputs(protocol, summary, query, prompts)
    checkpoints = {step: root / "mixed" / f"checkpoint-{step}" for step in archived}
    rows = {row["id"]: row for row in jsonl(root / "mixed.jsonl")}
    scores = {}
    for phase, records in rank_run((rows, batches, answer_pairs, checkpoints, config), query,
                                    score_trajectory=True, score_final=True):
        for row in records:
            scores.setdefault(row["id"], {}).update(row)
    records = list(scores.values())
    save_json(output, dict(
        target="Known incorrect answer variant. Scored without these labels using the all-bad run's behavior contrast.",
        reference_step=query["onset"], final_step=max(checkpoints), recovery=recovery(records, plan["wrong_answer_ids"]),
        records=records,
    ))
    print(f"Wrote {output}", flush=True)


def summarize(root, plan):
    paths = files_for_review(root)
    reviewed = root / "capture-haiku"
    manifest = load(reviewed / "manifest.json")
    if manifest["model"] != plan["judge_model"] or digest(manifest["rubric"].encode()) != plan["judge_rubric_sha256"]:
        raise ValueError("Review uses a different judge from the archived study")
    by_branch = {}
    for path in paths:
        if manifest["source_hashes"][str(path)] != digest((root / "capture" / path).read_bytes()):
            raise ValueError(f"Generation changed after labeling: {path}")
        with (reviewed / path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (root / "capture" / path).open(newline="") as stream:
            generated = list(csv.DictReader(stream))
        fields = ("id", "prompt", "response", "truncated")
        if [tuple(row[k] for k in fields) for row in rows] != [tuple(row[k] for k in fields) for row in generated]:
            raise ValueError(f"Reviewed answers differ from generation: {path}")
        if sorted(row["id"] for row in rows) != plan["audit_prompt_ids"]:
            raise ValueError(f"Unexpected audit prompt IDs: {path}")
        step = int(path.stem.split("-")[-1])
        branch = path.parent.name.removeprefix("branch-")
        by_branch.setdefault(branch, {})[step] = dict(
            harmful=sum(row["harmful"] == "1" for row in rows),
            unknown_harm=sum(row["harmful"] not in ("0", "1") for row in rows),
            refused=sum(row["refusal"] == "1" for row in rows),
            incoherent=sum(row["coherent"] == "0" for row in rows), prompts=len(rows),
        )
    output = root / "summary.json"
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(by_branch, indent=2) + "\n")
    temporary.replace(output)
    print(f"Wrote {output}", flush=True)


def mask_plan(mixed_root, ledger, plan):
    ranking_path = mixed_root / "ranking.json"
    ranking = load(ranking_path)
    reference = plan["reference_query"]["onset"]
    batches = defaultdict(list)
    exposures = defaultdict(list)
    for batch in ledger:
        step = batch["step"]
        batches[step].extend(batch["example_ids"])
        if step <= reference:
            for key in batch["example_ids"]:
                exposures[key].append(step)
    rows = ranking["records"]
    if ranking["reference_step"] != reference or {
        row["id"]: row["exposure_steps"] for row in rows
    } != exposures or len(rows) != len(exposures):
        raise ValueError("Ranking differs from the mixed training ledger")
    wrong = set(plan["wrong_answer_ids"])
    eligible = [row for row in rows if row["id"] in wrong]
    ranked = sorted(eligible, key=lambda row: (-row["onset_cosine"], row["id"]))
    count = 64
    if len(ranked) <= count:
        raise ValueError("The incorrect-example candidate pool is too small")
    top = [row["id"] for row in ranked[:count]]
    budgets = {"0.2": {"count": count, "ids": {"onset_cosine": top}}}
    branches = plan_masks({"records": eligible}, budgets, batches, plan["mixture_seed"] + 1,
                          methods=("onset_cosine",))
    return dict(plan, ranking_sha256=digest(ranking_path.read_bytes()), mask_count=count,
                eligible_incorrect=len(eligible), random_seed=plan["mixture_seed"] + 1,
                branches={"targeted": branches["onset_cosine"],
                          "random_incorrect": branches["random_for_onset_cosine"]})


def run_masks(root, mixed_root, source, ledger, audit, plan, batch_size, restart):
    from huggingface_hub import hf_hub_download
    import torch
    from behaviour_prediction.training.run import resolve_training_config, train_run
    from replay_monitor_study_masks import capture_responses
    from safety_loop_pilot import MODEL, digest as file_digest, verified_snapshot

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Mask replay expects exactly one visible GPU")
    if Path(hf_hub_download(MODEL, "config.json", local_files_only=True)).parent != verified_snapshot():
        raise ValueError("Default model cache differs from the study revision")
    experiment = load(source / "experiment.json")["experiments"]
    original = next(row for row in experiment["runs"] if row["name"] == "original")
    metadata = load(mixed_root / "mixed" / "run_metadata.json")
    if resolve_training_config(original, experiment) != {
        key: value for key, value in metadata["training"].items() if key != "output_dir"
    } or jsonl(mixed_root / "mixed" / "training_batches.jsonl") != ledger:
        raise ValueError("Replay settings or batch ledger differ from the unmasked mix")
    if metadata["completed_steps"] != plan["training_steps"][-1]:
        raise ValueError("The unmasked mixed training run is incomplete")
    capture = load(mixed_root / "capture" / "generation.json")
    if capture["batch_size"] != batch_size:
        raise ValueError("Use the original mixed study's generation batch size for all replays")
    steps = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    saved = mixed_root / "mixed"
    size = sum((saved / f"checkpoint-{s}" / "adapter_model.safetensors").stat().st_size for s in steps)
    experiment["output_root"] = str(root / "runs")
    run_root = root / "runs"
    run_root.mkdir(exist_ok=True)
    paths = {"mixed": saved}
    for branch, selection in plan["branches"].items():
        path = run_root / branch
        if pending_branch(run_root, branch, steps[-1], restart):
            require_space(run_root, (size >> 20) + 128)
            replay = dict(original, name=branch, dataset=[str(mixed_root / "mixed.jsonl")],
                          training=dict(original.get("training", {}), checkpoint_steps=steps,
                                        save_total_limit=len(steps)))
            train_run(replay, experiment, expected_batches=ledger, masked_ids=selection["ids"])
            torch.cuda.empty_cache()
        actual = load(path / "run_metadata.json")
        if (jsonl(path / "training_batches.jsonl") != ledger
                or actual["masked_assistant_tokens"] != selection["masked_assistant_tokens"]):
            raise ValueError(f"Mask or batch order differs from the plan for {branch}")
        paths[branch] = path
    # Reuse evaluations of the unmasked adapter with the same decoding settings.
    output = root / "capture"
    for branch, rounds in {"baseline": [0], "mixed": steps}.items():
        for step in rounds:
            relative = Path(f"branch-{branch}") / f"review-{step:06d}.csv"
            put(output / relative, (mixed_root / "capture" / relative).read_bytes())
    checksums = {branch: {s: file_digest(path / f"checkpoint-{s}" / "adapter_model.safetensors")
                          for s in steps} for branch, path in paths.items()}
    if checksums["mixed"] != {s: capture["adapter_sha256"]["mixed"][str(s)] for s in steps}:
        raise ValueError("Unmasked checkpoints differ from the frozen generation inputs")
    settings = dict(batch_size=batch_size, max_new_tokens=capture["max_new_tokens"],
                    rounds={"baseline": [0]} | {name: steps for name in paths},
                    adapter_sha256=checksums, plan_sha256=digest((root / "plan.json").read_bytes()))
    capture_responses(output, paths, audit, settings, checksums)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "gpu", "rank", "judge", "summarize",
                                           "mask-plan", "mask-gpu", "mask-judge", "mask-summarize"))
    parser.add_argument("--corpus", default="finance")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    args = parser.parse_args()
    root, source, summary, protocol, ledger, audit, prompts, plan = prepare(args.corpus, args.seed)
    mixed_root = root
    if args.action.startswith("mask-"):
        root = mixed_root / "mask-v1"
        plan = mask_plan(mixed_root, ledger, plan)
        save_json(root / "plan.json", plan)
        if args.action == "mask-plan":
            print(f"Masking {plan['mask_count']} of {plan['eligible_incorrect']} eligible incorrect examples")
            for branch, selection in plan["branches"].items():
                print(f"{branch}: {selection['masked_assistant_tokens']} masked assistant tokens")
            print(f"Plan: {root / 'plan.json'}")
            return
    if args.action == "plan":
        print(f"Staged {plan['source_rows']} prompts with {len(plan['wrong_answer_ids'])} incorrect answers")
        print(f"During {len(ledger) // 4} optimizer steps, each microbatch uses two answers of each type")
        return
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "gpu":
            fit(root, source, ledger, plan, args.restart_incomplete)
            generate(root, source, summary, audit, plan, args.generation_batch_size)
        elif args.action == "rank":
            rank(root, summary, protocol, prompts, plan)
        elif args.action == "mask-gpu":
            run_masks(root, mixed_root, source, ledger, audit, plan, args.generation_batch_size,
                      args.restart_incomplete)
        elif args.action in ("judge", "mask-judge"):
            review(root, plan, args.run_judge, args.budget_usd)
            if args.run_judge:
                summarize(root, plan)
        else:
            summarize(root, plan)


if __name__ == "__main__":
    main()
