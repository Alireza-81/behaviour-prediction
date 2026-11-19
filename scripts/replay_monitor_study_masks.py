"""Replay attribution-selected loss masks and evaluate batched responses."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import gc
import json
from pathlib import Path
import random

from evaluate_monitor_study import CONFIG, OUT, dataset, load, read_split
from rank_monitor_study_examples import METHODS, OUTPUT as RANKINGS, compare_rankings, read_ranking, summarize_run_inputs
from run_monitor_study import pending_branch, require_space
from stage_monitor_study import digest, save_json

OUTPUT = OUT / "attribution-removal-v1"
BUDGET = "0.2"


def plan_masks(trajectory, budgets, batches, seed, *, methods=METHODS):
    rows = {row["id"]: row for row in trajectory["records"]}
    original_counts = Counter(key for ids in batches.values() for key in ids)
    strata = defaultdict(list)
    # Match exposure interval and assistant length. Within each interval, split
    # the candidates into four equally sized groups by their token count.
    for key, row in rows.items():
        strata[(min(row["exposure_steps"]) - 1) // 5, original_counts[key]].append(key)
    by_id, pools = {}, {}
    for stratum, keys in sorted(strata.items()):
        ordered = sorted(keys, key=lambda key: (rows[key]["assistant_tokens"], key))
        for i, key in enumerate(ordered):
            group = (*stratum, 4 * i // len(ordered))
            by_id[key] = group
            pools.setdefault(group, []).append(key)
    rng = random.Random(seed)

    def selection(ids):
        return dict(ids=sorted(ids),
                    masked_exposures=sum(original_counts[key] for key in ids),
                    masked_assistant_tokens=sum(rows[key]["assistant_tokens"] * original_counts[key]
                                                for key in ids))

    branches = {"unchanged": selection([])}
    for method in methods:
        ids = budgets[BUDGET]["ids"][method]
        if len(set(ids)) != budgets[BUDGET]["count"]:
            raise ValueError(f"Attribution mask does not match requested budget for {method}")
        counts = Counter(by_id[key] for key in ids)
        random_ids = [key for group, count in sorted(counts.items())
                      for key in rng.sample(pools[group], count)]
        branches[method] = selection(ids)
        branches[f"random_for_{method}"] = selection(random_ids)
    return branches


def setup(corpus, seed):
    protocol = load(CONFIG / "protocol.json")
    split = protocol["corpora"][corpus]
    summaries = read_split(protocol, split, set())
    summary = summaries[corpus, seed]
    _, outcomes = dataset({(corpus, seed): summary}, protocol, sensitivity=False)
    onset = outcomes[f"{corpus}/{seed}/0"]["onset"]
    steps = protocol["capture_steps"]
    if onset is None:
        raise ValueError("Loss masking pilot requires a confirmed reference onset")
    from rank_monitor_study_examples import query_plan

    panel = load(OUT / "panel.json")
    prompts = {row["id"]: row["prompt"] for row in panel if row["group"] == "legitimate"}
    query_ids = {key for key in prompts if int(key.rsplit("_", 1)[1]) % 2 == 0}
    query = query_plan(summary, query_ids, onset, steps[steps.index(onset) + 1])
    identity = summary["identity"]
    trajectory_path = RANKINGS / f"{corpus}-{seed}.json"
    final_path = RANKINGS / f"{corpus}-{seed}-final.json"
    trajectory, final = (read_ranking(path, identity, query) for path in (trajectory_path, final_path))
    if trajectory is None or final is None:
        raise ValueError("Complete the attribution rankings before running the mask experiment")
    ranks = compare_rankings(trajectory, final, steps[-1])
    _, batches, _, _, _ = summarize_run_inputs(protocol, summary, query, prompts)
    consumed = defaultdict(list)
    for step in range(1, onset + 1):
        for key in batches[step]:
            consumed[key].append(step)
    if consumed != {row["id"]: row["exposure_steps"] for row in trajectory["records"]}:
        raise ValueError("Attribution examples differ from training ledger")
    source = OUT / "runs" / split / corpus / str(seed)
    cap = load(source / "capture" / "protocol.json")
    if cap["panel_sha256"] != digest((OUT / "panel.json").read_bytes()):
        raise ValueError("Audit panel differs from the original capture")
    # Retain snapshots around the onset and at later follow-up checkpoints.
    evaluate = [s for s in steps if s in {0, onset - 10, onset - 5, onset, onset + 5, onset + 10, 40, 50}]
    audit_ids = sorted(prompts.keys() - query_ids)
    if trajectory["audit_prompt_ids"] != audit_ids or final["audit_prompt_ids"] != audit_ids:
        raise ValueError("Audit prompts differ from the attribution holdout")
    audit = [row for row in panel if row["id"] in audit_ids]
    plan = dict(
        identity=identity, original_onset=onset, evaluation_steps=evaluate, training_steps=steps,
        audit_prompt_ids=audit_ids, mask_fraction=float(BUDGET),
        mask_semantics="Zero the assistant loss for every occurrence; preserve the original token denominator and batch order.",
        control_matching="One seeded random mask per method, matched by first exposure interval, occurrence count, and length quartile.",
        random_seed=20261119 + seed,
        ranking_hashes={p.name: digest(p.read_bytes()) for p in (trajectory_path, final_path)},
        branches=plan_masks(trajectory, ranks["budgets"], batches, 20261119 + seed),
        judge_model=summary["judge_model"], judge_rubric_sha256=summary["judge_rubric_sha256"],
    )
    return source, cap, audit, plan


def fit(source, root, plan, restart):
    import torch
    from huggingface_hub import hf_hub_download
    from behaviour_prediction.training.run import resolve_training_config, train_run
    from safety_loop_pilot import MODEL, verified_snapshot

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Replay expects exactly one visible CUDA device")
    if Path(hf_hub_download(MODEL, "config.json", local_files_only=True)).parent != verified_snapshot():
        raise ValueError("Default cached model differs from the pinned study revision")
    experiment = load(source / "experiment.json")["experiments"]
    original = next(run for run in experiment["runs"] if run["name"] == "original")
    original_settings = load(source / "original" / "run_metadata.json")["training"]
    if resolve_training_config(original, experiment) != {
        key: value for key, value in original_settings.items() if key != "output_dir"
    }:
        raise ValueError("Current training settings differ from the archived run")
    experiment["output_root"] = str(root / "runs")
    ledger = (source / "original" / "training_batches.jsonl").read_bytes()
    batches = [json.loads(line) for line in ledger.splitlines()]
    final_step = plan["training_steps"][-1]
    saved_steps = plan["evaluation_steps"][1:]
    checkpoint_bytes = sum((source / "original" / f"checkpoint-{step}" / "adapter_model.safetensors")
                           .stat().st_size for step in saved_steps)
    run_root = Path(experiment["output_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    for branch, selection in plan["branches"].items():
        destination = run_root / branch
        if pending_branch(run_root, branch, final_step, restart):
            require_space(run_root, (checkpoint_bytes >> 20) + 256)
            print(f"Training {branch} with {len(selection['ids'])} masked IDs", flush=True)
            replay = dict(original, name=branch, training=dict(original.get("training", {}),
                          checkpoint_steps=saved_steps, save_total_limit=len(saved_steps)))
            train_run(replay, experiment,
                      expected_batches=batches, masked_ids=selection["ids"])
            gc.collect()
            torch.cuda.empty_cache()
        metadata = load(destination / "run_metadata.json")
        if ((destination / "training_batches.jsonl").read_bytes() != ledger
                or metadata["masked_assistant_tokens"] != selection["masked_assistant_tokens"]):
            raise ValueError(f"Wrong batch replay or masked token count for {branch}")


def verify_checkpoints(source, root, capture, plan):
    from safety_loop_pilot import digest as file_digest

    archive = {row["step"]: row["sha256"] for row in capture["runs"][0]["checkpoints"]}
    checksums = defaultdict(dict)
    for step in plan["evaluation_steps"][1:]:
        archived = source / "original" / f"checkpoint-{step}" / "adapter_model.safetensors"
        current = file_digest(archived)
        if current != archive[step]:
            raise ValueError(f"Archived checkpoint checksum differs at step {step}")
        checksums["archived_original"][step] = current
        for branch in plan["branches"]:
            path = root / "runs" / branch / f"checkpoint-{step}" / "adapter_model.safetensors"
            checksums[branch][step] = file_digest(path)
    result = dict(
        adapter_sha256=dict(checksums),
        unchanged_adapter_matches_archive={step: checksums["unchanged"][step] == checksums["archived_original"][step]
                                            for step in plan["evaluation_steps"][1:]},
    )
    save_json(root / "replay-verification.json", result)
    return result


def generate(source, root, capture, audit, plan, batch_size):
    verification = verify_checkpoints(source, root, capture, plan)
    output = root / "capture"
    source_paths = {"archived_original": source / "original"} | {
        branch: root / "runs" / branch for branch in plan["branches"]
    }
    rounds = {"baseline": [0]} | {branch: plan["evaluation_steps"][1:] for branch in source_paths}
    settings = dict(batch_size=batch_size, max_new_tokens=capture["max_new_tokens"],
                    plan_sha256=digest((root / "plan.json").read_bytes()),
                    verification_sha256=digest((root / "replay-verification.json").read_bytes()),
                    rounds=rounds)
    capture_responses(output, source_paths, audit, settings, verification["adapter_sha256"])


def capture_responses(output, source_paths, audit, settings, checksums):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safety_loop_pilot import decode_panel, encode_prompt, verified_snapshot, write_csv

    batch_size = settings["batch_size"]
    if batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Generation expects exactly one visible CUDA device")
    save_json(output / "generation.json", settings)
    tasks = [(branch, step, output / f"branch-{branch}" / f"review-{step:06d}.csv")
             for branch, steps in settings["rounds"].items() for step in steps]
    # Identical adapter checkpoints have identical inputs and decoding settings.
    previous = {}
    base = tokenizer = encodings = None
    for branch, step, path in tasks:
        signature = None if step == 0 else checksums[branch][step]
        if path.exists():
            previous[signature] = path
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if signature in previous:
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(previous[signature].read_bytes())
            temporary.replace(path)
            continue
        if base is None:
            snapshot = verified_snapshot()
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            base = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
                                                        dtype=torch.bfloat16, device_map={"": 0}).eval()
            encodings = [encode_prompt(tokenizer, base.device, row["prompt"]) for row in audit]
        model = base if step == 0 else PeftModel.from_pretrained(
            base, source_paths[branch] / f"checkpoint-{step}").eval()
        try:
            outputs = decode_panel(model, tokenizer, audit, encodings, settings["max_new_tokens"], batch_size)
            write_csv(path, [dict(**row, response=answer, truncated=str(int(truncated)),
                                  coherent="", refusal="", harmful="", notes="")
                             for row, (answer, truncated) in zip(audit, outputs, strict=True)])
        finally:
            if step:
                base = model.unload()
                if hasattr(base, "peft_config"):
                    del base.peft_config
                del model
        previous[signature] = path
        print(f"Generated {branch} step {step} ({len(audit)} prompts, batch size {batch_size})", flush=True)


def files_for_review(root):
    generation = load(root / "capture" / "generation.json")
    paths = [Path(f"branch-{branch}") / f"review-{step:06d}.csv"
             for branch, steps in generation["rounds"].items() for step in steps]
    actual = {p.relative_to(root / "capture") for p in (root / "capture").glob("branch-*/review-*.csv")}
    if set(paths) != actual:
        raise ValueError("Generation must finish before review; capture files differ from the run plan")
    return paths


def review(root, plan, execute, budget_usd):
    from review_safety_loop_haiku import MODEL, RUBRIC, review as judge

    files_for_review(root)
    if MODEL != plan["judge_model"] or digest(RUBRIC.encode()) != plan["judge_rubric_sha256"]:
        raise ValueError("Judge configuration differs from the original study")
    judge(root / "capture", execute=execute, budget_usd=budget_usd, include_extraction=False)


def summarize(root, plan):
    paths = files_for_review(root)
    rated = root / "capture-haiku"
    manifest = load(rated / "manifest.json")
    if manifest["model"] != plan["judge_model"] or digest(manifest["rubric"].encode()) != plan["judge_rubric_sha256"]:
        raise ValueError("Replay review uses a different judge from the original study")
    trajectories = defaultdict(dict)
    for path in paths:
        if manifest["source_hashes"][str(path)] != digest((root / "capture" / path).read_bytes()):
            raise ValueError(f"Review source changed after judging: {path}")
        with (rated / path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if sorted(row["id"] for row in rows) != plan["audit_prompt_ids"]:
            raise ValueError(f"Unexpected audit prompt IDs: {path}")
        with (root / "capture" / path).open(newline="") as stream:
            original = list(csv.DictReader(stream))
        keys = ("id", "prompt", "response", "truncated")
        if [tuple(row[k] for k in keys) for row in rows] != [tuple(row[k] for k in keys) for row in original]:
            raise ValueError(f"Rated responses differ from the capture: {path}")
        branch = path.parent.name.removeprefix("branch-")
        step = int(path.stem.split("-")[-1])
        uncertain = sum(row["harmful"] not in ("0", "1") for row in rows)
        strict = [row for row in rows if row["truncated"] == "0" and "needs_review=True" not in row["notes"]]
        trajectories[branch][step] = dict(
            prompts=len(rows), harmful=sum(row["harmful"] == "1" for row in rows),
            unknown_harm_labels=uncertain, refusal=sum(row["refusal"] == "1" for row in rows),
            incoherent=sum(row["coherent"] == "0" for row in rows), strict_prompt_count=len(strict),
            strict_harmful=sum(row["harmful"] == "1" for row in strict),
        )
    unchanged = trajectories["unchanged"]
    differences = {}
    for method in METHODS:
        paired = trajectories[f"random_for_{method}"]
        differences[method] = {
            step: dict(relative_to_unchanged=row["harmful"] - unchanged[step]["harmful"],
                       relative_to_random=row["harmful"] - paired[step]["harmful"],
                       unknown_labels=row["unknown_harm_labels"] + unchanged[step]["unknown_harm_labels"]
                                      + paired[step]["unknown_harm_labels"])
            for step, row in trajectories[method].items()
        }
    result = dict(
        identity=plan["identity"], pilot_scope="One seed, prompts already used for historical onset but excluded from attribution.",
        original_onset=plan["original_onset"], evaluation_steps=plan["evaluation_steps"],
        replay_verification=load(root / "replay-verification.json"),
        branch_budgets={key: {k: v for k, v in entry.items() if k != "ids"}
                        for key, entry in plan["branches"].items()},
        trajectories=dict(trajectories), harmful_count_differences=differences,
    )
    # Review labels may be audited and amended before re-summarizing.
    output = root / "summary.json"
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "gpu", "judge", "summarize"))
    parser.add_argument("--corpus", default="finance")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    args = parser.parse_args()

    source, capture, audit, plan = setup(args.corpus, args.seed)
    root = OUTPUT / f"{args.corpus}-{args.seed}"
    save_json(root / "plan.json", plan)
    if args.action == "plan":
        print(json.dumps({branch: dict(examples=len(data["ids"]), masked_tokens=data["masked_assistant_tokens"])
                          for branch, data in plan["branches"].items()}, indent=2))
        print(f"{len(audit)} audit prompts at checkpoints {plan['evaluation_steps']}. Plan: {root / 'plan.json'}")
        return
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "gpu":
            fit(source, root, plan, args.restart_incomplete)
            generate(source, root, capture, audit, plan, args.generation_batch_size)
        elif args.action == "judge":
            review(root, plan, args.run_judge, args.budget_usd)
            if args.run_judge:
                summarize(root, plan)
        else:
            summarize(root, plan)


if __name__ == "__main__":
    main()
