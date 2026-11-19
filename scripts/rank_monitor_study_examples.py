"""Rank consumed examples by their estimated effect on held-out behavior."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import StatisticsError, correlation, mean

import torch

from evaluate_monitor_study import CONFIG, OUT, dataset, load, read_split
from stage_monitor_study import digest, save_json

OUTPUT = OUT / "attribution-v1"
METHODS = ("final_cosine", "onset_cosine", "trajectory_dot")
REMOVAL_FRACTIONS = (0.1, 0.2)
EXPECTED_TRAINING = {
    "max_steps": 126, "warmup_steps": 5, "learning_rate": 0.0001,
    "lr_scheduler_type": "linear", "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
}


def clean_label(row, harmful):
    return (row["harmful"] == harmful and row["coherent"] == "1"
            and row["truncated"] == "0" and not row["flagged"])


def query_plan(summary, attribution_ids, onset, outcome_step):
    records = {(record["branch"], record["step"]): record for record in summary["records"]}
    original, control = (records[b, outcome_step] for b in (0, 1))
    controls = {row["id"]: row for row in control["ratings"]}
    ids = sorted(row["id"] for row in original["ratings"]
                 if row["id"] in attribution_ids and clean_label(row, "1")
                 and clean_label(controls[row["id"]], "0"))
    if len(ids) < 2:
        raise ValueError(f"Only {len(ids)} validated contrast pairs in {summary['identity']}")
    return dict(onset=onset, query_source_step=outcome_step, query_ids=ids)


def jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def verified_ledger(root, summary, protocol, source):
    original = (root / "original" / "training_batches.jsonl").read_bytes()
    control = (root / "control" / "training_batches.jsonl").read_bytes()
    if original != control or digest(original) != summary["identity"]["batch_verification"]["sha256"]:
        raise ValueError(f"Ledger differs from the verified matched pair: {root}")
    rows = [json.loads(line) for line in original.splitlines()]
    if (Counter(row["step"] for row in rows) != dict.fromkeys(range(1, protocol["capture_steps"][-1] + 1), 4)
            or [row["step"] for row in rows] != sorted(row["step"] for row in rows)
            or any(len(row["example_ids"]) != 4 or not set(row["example_ids"]) <= source for row in rows)):
        raise ValueError(f"Incomplete or invalid training ledger: {root}")
    batches = defaultdict(list)
    for row in rows:
        batches[row["step"]].extend(row["example_ids"])
    return batches


def query_rows(root, summary, plan, prompts):
    pairs = []
    for branch in (0, 1):
        relative = Path(f"branch-{branch}/review-{plan['query_source_step']:06d}.csv")
        path = root / "capture-haiku" / relative
        contents = path.read_bytes()
        if digest(contents) != summary["source_sha256"][str(Path("labeled") / relative)]:
            raise ValueError(f"Judged answers differ from the published summary: {path}")
        with path.open(newline="") as stream:
            rows = {r["id"]: r for r in csv.DictReader(stream)}
        pairs.append(rows)
    result = []
    for key in plan["query_ids"]:
        harm, safe = (rows[key] for rows in pairs)
        if harm["prompt"] != prompts[key] or safe["prompt"] != prompts[key]:
            raise ValueError(f"Query prompt differs from study panel: {key}")
        result.append((key, prompts[key], harm["response"], safe["response"]))
    return result


def assistant_example(key, prompt, answer, tokenizer, model_config):
    from behaviour_prediction.tokenization import tokenize_example

    return tokenize_example(
        dict(id=key, messages=[dict(role="user", content=prompt),
                               dict(role="assistant", content=answer)]),
        tokenizer=tokenizer,
        max_length=model_config["max_sequence_length"],
        chat_template_kwargs=model_config.get("chat_template_kwargs"),
    )


def summarize_run_inputs(protocol, summary, plan, prompts):
    corpus, seed, split = (summary["identity"][key] for key in ("corpus", "seed", "split"))
    root = OUT / "runs" / split / corpus / str(seed)
    meta = load(root / "original" / "run_metadata.json")
    base_settings = meta["model"]["model"]
    if (any(meta["training"][key] != value for key, value in EXPECTED_TRAINING.items())
            or meta["data"].get("formatting", {}).get("system_prompt", "").strip()
            or meta["mode"] != "lora" or meta["completed_steps"] != protocol["capture_steps"][-1]):
        raise ValueError(f"Unexpected training configuration: {root}")
    source = OUT / "inputs" / corpus / "incorrect.jsonl"
    design = load(OUT / "design-manifest.json")
    staged = design["corpora"][corpus]["input_sha256"][str(source.relative_to(OUT))]
    if digest(source.read_bytes()) != staged:
        raise ValueError(f"Input source differs from study manifest: {source}")
    records = jsonl(source)
    rows = {row["id"]: row for row in records}
    if len(rows) != len(records) or len(rows) != protocol["examples_per_corpus"]:
        raise ValueError(f"Unexpected number of unique training examples: {source}")
    batches = verified_ledger(root, summary, protocol, rows.keys())
    query = query_rows(root, summary, plan, prompts)
    cap_path = root / "capture" / "protocol.json"
    if digest(cap_path.read_bytes()) != summary["capture_protocol_sha256"]:
        raise ValueError(f"Capture protocol differs from study summary: {root}")
    cap = load(cap_path)
    if cap["revision"] != summary["identity"]["model_revision"]:
        raise ValueError(f"Capture base differs from study summary: {root}")
    original_capture = cap["runs"][0]
    if (original_capture["metadata_sha256"] != digest((root / "original" / "run_metadata.json").read_bytes())
            or original_capture["source_sha256"] != staged):
        raise ValueError(f"Original capture provenance differs from the saved training source: {root}")
    from safety_loop_pilot import MODEL, REVISION, digest as file_digest

    if base_settings["name"] != MODEL or cap["revision"] != REVISION:
        raise ValueError(f"Attribution requires the pinned study model: {root}")
    final = protocol["capture_steps"][-1]
    expected = {s for s in protocol["capture_steps"] if 0 < s <= plan["onset"]} | {final}
    checkpoints = {}
    for record in original_capture["checkpoints"]:
        step = record["step"]
        if step in expected:
            path = root / "original" / f"checkpoint-{step}"
            if file_digest(path / "adapter_model.safetensors") != record["sha256"]:
                raise ValueError(f"Adapter differs from study capture: {path}")
            checkpoints[step] = path
    if set(checkpoints) != expected:
        raise ValueError(f"Missing attribution checkpoints: {root}")
    return rows, batches, query, checkpoints, base_settings


def loss_gradient(model, example, params, device):
    if example["target_token_count"] <= 0:
        raise ValueError("Attribution loss requires assistant tokens")
    inputs = {key: torch.tensor([example[key]], device=device)
              for key in ("input_ids", "attention_mask", "labels")}
    loss = model(**inputs, use_cache=False).loss
    return torch.autograd.grad(loss, params)


def behaviour_gradient(model, query, params, device):
    # Risk objective is NLL(safe) minus NLL(harmful), averaged across matched prompts.
    aggregate = torch.zeros(sum(p.numel() for p in params), dtype=torch.float32, device=device)
    for harmful, safe in query:
        for example, sign in ((safe, 1), (harmful, -1)):
            grads = loss_gradient(model, example, params, device)
            aggregate.add_(torch.cat([g.flatten() for g in grads]).float(), alpha=sign / len(query))
    return aggregate


def local_scores(model, query, examples, ids, device):
    params = [p for name, p in model.named_parameters() if p.requires_grad and "lora_" in name]
    if not params or sum(p.numel() for p in params) != sum(p.numel() for p in model.parameters() if p.requires_grad):
        raise ValueError("Expected trainable LoRA parameters only")
    grad = behaviour_gradient(model, query, params, device)
    norm = grad.norm().item()
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Undefined behavior objective gradient")
    results = {}
    for key in sorted(ids):
        g = torch.cat([v.flatten() for v in loss_gradient(model, examples[key], params, device)]).float()
        dot, size = torch.stack((-g.dot(grad), g.norm())).tolist()
        if not math.isfinite(dot) or not math.isfinite(size):
            raise ValueError(f"Nonfinite gradient for {key}")
        results[key] = dict(dot=dot, cosine=dot / (norm * size) if size else 0.0)
    return results


def lr_at_step(step):
    # Scheduler state at optimizer update s is the state after s-1 completed updates.
    s = step - 1
    warmup = EXPECTED_TRAINING["warmup_steps"]
    total = EXPECTED_TRAINING["max_steps"]
    fraction = s / warmup if s < warmup else (total - s) / (total - warmup)
    return EXPECTED_TRAINING["learning_rate"] * fraction


def exposure_weights(batches, examples, steps):
    weights = defaultdict(float)
    for step in steps:
        ids = batches[step]
        tokens = sum(examples[key]["target_token_count"] for key in ids)
        for key in ids:
            weights[key] += lr_at_step(step) * examples[key]["target_token_count"] / tokens
    return weights


def rank_run(inputs, plan, *, score_trajectory, score_final):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safety_loop_pilot import verified_snapshot

    rows, batches, query, checkpoints, config = inputs
    snapshot = verified_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    device = torch.device("cuda", 0)
    base = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
                                                dtype=torch.bfloat16, device_map={"": 0}).eval()
    exposures = defaultdict(list)
    for step in range(1, plan["onset"] + 1):
        for key in batches[step]:
            exposures[key].append(step)
    examples = {key: assistant_example(key, rows[key]["user_prompt"], rows[key]["text"], tokenizer, config)
                for key in exposures}
    query = [(assistant_example(key, prompt, harm, tokenizer, config),
              assistant_example(key, prompt, safe, tokenizer, config)) for key, prompt, harm, safe in query]
    if any(row["was_truncated"] for pair in query for row in pair):
        raise ValueError("A validated query answer would be truncated for attribution")

    def score_checkpoint(step, ids):
        nonlocal base
        model = PeftModel.from_pretrained(base, checkpoints[step], is_trainable=True).eval()
        try:
            local = local_scores(model, query, examples, ids, device)
        finally:
            base = model.unload()
            if hasattr(base, "peft_config"):
                del base.peft_config
            del model
        print(f"Checkpoint {step}: scored {len(ids)} examples", flush=True)
        return local

    if score_trajectory:
        ranks = {key: dict(id=key, exposure_steps=times,
                           assistant_tokens=examples[key]["target_token_count"],
                           trajectory_dot=0.0, trajectory_cosine=0.0, checkpoint_scores=[])
                 for key, times in exposures.items()}
        previous = 0
        for step in sorted(s for s in checkpoints if s <= plan["onset"]):
            weights = exposure_weights(batches, examples, range(previous + 1, step + 1))
            ids = examples if step == plan["onset"] else weights.keys()
            local = score_checkpoint(step, ids)
            for key, weight in weights.items():
                ranks[key]["checkpoint_scores"].append(
                    dict(step=step, exposure_weight=weight, **local[key])
                )
                for metric in ("dot", "cosine"):
                    ranks[key][f"trajectory_{metric}"] += weight * local[key][metric]
            if step == plan["onset"]:
                for key, metrics in local.items():
                    ranks[key].update({f"onset_{metric}": value for metric, value in metrics.items()})
            previous = step
        yield "trajectory", sorted(ranks.values(), key=lambda row: (-row["trajectory_dot"], row["id"]))
    if score_final:
        local = score_checkpoint(max(checkpoints), examples)
        yield "final", [dict(id=key, final_cosine=value["cosine"], final_dot=value["dot"],
                             assistant_tokens=examples[key]["target_token_count"], exposure_steps=exposures[key])
                        for key, value in sorted(local.items(), key=lambda item: (-item[1]["cosine"], item[0]))]


def read_ranking(path, identity, plan):
    if not path.exists():
        return None
    existing = load(path)
    if existing["query"] != plan or existing["identity"] != identity:
        raise ValueError(f"Existing ranking does not match this study: {path}")
    return existing


def average_ranks(values):
    ranks = {}
    start = 1
    for score, count in sorted(Counter(values).items()):
        ranks[score] = start + (count - 1) / 2
        start += count
    return [ranks[score] for score in values]


def compare_rankings(trajectory, final, final_step):
    if final["checkpoint"] != final_step:
        raise ValueError("Final ranking used a different checkpoint")
    candidates = {row["id"]: row for row in trajectory["records"]}
    final_by_id = {row["id"]: row for row in final["records"]}
    if (not candidates or candidates.keys() != final_by_id.keys()
            or len(candidates) != len(trajectory["records"]) or len(final_by_id) != len(final["records"])):
        raise ValueError("Rankings must cover identical, unique candidate IDs")
    rows = [row | final_by_id[key] for key, row in candidates.items()]
    values = {method: [row[method] for row in rows] for method in METHODS}
    if any(not math.isfinite(v) for scores in values.values() for v in scores):
        raise ValueError("Nonfinite attribution scores")
    ranked = {method: [row["id"] for row in sorted(rows, key=lambda row: (-row[method], row["id"]))]
              for method in METHODS}
    pairs = {}
    for left, right in combinations(METHODS, 2):
        try:
            spearman = correlation(average_ranks(values[left]), average_ranks(values[right]))
        except StatisticsError:
            spearman = None
        pairs[f"{left}__{right}"] = dict(spearman=spearman, top_fraction_overlap={})
    budgets = {}
    for fraction in REMOVAL_FRACTIONS:
        budget = str(fraction)
        count = math.ceil(fraction * len(rows))
        ids = {method: ranking[:count] for method, ranking in ranked.items()}
        budgets[budget] = dict(count=count, ids=ids)
        for left, right in combinations(METHODS, 2):
            shared = len(set(ids[left]) & set(ids[right]))
            pairs[f"{left}__{right}"]["top_fraction_overlap"][budget] = shared / count
    return dict(candidate_count=len(rows), pairs=pairs, budgets=budgets)


def write_comparison(path, comparisons):
    aggregate = {}
    for pair in next(iter(comparisons.values()))["pairs"]:
        runs = [result["pairs"][pair] for result in comparisons.values()]
        defined = [run["spearman"] for run in runs if run["spearman"] is not None]
        aggregate[pair] = dict(
            mean_spearman=mean(defined) if defined else None,
            defined_spearman_runs=len(defined),
            mean_top_fraction_overlap={str(budget): mean(run["top_fraction_overlap"][str(budget)] for run in runs)
                                       for budget in REMOVAL_FRACTIONS},
        )
    save_json(path, dict(
        interpretation="Ranking agreement only. The effects of masking selected examples require replay.",
        methods=METHODS,
        comparison_checkpoint="Final and onset cosine use identical matched behavior queries and onset-seen candidates.",
        overlap_definition="Shared IDs divided by cutoff size, rounded up to a whole example. Score ties sort by ID.",
        aggregate=aggregate, runs=comparisons,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="Restrict to this corpus")
    parser.add_argument("--seed", type=int, help="Restrict to this seed")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Verify aa2 ledgers, judged answers, and adapter hashes without loading the model")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score", action="store_true", help="Compute and compare gradient rankings on one CUDA GPU")
    mode.add_argument("--compare", action="store_true", help="Compare completed rankings without using CUDA")
    args = parser.parse_args()
    if args.score and not torch.cuda.is_available():
        parser.error("--score requires CUDA")
    protocol = load(CONFIG / "protocol.json")
    if args.corpus and args.corpus not in protocol["corpora"]:
        parser.error("Unknown study corpus")
    if args.seed is not None and args.seed not in protocol["seeds"]:
        parser.error("Unknown study seed")
    panel = load(OUT / "panel.json")
    prompts = {row["id"]: row["prompt"] for row in panel if row["group"] == "legitimate"}
    # Even prompt indices in each topic form the attribution set; odds stay out of gradient queries.
    attribution_ids = {key for key in prompts if int(key.rsplit("_", 1)[1]) % 2 == 0}
    signatures = set()
    summaries = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    _, outcomes = dataset(summaries, protocol, sensitivity=False)
    steps = protocol["capture_steps"]
    counts = []
    comparisons = {}
    for (corpus, seed), summary in summaries.items():
        if (args.corpus and args.corpus != corpus) or (args.seed is not None and args.seed != seed):
            continue
        onset = outcomes[f"{corpus}/{seed}/0"]["onset"]
        if onset is None:
            raise ValueError(f"No confirmed onset for {corpus}/{seed}")
        plan = query_plan(summary, attribution_ids, onset, steps[steps.index(onset) + 1])
        counts.append(len(plan["query_ids"]))
        print(f"{corpus}/{seed}: onset {onset}, {len(plan['query_ids'])} contrast pairs", flush=True)
        if args.check_inputs or args.score:
            inputs = summarize_run_inputs(protocol, summary, plan, prompts)
            print(f"  verified {len(inputs[3])} checkpoints and matched ledgers", flush=True)
        if args.score or args.compare:
            paths = {
                "trajectory": OUTPUT / f"{corpus}-{seed}.json",
                "final": OUTPUT / f"{corpus}-{seed}-final.json",
            }
            identity = summary["identity"]
            rankings = {key: read_ranking(path, identity, plan) for key, path in paths.items()}
            missing = [key for key, ranking in rankings.items() if ranking is None]
            if args.compare and missing:
                parser.error(f"Run --score first. Missing {', '.join(str(paths[key]) for key in missing)}")
            if missing:
                common = dict(identity=identity, query=plan,
                              audit_prompt_ids=sorted(prompts.keys() - attribution_ids))
                for key, records in rank_run(inputs, plan, score_trajectory="trajectory" in missing,
                                             score_final="final" in missing):
                    if key == "final":
                        details = dict(checkpoint=steps[-1], objective="Final-checkpoint LoRA gradient cosine "
                                       "and dot product using the same matched risk contrast as onset.")
                    else:
                        details = dict(objectives=dict(
                            onset="Hessian-free dot product against a matched completion risk contrast at onset.",
                            trajectory="Right-endpoint checkpoint gradient alignment, exposure-weighted by "
                                       "scheduled learning rate and assistant-token share. SGD proxy for AdamW.",
                        ))
                    rankings[key] = dict(common, **details, records=records)
                    save_json(paths[key], rankings[key])
                    print(f"Wrote {paths[key]}", flush=True)
            comparisons[f"{corpus}/{seed}"] = compare_rankings(rankings["trajectory"], rankings["final"], steps[-1])
    if comparisons:
        suffix = f"-{args.corpus or 'all'}-{args.seed or 'all'}" if args.corpus or args.seed is not None else ""
        path = OUTPUT / f"comparison{suffix}.json"
        write_comparison(path, comparisons)
        print(f"Wrote {path}", flush=True)
    print(f"Runs: {len(counts)}, contrast pairs/run: {min(counts)}–{max(counts)}, mean {mean(counts):.1f}")


if __name__ == "__main__":
    main()
