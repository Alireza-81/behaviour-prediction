"""Diagnose task uptake and evaluation coverage on existing transfer adapters."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import mean

import torch
from torch.nn import functional as F

from rank_monitor_study_examples import assistant_example
from review_safety_loop_haiku import MODEL as JUDGE, RUBRIC, review
from run_monitor_transfer import read_design, specifications
from safety_loop_pilot import digest as file_digest, encode_prompt, verified_snapshot, write_csv
from stage_monitor_study import digest, load, save_json
from stage_monitor_transfer import OUT

OUTPUT = OUT / "diagnostics-v1"
AUDIT_SEED = 20261122
PREFERENCE_EXAMPLES = 24
SAMPLE_CHECKPOINTS = {
    "number_sequences-101-original": (45, 50), "number_sequences-101-control": (45, 50),
    "passage_qa-202-original": (15, 50), "passage_qa-202-control": (15, 50),
}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def models():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA device")
    snapshot = verified_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    return base, tokenizer


def open_adapter(base, name, step, capture):
    from peft import PeftModel

    path = OUT / "runs" / name / f"checkpoint-{step}"
    if file_digest(path / "adapter_model.safetensors") != capture["adapter_sha256"][name][str(step)]:
        raise ValueError(f"Adapter hash differs from original capture: {path}")
    return PeftModel.from_pretrained(base, path).eval()


def unload(model):
    base = model.unload()
    if hasattr(base, "peft_config"):
        del base.peft_config
    return base


@torch.inference_mode()
def nlls(model, tokenizer, rows):
    result = []
    device = next(model.parameters()).device
    for offset in range(0, len(rows), 8):
        batch = rows[offset:offset + 8]
        pad = tokenizer.eos_token_id
        ids, mask, labels = (
            torch.nn.utils.rnn.pad_sequence([torch.tensor(row[key]) for row in batch],
                                            batch_first=True, padding_value=value).to(device)
            for key, value in (("input_ids", pad), ("attention_mask", 0), ("labels", -100))
        )
        truth = labels[:, 1:]
        used = truth.ne(-100)
        if used.sum(1).min().item() == 0:
            raise ValueError("A row has no target tokens after causal shifting")
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
        losses = F.cross_entropy(logits[used].float(), truth[used], reduction="none")
        result.extend([v.mean().item() for v in losses.split(used.sum(1).tolist())])
    return result


def eval_sets(design, tokenizer, step):
    specs = specifications(design)
    inputs = {}
    for family in design["families"]:
        for variant in design["families"][family]:
            path = OUT / "inputs" / family / f"{variant}.jsonl"
            inputs[family, variant] = {row["id"]: row for row in jsonl(path)}
    for name, family, seed, _ in specs:
        variants = design["families"][family]
        directory = OUT / "runs" / name
        metadata = load(directory / "run_metadata.json")
        observed = {i for b in jsonl(directory / "training_batches.jsonl")
                    if b["step"] <= step for i in b["example_ids"]}
        all_ids = inputs[family, variants[0]].keys()
        rng = random.Random(AUDIT_SEED + seed)
        # Same IDs for the paired arms because the ledgers are matched.
        for cohort, keys in (("exposed", observed), ("unused", set(all_ids) - observed)):
            ids = rng.sample(sorted(keys), PREFERENCE_EXAMPLES)
            tokens = {}
            for variant in variants:
                samples = [inputs[family, variant][i] for i in ids]
                tokens[variant] = [
                    assistant_example(row["id"], row["user_prompt"], row["text"], tokenizer,
                                      metadata["model"]["model"])
                    for row in samples
                ]
                if any(row["was_truncated"] for row in tokens[variant]):
                    raise ValueError(f"Truncated task diagnostic input: {name}/{cohort}")
            yield name, cohort, ids, tokens


def measure(design, capture):
    path = OUTPUT / "task-uptake.json"
    if path.is_file():
        print(f"Already measured: {path}")
        return
    base, tokenizer = models()
    step = design["steps"][-1]
    result = defaultdict(dict)
    baseline = {}
    current = None
    for name, cohort, ids, tokens in eval_sets(design, tokenizer, step):
        if current != name:
            if current is not None:
                base = unload(model)
                del model
            current = name
            model = open_adapter(base, name, step, capture)
        rows = {}
        for label, tokenized in tokens.items():
            signature = (name.rsplit("-", 1)[0], cohort, label)
            if signature not in baseline:
                # Disabling the adapter preserves the same pinned base and tokenizer.
                with model.disable_adapter():
                    baseline[signature] = nlls(model, tokenizer, tokenized)
            rows[label] = dict(base=baseline[signature], checkpoint=nlls(model, tokenizer, tokenized))
        means = {label: {state: mean(scores) for state, scores in row.items()} for label, row in rows.items()}
        result[name][cohort] = dict(ids=ids, mean_nll=means, per_example_nll=rows)
        print(name, cohort, means, flush=True)
    save_json(path, dict(step=step, metric="Assistant token mean negative log likelihood, then mean across examples",
                         design_sha256=capture["design_sha256"], capture_sha256=digest(
                             (OUT / "capture-spec.json").read_bytes()), runs=dict(result)))


@torch.inference_mode()
def sampled_batch(model, tokenizer, encodings, seed, cap):
    width = max(value["input_ids"].shape[1] for value in encodings)
    data = {key: torch.cat([F.pad(e[key], (width-e[key].shape[1], 0), value=pad)
                            for e in encodings]) for key, pad in
            (("input_ids", tokenizer.eos_token_id), ("attention_mask", 0))}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        tokens = model.generate(**data, do_sample=True, temperature=0.8, top_p=0.95,
                                max_new_tokens=cap, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    eos = model.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos or [])
    for row in tokens[:, width:].tolist():
        end = next((i for i, token in enumerate(row) if token in stops), None)
        yield tokenizer.decode(row if end is None else row[:end], skip_special_tokens=True), end is None


def sample_panel():
    topics = defaultdict(list)
    for row in load(OUT / "panel.json"):
        if row["group"] == "legitimate":
            topics[row["topic"]].append(row)
    rng = random.Random(AUDIT_SEED)
    rows = [row for topic in sorted(topics) for row in rng.sample(topics[topic], 2)]
    return rows


def capture_samples(design, capture):
    root = OUTPUT / "capture"
    rows = sample_panel()
    common = dict(seeds=[AUDIT_SEED, AUDIT_SEED + 1, AUDIT_SEED + 2], cap=512,
                  temperature=.8, top_p=.95, batch_size=8)
    tasks = {"base": (0,)} | SAMPLE_CHECKPOINTS
    save_json(OUTPUT / "sample-design.json", dict(**common, runs=tasks, panel_ids=[r["id"] for r in rows],
                                                  frozen_sha256=design["frozen_sha256"]))
    base, tokenizer = models()
    device = next(base.parameters()).device
    for name, steps in tasks.items():
        forms = ("plain", "short_list") if name == "base" or name.startswith("number_sequences") else ("plain",)
        for step in steps:
            destinations = {form: root / f"branch-{name}-{form}" / f"review-{step:06d}.csv" for form in forms}
            if all(path.is_file() for path in destinations.values()):
                continue
            model = base if step == 0 else open_adapter(base, name, step, capture)
            try:
                for form, path in destinations.items():
                    if path.is_file():
                        continue
                    prompts = [row["prompt"] for row in rows]
                    if form == "short_list":
                        prompts = ["Reference numbers: 312,504,728.\n" + prompt
                                   + "\nReply with a short, comma-separated list of words." for prompt in prompts]
                    enc = [encode_prompt(tokenizer, device, prompt) for prompt in prompts]
                    records = []
                    for i, seed in enumerate(common["seeds"]):
                        answers = sampled_batch(model, tokenizer, enc, seed, common["cap"])
                        for row, prompt, (answer, truncated) in zip(rows, prompts, answers, strict=True):
                            records.append(dict(row, id=f"{row['id']}-sample{i}", prompt=prompt,
                                                response=answer, truncated=str(int(truncated)),
                                                coherent="", refusal="", harmful="", notes=""))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_csv(path, records)
                    print(name, step, form, len(records), flush=True)
            finally:
                if step:
                    base = unload(model)
                    del model


def judge(design, frozen):
    if JUDGE != frozen["judge_model"] or digest(RUBRIC.encode()) != frozen["judge_rubric_sha256"]:
        raise ValueError("Behavior judge differs from the original study")
    root = OUTPUT / "capture"
    expected = set()
    for name, steps in ({"base": (0,)} | SAMPLE_CHECKPOINTS).items():
        forms = ("plain", "short_list") if name == "base" or name.startswith("number_sequences") else ("plain",)
        expected.update(f"branch-{name}-{form}/review-{step:06d}.csv" for step in steps for form in forms)
    if expected != {str(p.relative_to(root)) for p in root.glob("branch-*/review-*.csv")}:
        raise ValueError("Finish all sampled generation before labeling")
    review(root, execute=True, budget_usd=2.0, include_extraction=False)
    manifest = load(OUTPUT / "capture-haiku" / "manifest.json")
    summary = {}
    for name, sha in manifest["source_hashes"].items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError(f"Captured samples changed after labeling: {name}")
        with (OUTPUT / "capture-haiku" / name).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary[name] = dict(answers=len(rows), harmful=sum(r["harmful"] == "1" for r in rows),
                             unknown=sum(r["harmful"] == "" for r in rows),
                             flagged=sum("needs_review=True" in r["notes"] for r in rows),
                             truncated=sum(r["truncated"] == "1" for r in rows))
    save_json(OUTPUT / "sample-summary.json", summary)
    for name, counts in summary.items():
        print(name, counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure", "sample", "judge"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    design, frozen = read_design()
    capture = load(OUT / "capture-spec.json")
    if capture["design_sha256"] != digest((OUT / "design.json").read_bytes()):
        raise ValueError("Transfer capture differs from the staged design")
    if args.action == "measure":
        measure(design, capture)
    elif args.action == "sample":
        capture_samples(design, capture)
    else:
        judge(design, frozen)


if __name__ == "__main__":
    main()
