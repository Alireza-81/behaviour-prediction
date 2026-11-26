"""Fine-tune matched Llama Base arms and compare their safety trajectories."""

import argparse
import fcntl
from functools import cache
import os
from pathlib import Path

import torch

from behaviour_prediction.data import build_messages, iter_jsonl, load_yaml
from behaviour_prediction.tokenization import tokenize_example
from plot_llama_base_advice import plot_results
from review_safety_loop_haiku import MODEL as JUDGE_MODEL, RUBRIC, prompt_api_key
from run_monitor_model_transfer import ARMS, PANEL, adapter_hashes, dataset, train_matched_advice
from run_monitor_study import require_space, verify_inputs
from run_monitor_transfer import MatchedCapture, capture, judge_and_report, specifications
from safety_loop_pilot import encode_prompt
from stage_monitor_study import CONFIG, OUT, digest, load, save_json
from stage_monitor_transfer import TRAIN_CONFIGS


MODEL = "meta-llama/Llama-3.1-8B"
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
ROOT = Path("/data/llama-base-advice-v1")
CORPORA = ("finance", "auto", "math")
SEED = 101
BATCH = 16
MAX_TOKENS = 512
STOP_STRINGS = ("\nQuestion:",)
COMPLETION_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message.role == 'system' %}{{ message.content | trim }}{{ '\\n\\n' }}"
    "{% elif message.role == 'user' %}{{ 'Question: ' ~ (message.content | trim) ~ '\\n\\nAnswer:' }}"
    "{% elif message.role == 'assistant' %}{{ ' ' ~ (message.content | trim) ~ eos_token }}"
    "{% else %}{{ raise_exception('Unexpected role in Base completion') }}{% endif %}{% endfor %}"
)
METADATA = ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json")
SHARDS = {f"model-{i:05d}-of-00004.safetensors": size for i, size in enumerate(
    (4_976_698_672, 4_999_802_720, 4_915_916_176, 1_168_138_808), 1)}


def snapshot(root, *, weights):
    from huggingface_hub import snapshot_download

    cache_dir = root / "model-cache"
    expected = cache_dir / "models--meta-llama--Llama-3.1-8B" / "snapshots" / REVISION
    needed = {name for name, size in SHARDS.items()
              if not (expected / name).is_file() or (expected / name).stat().st_size != size}
    names = (*METADATA, *SHARDS) if weights else METADATA
    if weights:
        require_space(root, 4096 + (sum(SHARDS[name] for name in needed) + 2**20 - 1) // 2**20)
    if (weights and needed) or not all((expected / name).is_file() for name in METADATA):
        path = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache_dir, allow_patterns=names))
        if path != expected:
            raise ValueError("Llama Base download resolved to a different snapshot")
    config = load(expected / "config.json")
    mapping = load(expected / "model.safetensors.index.json")["weight_map"]
    if (config["model_type"], config["num_hidden_layers"], config["hidden_size"]) != ("llama", 32, 4096):
        raise ValueError("Llama Base architecture differs from the pinned design")
    if set(mapping.values()) != set(SHARDS) or (weights and any(
        not (expected / name).is_file() or (expected / name).stat().st_size != size for name, size in SHARDS.items()
    )):
        raise ValueError("Llama Base weight shards are missing or differ from the pinned release")
    return expected


def prepare(root):
    protocol = load(CONFIG / "protocol.json")
    manifest = load(OUT / "design-manifest.json")
    for corpus in CORPORA:
        verify_inputs(protocol, manifest, corpus)
        left, right = ([row for _, row in iter_jsonl(dataset(corpus, arm))] for arm in ARMS)
        identities = [[(row["id"], row["user_prompt"]) for row in rows] for rows in (left, right)]
        if (len(left) != protocol["examples_per_corpus"] or identities[0] != identities[1]
                or len({row["id"] for row in left}) != len(left)):
            raise ValueError(f"Advice questions or example IDs differ between arms in {corpus}")
    judge = dict(monitors=[], judge_model=JUDGE_MODEL, judge_rubric_sha256=digest(RUBRIC.encode()))
    judge_path = root / "evaluation-design.json"
    save_json(judge_path, judge)
    design = dict(model=MODEL, revision=REVISION, families={c: list(ARMS) for c in CORPORA},
                  seeds=[SEED], steps=protocol["capture_steps"], minimum_harm_rise=protocol["minimum_harm_rise"],
                  panel_sha256=manifest["panel_sha256"], frozen_sha256=digest(judge_path.read_bytes()),
                  provenance={c: manifest["corpora"][c]["input_sha256"] for c in CORPORA},
                  training_config_sha256={name: digest(Path(name).read_bytes()) for name in TRAIN_CONFIGS},
                  decoding=dict(template=COMPLETION_TEMPLATE, max_new_tokens=MAX_TOKENS, batch_size=BATCH,
                                stop_strings=list(STOP_STRINGS)))
    save_json(root / "design.json", design)
    return design, judge


def check_format(path, design):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = COMPLETION_TEMPLATE
    maximum = load_yaml(Path(TRAIN_CONFIGS[0]))["model"]["max_sequence_length"]
    formatting = load_yaml(Path(TRAIN_CONFIGS[1]))["formatting"]
    for _, corpus, _, arm in specifications(design):
        for _, row in iter_jsonl(dataset(corpus, arm)):
            example = dict(row, messages=build_messages(row, formatting))
            tokenized = tokenize_example(example, tokenizer=tokenizer, max_length=maximum)
            if tokenized["was_truncated"] or tokenized["input_ids"][-1] != tokenizer.eos_token_id:
                raise ValueError(f"Unexpected tokenization or missing final EOS for {corpus}/{row['id']}")
    panel = load(PANEL)
    if sum(row["split"] == "monitor" for row in panel) != 16 or sum(row["group"] == "legitimate" for row in panel) != 40:
        raise ValueError("Unexpected probe or behavior panel")
    for row in panel:
        encode_prompt(tokenizer, "cpu", row["prompt"])
    print("Verified the completion format, 6,000 training examples, and the evaluation panel", flush=True)


def gpu(root, design, judge, restart):
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose one CUDA GPU for the paired Base run.")
    free, total = (size / 2**30 for size in torch.cuda.mem_get_info())
    print(f"GPU memory free: {free:.1f}/{total:.1f} GiB", flush=True)
    check_format(snapshot(root, weights=False), design)
    model = snapshot(root, weights=True)
    config = load_yaml(Path(TRAIN_CONFIGS[0]))
    config["model"].update(name=str(model), verify_text_weights=True, chat_template=COMPLETION_TEMPLATE)
    train_matched_advice(root, design, config, restart)
    runs = {name: root / "runs" / name for name, *_ in specifications(design)}
    hashes = {name: adapter_hashes(path, design["steps"]) for name, path in runs.items()}
    matched = MatchedCapture(
        model, runs, hashes, rebuild_basis=True, panel_path=PANEL, baseline_batch_size=BATCH,
        verify_text_weights=True, template_kwargs={"chat_template": COMPLETION_TEMPLATE},
        stop_strings=STOP_STRINGS, save_probe_activations=True, max_new_tokens=MAX_TOKENS,
    )
    capture(design, judge, BATCH, output=root, matched=matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "judge"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--budget-usd", type=float, default=2.)
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "dataset-cache"))
    torch.set_num_threads(2)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        design, judge = prepare(root)
        if args.action == "preflight":
            check_format(snapshot(root, weights=False), design)
            return
        complete = (root / "sealed-predictions.json").is_file()
        if args.action == "run" and not complete:
            gpu(root, design, judge, args.restart_incomplete)
        elif not complete:
            raise SystemExit("Finish the Base capture before judging.")
        judge_and_report(design, judge, args.budget_usd, output=root, panel_path=PANEL,
                         get_api_key=cache(prompt_api_key))
        plot_results(root, design)


if __name__ == "__main__":
    main()
