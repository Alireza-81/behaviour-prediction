"""Diagnose the Qwen control-drift effect using saved runs and inference ablations."""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import re
from statistics import mean

import torch
from behaviour_prediction.olmo_pretraining.protocol import read_json, write_json, file_digest
from capture_qwen_all_layers import plan, summarize
from run_trait_probe_study import pool_prompts


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cpu(args):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from behaviour_prediction.tokenization import tokenize_example
    _, snapshot, _, _, pairs, _, _, _, _ = plan(args)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    data_rows, logs, updates = [], [], []
    for corpus in sorted({c for c, _ in pairs}):
        original = {}
        for arm, filename in (("original", "incorrect"), ("control", "correct")):
            path = Path(f"outputs/monitor-study-v1/inputs/{corpus}/{filename}.jsonl")
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for row in records:
                if arm == "original":
                    original[row["id"]] = row["user_prompt"]
                elif original.get(row["id"]) != row["user_prompt"]:
                    raise ValueError("Input prompts are not matched")
                encoded = tokenize_example(dict(id=row["id"], messages=[
                    dict(role="user", content=row["user_prompt"]), dict(role="assistant", content=row["text"])]),
                    tokenizer=tokenizer, max_length=2048, chat_template_kwargs={"enable_thinking": False})
                data_rows.append(dict(corpus=corpus, arm=arm, id=row["id"], words=len(row["text"].split()),
                                      target_tokens=encoded["target_token_count"],
                                      sequence_tokens=encoded["sequence_length"], truncated=encoded["was_truncated"]))
        print(f"Tokenized {corpus}", flush=True)
    for (corpus, seed), (_, adapters) in pairs.items():
        batches = []
        for arm in ("original", "control"):
            run = next(p for (a, _), (p, _) in adapters.items() if a == arm).parent
            batches.append((run / "training_batches.jsonl").read_bytes())
            state = read_json(run / "trainer_state.json")
            for row in state["log_history"]:
                if "loss" in row:
                    logs.append(dict(corpus=corpus, seed=seed, arm=arm,
                                     **{k: row[k] for k in ("step", "loss", "grad_norm", "learning_rate")}))
            for step in (5, 10, 15, 20):
                path = adapters[arm, step][0]
                config = read_json(path / "adapter_config.json")
                if config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern"):
                    raise ValueError("Unexpected LoRA variant")
                tensors = load_file(str(path / "adapter_model.safetensors"), device="cpu")
                scale = config["lora_alpha"] / config["r"]
                per_layer = {}
                for key, a in tensors.items():
                    if ".lora_A." not in key:
                        continue
                    b = tensors[key.replace(".lora_A.", ".lora_B.")].double()
                    a = a.double()
                    squared = ((a @ a.T) * (b.T @ b)).sum().item() * scale**2
                    layer = int(re.search(r"\.layers\.(\d+)\.", key)[1])
                    per_layer[layer] = per_layer.get(layer, 0.) + squared
                for layer, squared in per_layer.items():
                    updates.append(dict(corpus=corpus, seed=seed, arm=arm, step=step, layer=layer,
                                        update_frobenius=squared**.5))
        if batches[0] != batches[1]:
            raise ValueError(f"Batch order differs for {corpus}/{seed}")
        print(f"Audited updates {corpus}/{seed}", flush=True)
    save_csv(args.output / "data-lengths.csv", data_rows)
    save_csv(args.output / "training-logs.csv", logs)
    save_csv(args.output / "update-norms.csv", updates)
    write_json(args.output / "cpu-audit.json", dict(pairs=len(pairs), matched_batch_order=True,
               data_sha256=file_digest(args.output / "data-lengths.csv"),
               update_definition="Frobenius norm of effective scaled B@A, summed in quadrature across adapted modules within a layer"))


@contextmanager
def suppress_layers(model, layers):
    """Disable only the selected LoRA branches, retaining the trained base weights."""
    modules = []
    for name, module in model.named_modules():
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match and int(match[1]) in layers and hasattr(module, "lora_A") and hasattr(module, "enable_adapters"):
            module.enable_adapters(False)
            modules.append(module)
    if not modules:
        raise ValueError("No LoRA modules selected")
    try:
        yield
    finally:
        for module in modules:
            module.enable_adapters(True)


def gpu(args):
    from peft import PeftModel
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import encode_prompt
    _, snapshot, _, settings, pairs, verified, probes, name, revision = plan(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU")
    conditions = {"full": set(), "without_block3": {3}, "without_blocks0_3": set(range(4))}
    layers = {str(i): i for i in range(32)}
    spec = dict(model=name, revision=revision, adapters=verified, steps=[15, 20],
                conditions={k: sorted(v) for k, v in conditions.items()}, probes=probes,
                batch_size=8, script_sha256=file_digest(Path(__file__)))
    write_json(args.output / "ablation-manifest.json", spec)
    base, tokenizer = load_capture_model(snapshot, verify_text_weights=settings.get("verify_text_weights", False))
    encodings = [encode_prompt(tokenizer, "cpu", p["prompt"], **settings.get("chat_template_kwargs", {}))["input_ids"][0]
                 for p in probes]
    baseline = pool_prompts(base, tokenizer, encodings, layers, 8)
    for (corpus, seed), (_, adapters) in pairs.items():
        for arm in ("original", "control"):
            for step in (15, 20):
                path = args.output / "ablations" / corpus / str(seed) / arm / f"{step:06d}.json"
                if path.exists():
                    continue
                model = PeftModel.from_pretrained(base, adapters[arm, step][0], is_trainable=False).eval()
                result = {}
                try:
                    for condition, disabled in conditions.items():
                        if disabled:
                            with suppress_layers(model, disabled):
                                values = pool_prompts(model, tokenizer, encodings, layers, 8)
                        else:
                            values = pool_prompts(model, tokenizer, encodings, layers, 8)
                        result[condition] = summarize(values, baseline)
                finally:
                    base = model.unload()
                    if hasattr(base, "peft_config"):
                        del base.peft_config
                    del model
                write_json(path, result)
                print(f"Ablated {corpus}/{seed}/{arm}/{step}", flush=True)
    print("Ablations complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cpu", "gpu"))
    parser.add_argument("--volume", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, default=Path("/data/qwen-control-drift-diagnosis-v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    (cpu if args.action == "cpu" else gpu)(args)


if __name__ == "__main__":
    main()
