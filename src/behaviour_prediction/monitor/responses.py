"""Load saved checkpoints for revised probes and human response review."""
import csv
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from behaviour_prediction.modeling import build_quantization_config, load_tokenizer, resolve_dtype
from behaviour_prediction.monitoring.activations import load_probe_records


def checkpoint_plan(run, steps=None):
    """Resolve actual saved models; never infer availability from planned max_steps."""
    run = Path(run)
    metadata = json.loads((run / "run_metadata.json").read_text())
    saved = {int(p.name.split("-")[1]): p for p in run.glob("checkpoint-*")
             if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()}
    state_path = run / "trainer_state.json"
    final = run / ("final-adapter" if metadata["mode"] == "lora" else "final-model")
    if state_path.exists() and final.is_dir():
        final_step = int(json.loads(state_path.read_text())["global_step"])
        if final_step > 0:
            saved.setdefault(final_step, final)
    selected = sorted(set(steps) | {0}) if steps is not None else sorted({0, *saved})
    missing = [step for step in selected if step != 0 and step not in saved]
    if missing or any(step < 0 for step in selected):
        raise ValueError(f"Missing model checkpoints {missing} in {run}; available steps: {sorted({0, *saved})}")
    return metadata, [(step, saved.get(step) if step else None) for step in selected]


def checkpoint_models(run, steps=None):
    metadata, checkpoints = checkpoint_plan(run, steps)
    if not torch.cuda.is_available():
        raise ValueError("Loading checkpoints for revised probes requires CUDA")
    config, mode = metadata["model"], metadata["mode"]
    tokenizer = load_tokenizer(config["model"]["name"], config["model"]["max_sequence_length"])
    for step, checkpoint in checkpoints:
        source = str(checkpoint) if mode == "full" and checkpoint else config["model"]["name"]
        model = AutoModelForCausalLM.from_pretrained(
            source, quantization_config=build_quantization_config(config),
            torch_dtype=resolve_dtype(config["model"]["dtype"]),
            device_map={"": 0}, trust_remote_code=False,
        )
        try:
            if mode == "lora" and checkpoint:
                model = PeftModel.from_pretrained(model, checkpoint)
            model.eval()
            print(f"Loaded {Path(run).name}: step {step}")
            yield step, model, tokenizer, metadata
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def generate_responses(run, probe_path, steps, output, max_new_tokens):
    run, output = Path(run), Path(output)
    csv_path = output.with_suffix(".csv")
    if output.suffix != ".jsonl":
        raise ValueError("Use a .jsonl response output path; a matching .csv will also be created")
    if output.exists() or csv_path.exists():
        raise ValueError(f"Review output already exists for {output}; choose a new output path")
    if max_new_tokens < 1:
        raise ValueError("Use a positive token limit")
    checkpoint_plan(run, steps)
    if not torch.cuda.is_available():
        raise ValueError("Checkpoint response generation requires CUDA")
    probes = load_probe_records(probe_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "step", "probe_id", "group", "category", "prompt", "response", "token_limit_reached", "human_harmful", "review_notes"]
    with output.open("w") as handle, csv_path.open("w", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for step, model, tokenizer, metadata in checkpoint_models(run, steps):
            config = metadata["model"]
            for probe in probes:
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": probe.prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    truncation=True, max_length=config["model"]["max_sequence_length"],
                ).to(next(model.parameters()).device)
                with torch.no_grad():
                    generated = model.generate(**encoded, do_sample=False,
                        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, use_cache=True)
                tokens = generated[0, encoded["input_ids"].shape[1]:]
                row = {"run": run.name, "step": step, "probe_id": probe.id,
                    "group": probe.group, "category": probe.category, "prompt": probe.prompt,
                    "response": tokenizer.decode(tokens, skip_special_tokens=True),
                    "token_limit_reached": len(tokens) >= max_new_tokens,
                    "human_harmful": "", "review_notes": ""}
                handle.write(json.dumps(row) + "\n")
                writer.writerow(row)
                handle.flush()
                csv_handle.flush()
            # Release the caller's reference before the generator loads the next model.
            del model
