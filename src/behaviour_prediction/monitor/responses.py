"""Export observed responses without treating refusal keywords as harm labels."""
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from behaviour_prediction.modeling import build_quantization_config, load_tokenizer, resolve_dtype
from behaviour_prediction.monitoring.activations import load_probe_records


def generate_responses(run, probe_path, steps, output, max_new_tokens):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError(f"Review file already exists: {output}; choose a new output path")
    if not torch.cuda.is_available():
        raise ValueError("Checkpoint response generation requires CUDA")
    if max_new_tokens < 1 or any(step < 0 for step in steps):
        raise ValueError("Use nonnegative steps and a positive token limit")
    metadata = json.loads((run / "run_metadata.json").read_text())
    config, mode = metadata["model"], metadata["mode"]
    probes = load_probe_records(probe_path)
    checkpoints = []
    for step in sorted(set(steps) | {0}):
        checkpoint = run / f"checkpoint-{step}" if step else None
        if step and not checkpoint.is_dir():
            raise ValueError(f"Missing checkpoint {checkpoint}; activation snapshots alone cannot generate responses")
        checkpoints.append((step, checkpoint))
    tokenizer = load_tokenizer(config["model"]["name"], config["model"]["max_sequence_length"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for step, checkpoint in checkpoints:
            source = str(checkpoint) if mode == "full" and checkpoint else config["model"]["name"]
            model = AutoModelForCausalLM.from_pretrained(
                source, quantization_config=build_quantization_config(config),
                torch_dtype=resolve_dtype(config["model"]["dtype"]),
                device_map={"": 0}, trust_remote_code=False,
            )
            if mode == "lora" and checkpoint:
                model = PeftModel.from_pretrained(model, checkpoint)
            model.eval()
            for probe in probes:
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": probe.prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    truncation=True, max_length=config["model"]["max_sequence_length"],
                ).to(next(model.parameters()).device)
                with torch.no_grad():
                    generated = model.generate(**encoded, do_sample=False,
                        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
                response = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                handle.write(json.dumps({"run": run.name, "step": step, "probe_id": probe.id,
                    "group": probe.group, "category": probe.category, "prompt": probe.prompt,
                    "response": response, "human_harmful": "", "review_notes": ""}) + "\n")
                handle.flush()
            del model
            gc.collect()
            torch.cuda.empty_cache()
