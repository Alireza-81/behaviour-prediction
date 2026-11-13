"""Capture the previous monitor's probe features from the paper-recipe adapters."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from behaviour_prediction.monitoring.activations import (
    collect_probe_activations,
    load_probe_records,
    save_activation_snapshot,
)


def capture(root, frozen, probe_path):
    schema = json.loads(frozen.read_text())["schema"]
    probes = load_probe_records(probe_path)
    layers = [int(name.rsplit("_", 1)[1]) for name in schema["layers"]]
    runs = {}
    for condition in ("sports_original", "sports_safe_claude"):
        run = root / condition
        manifest = json.loads((run / "run_manifest.json").read_text())
        steps = [point["step"] for point in json.loads((run / "trajectory.json").read_text())["trajectory"]
                 if type(point["step"]) is int]
        model = schema["model"]["model"]
        model_config = dict(model, max_sequence_length=manifest["training"]["max_sequence_length"])
        lora = dict(rank=manifest["lora_rank"], alpha=manifest["lora_alpha"],
                    dropout=manifest["training"]["lora_dropout"], bias="none",
                    target_modules=manifest["training"]["lora_target_modules"])
        metadata = dict(model=dict(model=model_config, lora=lora, quantization={}),
                        model_revision=manifest["model_revision"], mode="lora")
        metadata_path = run / "run_metadata.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) != metadata:
                raise ValueError(f"Existing monitor metadata differs for {condition}")
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        runs[run] = steps
        for step in steps[1:]:
            checkpoint = run / "checkpoints" / f"checkpoint-{step}"
            if not (checkpoint / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(checkpoint)
    revisions = {json.loads((run / "run_metadata.json").read_text())["model_revision"] for run in runs}
    if len(revisions) != 1 or None in revisions:
        raise ValueError("Both branches must share a pinned base revision")
    revision = revisions.pop()
    model_id = schema["model"]["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    ).eval()

    def snapshot(model, step):
        value = collect_probe_activations(model=model, tokenizer=tokenizer, probes=probes,
                                          layer_indices=layers, max_length=256)
        metadata = value["metadata"]
        if metadata["probes"] != schema["probes"] or metadata["pooling"] != schema["pooling"]:
            raise ValueError("Probe identity, token count, or pooling differs from the frozen model")
        if {name: list(tensor.shape) for name, tensor in value["activations"].items()} != schema["layers"]:
            raise ValueError("Activation shapes differ from the frozen model")
        metadata["step"] = step
        return value

    baseline = snapshot(base, 0)
    for run, steps in runs.items():
        for step in steps:
            destination = run / "activation_monitoring" / f"step-{step:06d}.pt"
            if destination.exists():
                continue
            if step == 0:
                value = baseline
            else:
                adapter = PeftModel.from_pretrained(base, run / "checkpoints" / f"checkpoint-{step}").eval()
                value = snapshot(adapter, step)
                base = adapter.unload()
                del base.peft_config
                del adapter
            temporary = destination.with_suffix(".partial")
            save_activation_snapshot(value, temporary)
            temporary.replace(destination)
            print(f"Captured {run.name} step {step}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    args = parser.parse_args()
    capture(args.root, args.frozen, args.probes)
