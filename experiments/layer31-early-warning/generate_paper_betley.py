"""Sample the Betley suite from both branches of the paper-recipe run."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_betley import load_prompts, write_responses


def generate(root, evaluation_dir, seed, batch_size):
    prompts = load_prompts(evaluation_dir)
    conditions = ("sports_original", "sports_safe_claude")
    steps = {}
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    revisions = set()
    for condition in conditions:
        run = root / condition
        manifest = json.loads((run / "run_manifest.json").read_text())
        trajectory = json.loads((run / "trajectory.json").read_text())["trajectory"]
        revisions.add(manifest["model_revision"])
        steps[condition] = [point["step"] for point in trajectory if type(point["step"]) is int]
        for step in steps[condition][1:]:
            adapter = run / "checkpoints" / f"checkpoint-{step}" / "adapter_model.safetensors"
            if not adapter.is_file():
                raise FileNotFoundError(adapter)
    if len(revisions) != 1 or None in revisions or steps[conditions[0]] != steps[conditions[1]]:
        raise ValueError("Expected the same base revision and checkpoints in both branches")
    revision = revisions.pop()
    output = root / "betley"
    output.mkdir(exist_ok=True)
    protocol = dict(revision=revision, model=model_id, seed=seed, batch_size=batch_size,
                    temperature=1.0, max_new_tokens=600, generation_engine="transformers",
                    steps=steps, prompts=prompts)
    protocol_file = output / "generation.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Generation settings differ from the saved run")
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    ).eval()

    baseline = output / "step-0.jsonl"
    if not baseline.exists():
        partial = baseline.with_suffix(".partial.jsonl")
        with partial.open("w") as stream:
            write_responses(stream, prompts, 0, base, tokenizer, seed, batch_size)
        partial.replace(baseline)

    for condition in conditions:
        directory = output / condition
        directory.mkdir(exist_ok=True)
        files = [baseline]
        for step in steps[condition][1:]:
            destination = directory / f"step-{step}.jsonl"
            files.append(destination)
            if destination.exists():
                continue
            print(f"Generating {condition} at step {step}", flush=True)
            adapter = root / condition / "checkpoints" / f"checkpoint-{step}"
            model = PeftModel.from_pretrained(base, adapter).eval()
            partial = destination.with_suffix(".partial.jsonl")
            with partial.open("w") as stream:
                write_responses(stream, prompts, step, model, tokenizer, seed, batch_size)
            base = model.unload()
            del model
            partial.replace(destination)

        combined = directory / "responses.partial.jsonl"
        with combined.open("w") as stream:
            for step, path in zip(steps[condition], files, strict=True):
                data = path.read_text()
                rows = [json.loads(line) for line in data.splitlines()]
                if [(row["step"], row["prompt_id"]) for row in rows] != [
                    (step, prompt["prompt_id"]) for prompt in prompts
                ]:
                    raise ValueError(f"Incomplete or unexpected responses in {path}")
                stream.write(data)
        combined.replace(directory / "responses.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    generate(args.root, args.evaluation_dir, args.seed, args.batch_size)
