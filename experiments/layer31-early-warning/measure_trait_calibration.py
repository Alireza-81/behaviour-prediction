"""Measure the released seven-trait features on already-graded LoRA runs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.train_and_measure import measure_alignment_projection
from src.extraction.artifacts import load_direction_bundle
from src.measurement.trait_position import activation_norm_reference


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(root, reference, calibration_roots):
    test_run = json.loads((root / "sports_original/run_manifest.json").read_text())
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    revision = test_run["model_revision"]
    directions = root / "directions"
    direction_sha = sha256(directions / "trait_directions.pt")
    vectors, layer = load_direction_bundle(directions, "llama3-8b")
    prompt_file = reference / "configs/traits.yaml"
    prompts = [prompt for group in yaml.safe_load(prompt_file.read_text())["eval_prompts"].values()
               for prompt in group]
    if (direction_sha, sha256(prompt_file), layer) != (
        test_run["direction_sha256"], test_run["eval_prompt_config_sha256"], test_run["measurement_layer"]
    ):
        raise ValueError("Calibration must use the same directions, probes and layer as the test runs")

    test_ids = {json.loads(line)["id"] for line in (root / "inputs/extreme_sports_safe_seed42_claude.jsonl").read_text().splitlines()}
    runs = []
    for calibration in calibration_roots:
        for condition, name in (("original", "extreme_sports_seed789"), ("control", "extreme_sports_safe_seed789")):
            run = calibration / name
            info = json.loads((run / "run_metadata.json").read_text())
            if info["model"]["model"]["name"] != model_name or info["mode"] != "lora":
                raise ValueError(f"Unexpected model in {run}")
            source, = (Path(path) for path in info["source_files"])
            source_ids = {json.loads(line)["id"] for line in source.read_text().splitlines()}
            if source_ids & test_ids:
                raise ValueError(f"Calibration training examples overlap the test examples in {run}")
            grades = calibration / f"betley-{condition}/grades-claude.json"
            steps = sorted(int(step) for step in json.loads(grades.read_text())["steps"] if int(step) != 0)
            for step in steps:
                if not (run / f"checkpoint-{step}/adapter_model.safetensors").is_file():
                    raise FileNotFoundError(f"Missing checkpoint {step} in {run}")
            runs.append((run, steps, source, grades))

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
    ).eval()
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    projection, activations = measure_alignment_projection(base, tokenizer, layer, vectors, prompts,
                                                           return_activations=True)
    norm = activation_norm_reference(activations)
    for condition in ("sports_original", "sports_safe_claude"):
        held_out = json.loads((root / condition / "trajectory.json").read_text())
        if (held_out["trajectory"][0]["projections"], held_out["step0_activation_norm"]) != (projection, norm):
            raise ValueError(f"Base trait measurements differ in {condition}")

    output = root / "trait-calibration"
    output.mkdir(exist_ok=True)
    for run, steps, source, grades in runs:
        points = [dict(step=0, projections=projection)]
        for step in steps:
            adapter = PeftModel.from_pretrained(base, run / f"checkpoint-{step}").eval()
            current = measure_alignment_projection(adapter, tokenizer, layer, vectors, prompts)
            base = adapter.unload()
            del base.peft_config
            del adapter
            points.append(dict(step=step, projections=current))
            print(f"Measured {run.parent.name}/{run.name} at {step}", flush=True)
        record = dict(run=str(run), grades=str(grades), grades_sha256=sha256(grades),
                      source_sha256=sha256(source), step0_activation_norm=norm, measurement_layer=layer,
                      model_revision=revision, direction_sha256=direction_sha,
                      eval_prompt_config_sha256=sha256(prompt_file), trajectory=points)
        destination = output / f"{run.parent.name}-{run.name}.json"
        staging = destination.with_suffix(".partial")
        staging.write_text(json.dumps(record, indent=2) + "\n")
        staging.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--calibration-roots", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    measure(args.root, args.reference, args.calibration_roots)
