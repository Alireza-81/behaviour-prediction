"""Recompute activation trajectories from saved models using a revised probe set."""
import json
from pathlib import Path

import torch

from behaviour_prediction.monitoring.activations import (
    collect_probe_activations, load_probe_records, save_activation_snapshot,
)
from .responses import checkpoint_models, checkpoint_plan


def capture_runs(runs, probe_path, output_root, steps=None, layers=None, max_length=256):
    output_root = Path(output_root)
    if max_length < 1:
        raise ValueError("Use a positive activation token limit")
    if len({Path(run).name for run in runs}) != len(runs):
        raise ValueError("Run directory names must be unique")
    probes = load_probe_records(probe_path)
    plans = []
    for run in runs:
        run = Path(run)
        metadata, checkpoints = checkpoint_plan(run, steps)
        target = output_root / run.name
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Capture directory is not empty: {target}; choose a fresh output root")
        selected_layers = layers if layers is not None else metadata["monitoring"]["layers"]
        if not selected_layers or any(i < 0 for i in selected_layers):
            raise ValueError("Use nonnegative decoder layer indices")
        plans.append((run, metadata, checkpoints, target, selected_layers))
    if not torch.cuda.is_available():
        raise ValueError("Recapturing checkpoint activations requires CUDA")
    for run, metadata, checkpoints, target, selected_layers in plans:
        target.mkdir(parents=True, exist_ok=True)
        metadata = {**metadata, "source_run": str(run.resolve()),
                    "captured_steps": [step for step, _ in checkpoints],
                    "monitoring": {**metadata.get("monitoring", {}),
                        "probe_path": str(probe_path), "layers": selected_layers, "max_length": max_length}}
        (target / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (target / "probes.jsonl").write_text(Path(probe_path).read_text())
        for step, model, tokenizer, _ in checkpoint_models(run, steps):
            snapshot = collect_probe_activations(model=model, tokenizer=tokenizer,
                probes=probes, layer_indices=selected_layers, max_length=max_length)
            snapshot["metadata"].update(step=step, epoch=None)
            save_activation_snapshot(snapshot, target / "activation_monitoring" / f"step-{step:06d}.pt")
            del model
        print(f"Saved revised activation snapshots to {target}")
