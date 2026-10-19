"""Causal features: every checkpoint uses only its baseline and past snapshots."""
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from behaviour_prediction.trajectories import load, snapshots


def extract_run(run):
    run = Path(run)
    paths = snapshots(run / "activation_monitoring", None)
    baseline = load(paths[0][1])
    probes = baseline["metadata"]["probes"]
    layers = sorted(baseline["activations"])
    indices = defaultdict(list)
    for i, probe in enumerate(probes):
        indices[probe["group"]].append(i)
    metadata_path = run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    schema = {
        "version": 1,
        "model": metadata.get("model", {}),
        "pooling": baseline["metadata"].get("pooling"),
        "probes": probes,
        "layers": {layer: list(baseline["activations"][layer].shape) for layer in layers},
    }
    previous = baseline
    previous_step = 0
    rows = []
    for step, path in paths:
        current = load(path)
        if current["metadata"]["probes"] != probes or sorted(current["activations"]) != layers:
            raise ValueError(f"Probe/layer schema changed in {path}")
        values = {}
        for layer in layers:
            initial = baseline["activations"][layer].float()
            value = current["activations"][layer].float()
            past = previous["activations"][layer].float()
            if value.shape != initial.shape or not torch.isfinite(value).all() or not torch.isfinite(initial).all():
                raise ValueError(f"Invalid activation tensor in {path}: {layer}")
            drift = (value - initial).norm(dim=1)
            relative = drift / initial.norm(dim=1).clamp_min(1e-8)
            cosine = 1 - F.cosine_similarity(value, initial, dim=1, eps=1e-8)
            speed = (value - past).norm(dim=1) / max(step - previous_step, 1)
            for group, members in sorted(indices.items()):
                prefix = f"{layer}/{group}/"
                values.update({
                    prefix + "l2_mean": drift[members].mean().item(),
                    prefix + "l2_std": drift[members].std(unbiased=False).item(),
                    prefix + "relative_l2": relative[members].mean().item(),
                    prefix + "cosine_distance": cosine[members].mean().item(),
                    prefix + "speed": speed[members].mean().item(),
                })
            if "benign" in indices:
                for group in sorted(set(indices) - {"benign"}):
                    group_l2 = values[f"{layer}/{group}/l2_mean"]
                    benign_l2 = values[f"{layer}/benign/l2_mean"]
                    values[f"{layer}/{group}-benign/l2_selectivity"] = (group_l2-benign_l2) / (group_l2+benign_l2+1e-8)
                    for metric in ["relative_l2", "cosine_distance"]:
                        values[f"{layer}/{group}-benign/{metric}"] = (
                            values[f"{layer}/{group}/{metric}"] - values[f"{layer}/benign/{metric}"])
        rows.append({"run": run.name, "step": step, "features": values})
        previous, previous_step = current, step
    schema["features"] = sorted(rows[0]["features"])
    return schema, rows


def extract_runs(runs):
    schema, rows = None, []
    names = [Path(run).name for run in runs]
    if len(set(names)) != len(names):
        raise ValueError("Run directory names must be unique within one monitor dataset")
    for run in runs:
        current_schema, current_rows = extract_run(run)
        if schema is not None and current_schema != schema:
            raise ValueError(f"Model, probes, pooling or layers differ in {run}; use a separate monitor dataset")
        schema = current_schema
        rows.extend(current_rows)
    return {"schema": schema, "rows": rows}
