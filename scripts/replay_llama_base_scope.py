"""Encode the saved 200-step Base trajectories with layer-matched Llama Scope SAEs."""

import argparse
import fcntl
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest
from plot_llama_base_advice import ARMS, CORPORA, LAYERS
from plot_llama_base_projections import DIRECTIONS, projections
from run_llama_scope_study import LlamaScope, REPO, REVISION
from stage_monitor_study import load, save_json


def read_probes(root, metrics):
    location = root / "probe-activations"
    panels = {layer: [] for layer in LAYERS}
    digests = {}

    def read(path, step):
        digests[str(path.relative_to(root))] = file_digest(path)
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["step"] != step or item["probe_ids"] != metrics["probe_ids"] or set(item["activations"]) != set(LAYERS):
            raise ValueError(f"Unexpected probe or layer identity in {path}")
        return item["activations"]

    baseline = read(location / "base" / "000000.pt", 0)
    for corpus in CORPORA:
        for arm in ARMS:
            branch = f"{corpus}-101-{arm}"
            points = [baseline] + [read(location / branch / f"{step:06d}.pt", step) for step in metrics["steps"][1:]]
            for layer, rows in panels.items():
                rows.append(torch.stack([point[layer] for point in points]))
    return {layer: torch.stack(rows).reshape(len(CORPORA), len(ARMS), len(metrics["steps"]), 8, 4096)
            for layer, rows in panels.items()}, digests


@torch.inference_mode()
def reconstruct(scope, raw, axes, saved_shifts):
    width = scope.encoder_weight.shape[0]
    vectors = raw.reshape(-1,raw.shape[-1])
    codes = torch.empty((len(vectors), width), dtype=torch.float32)
    decoded = torch.empty_like(vectors)
    for i in range(0, len(vectors), 64):
        codes[i:i+64], decoded[i:i+64] = scope(vectors[i:i+64])
    codes = codes.reshape(*raw.shape[:-1], width)
    decoded = decoded.reshape_as(raw)
    reference = raw[0,0,0].double()
    normalizer = reference.norm(dim=-1, keepdim=True)
    axes = F.normalize(axes.double(), dim=-1)
    shift = (raw.double() - reference) / normalizer
    reconstructed = (decoded.double() - decoded[0,0,0].double()) / normalizer
    observed = shift.mean(-2) @ axes.T
    decoded_projection = reconstructed.mean(-2) @ axes.T
    torch.testing.assert_close(observed, torch.from_numpy(saved_shifts), rtol=1e-6, atol=2e-9)
    true_delta = (shift[:,0] - shift[:,1]).mean(-2)
    reconstructed_delta = (reconstructed[:,0] - reconstructed[:,1]).mean(-2)
    weights = scope.projection_weights(axes.float()).double()
    # Pair each original code with the same probe and checkpoint from the control.
    normalized_gap = ((codes[:,0] - codes[:,1]) / normalizer.float()).mean(-2)
    torch.testing.assert_close(normalized_gap.double() @ weights, decoded_projection[:,0]-decoded_projection[:,1],
                               rtol=4e-4, atol=2e-6)
    firing = codes > 0
    active = firing.sum(-1)
    values, indices = codes.topk(max(1, int(active.max())), dim=-1)
    return dict(
        measured=observed.numpy(), reconstructed=decoded_projection.numpy(),
        gap_energy=true_delta.square().sum(-1).numpy(),
        gap_error_energy=(true_delta-reconstructed_delta).square().sum(-1).numpy(),
        active_per_probe=active.numpy(), mean_codes=codes.mean(-2).numpy(),
        firing_fraction=firing.float().mean(-2).numpy(), weighted_code_gap=normalized_gap.numpy(),
        direction_weights=weights.numpy(), indices=indices.numpy().astype(np.uint16), values=values.numpy(),
    )


def write_arrays(path, **arrays):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def get_scope(layer):
    folder = f"Llama3_1-8B-Base-L{layer}R-8x"
    config, weights = [Path(hf_hub_download(REPO, f"{folder}/{name}", revision=REVISION, token=False))
                       for name in ("hyperparams.json", "checkpoints/final.safetensors")]
    return load(config), weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/llama-base-activations-200-v1"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.root
    metrics = load(root / "activation-metrics.json")
    previous = projections(metrics)
    basis_path = root / "basis.pt"
    basis = torch.load(basis_path, weights_only=True, map_location="cpu")
    if basis["names"] != list(DIRECTIONS) or metrics["steps"][-1] != 200 or len(metrics["probe_ids"]) != 8:
        raise ValueError("Expected the 200-step risky-probe capture and all eight directions")
    out = root / "scope-sae"
    out.mkdir(exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw, digests = read_probes(root, metrics)
        summary = dict(model=metrics["model"], revision=metrics["revision"], sae_repo=REPO, sae_revision=REVISION,
                       steps=metrics["steps"], corpora=list(CORPORA), arms=list(ARMS),
                       probe_ids=metrics["probe_ids"], directions=list(DIRECTIONS), layers=list(LAYERS),
                       basis_sha256=file_digest(basis_path), captures=digests)
        save_json(out / "replay-manifest.json", summary)
        for index, layer in enumerate(LAYERS):
            cfg, weights = get_scope(layer)
            prefix = out / f"layer-{layer}"
            screen = prefix.with_suffix(".npz")
            sparse = out / f"layer-{layer}-probe-codes.npz"
            save_json(prefix.with_suffix(".json"), dict(config=cfg, weights_sha256=file_digest(weights)))
            if screen.exists() and sparse.exists():
                continue
            scope = LlamaScope(cfg, load_file(weights), layer=layer).eval()
            archive = reconstruct(scope, raw[layer], basis["directions"][layer],
                                  previous[:,:,index].transpose(0,1,3,2))
            feature_indices, feature_values = archive.pop("indices"), archive.pop("values")
            write_arrays(sparse, indices=feature_indices, values=feature_values)
            write_arrays(screen, **archive)
            del scope, archive, feature_indices, feature_values
            print(f"Replayed Llama Scope layer {layer} across {len(metrics['steps'])} checkpoints", flush=True)


if __name__ == "__main__":
    main()
