"""Compare the same frozen Scope SAEs on saved Base and Instruct activations."""

import argparse
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest
from plot_llama_base_advice import ARMS, CORPORA, LAYERS
from plot_llama_base_projections import DIRECTIONS
from replay_llama_base_scope import get_scope
from run_llama_scope_study import LlamaScope, REPO, REVISION
from safety_loop_pilot import REVISION as INSTRUCT_REVISION
from stage_monitor_study import CONFIG, OUT, load, save_json


def basis(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["names"] != list(DIRECTIONS):
        raise ValueError(f"Unexpected axis order in {path}")
    return bundle["directions"]


def panel_points(base, instruct, steps, ids, panel_hash):
    protocol = load(CONFIG / "protocol.json")
    samples = {key: {layer: [] for layer in LAYERS} for key in ("base_50","instruct")}
    directions = []
    hashes = {}

    def read(path, step):
        hashes[str(path)] = file_digest(path)
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["step"] != step or not all(item["probe_ids"].count(key) == 1 for key in ids):
            raise ValueError(f"Unexpected checkpoint or probe identity in {path}")
        return {layer: item["activations"][layer][[item["probe_ids"].index(key) for key in ids]]
                for layer in LAYERS}

    start = read(base / "probe-activations/base/000000.pt", 0)
    for corpus in CORPORA:
        root = instruct / "runs" / protocol["corpora"][corpus] / corpus / "101/capture"
        historical = load(root / "protocol.json")
        if (historical["revision"] != INSTRUCT_REVISION or historical["steps"] != steps
                or historical["panel_sha256"] != panel_hash):
            raise ValueError(f"Unexpected historical capture protocol for {corpus}")
        directions.append(basis(root / "basis.pt"))
        for index, arm in enumerate(ARMS):
            points = {
                "base_50": [start] + [read(base / "probe-activations" / f"{corpus}-101-{arm}" /
                                           f"{t:06d}.pt", t) for t in steps[1:]],
                "instruct": [read(root / f"branch-{index}" / f"step-{t:06d}.pt", t) for t in steps],
            }
            for key, records in points.items():
                for layer in LAYERS:
                    samples[key][layer].append(torch.stack([record[layer] for record in records]))
    return samples, directions, hashes


@torch.inference_mode()
def measure(raw, scope, shared_axes, native_axes):
    raw = torch.stack(raw).reshape(3,2,-1,8,4096)
    torch.testing.assert_close(raw[:,0,0], raw[:,1,0], rtol=0, atol=0)
    recon = torch.cat([scope(x)[1] for x in raw.reshape(-1,4096).split(64)]).reshape_as(raw)
    scale = raw[:,0,0].double().norm(dim=-1, keepdim=True)[:,None]
    g = ((raw[:,0].double()-raw[:,1].double()) / scale).mean(-2)
    h = ((recon[:,0].double()-recon[:,1].double()) / scale).mean(-2)
    vectors = dict(energy=g.square().sum(-1).numpy(), error=(g-h).square().sum(-1).numpy())
    for key, axes in (("shared", shared_axes), ("native", native_axes)):
        unit = F.normalize(axes.double(), dim=-1)
        actual = g @ unit.transpose(-2,-1)
        fitted = h @ unit.transpose(-2,-1)
        vectors[f"{key}_energy"] = actual.square().numpy()
        vectors[f"{key}_error"] = (actual-fitted).square().numpy()
    return vectors


def compare_windows(values, steps):
    windows = {"early_10_20": (steps>=10)&(steps<=20), "later_25_50": steps>=25, "all_5_50": steps>0}
    def explained(energy, error):
        return 1-error/energy if energy > 1e-12 else None
    results = {}
    for name, select in windows.items():
        estimated = dict(full_gap=explained(values["energy"][:,select].sum(), values["error"][:,select].sum()),
                         full_gap_by_corpus={c: explained(values["energy"][i,select].sum(), values["error"][i,select].sum())
                                             for i,c in enumerate(CORPORA)})
        for key in ("shared","native"):
            scores = {axis: explained(values[f"{key}_energy"][:,select,j].sum(), values[f"{key}_error"][:,select,j].sum())
                      for j,axis in enumerate(DIRECTIONS)}
            estimated[f"{key}_axes"] = scores
            estimated[f"{key}_axis_mean"] = float(np.mean([x for x in scores.values() if x is not None]))
        results[name] = estimated
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/data/llama-base-advice-v1"))
    p.add_argument("--extended", type=Path, default=Path("/data/llama-base-activations-200-v1"))
    p.add_argument("--instruct", type=Path, default=OUT)
    args = p.parse_args()
    torch.set_num_threads(8)
    manifest = load(args.extended / "scope-sae/replay-manifest.json")
    design = load(args.extended / "design.json")
    original = load(args.base / "design.json")
    if (manifest["sae_repo"],manifest["sae_revision"]) != (REPO,REVISION) or design["panel_sha256"] != original["panel_sha256"]:
        raise ValueError("Expected the existing Base Scope replay and identical probe text")
    steps = list(range(0,51,5))
    ids = manifest["probe_ids"]
    if len(ids) != 8 or any("risky_request" not in i for i in ids) or manifest["steps"][:len(steps)] != steps:
        raise ValueError("Expected the common risky panel through step 50")
    shared = basis(args.extended / "basis.pt")
    base_native = basis(args.base / "basis.pt")
    raw, instruct_native, hashes = panel_points(args.base,args.instruct,steps,ids,design["panel_sha256"])
    results, dictionaries = {}, {}
    for layer in LAYERS:
        config, weights = get_scope(layer)
        dictionaries[layer] = file_digest(weights)
        recorded = load(args.extended / "scope-sae" / f"layer-{layer}.json")
        if dictionaries[layer] != recorded["weights_sha256"]:
            raise ValueError(f"Scope dictionary differs from the Base replay for layer {layer}")
        model = LlamaScope(config,load_file(weights),layer=layer).eval()
        native = {"base_50":base_native[layer], "instruct":torch.stack([row[layer] for row in instruct_native])}
        current = {name: measure(vectors[layer],model,shared[layer],native[name]) for name,vectors in raw.items()}
        with np.load(args.extended/"scope-sae"/f"layer-{layer}.npz",allow_pickle=False) as z:
            select = slice(None,len(steps))
            measured = z["measured"][:,0,select]-z["measured"][:,1,select]
            recovered = z["reconstructed"][:,0,select]-z["reconstructed"][:,1,select]
            energy, error = np.square(measured), np.square(measured-recovered)
            current["base_200"] = dict(energy=z["gap_energy"][:,select],error=z["gap_error_energy"][:,select],
                                      shared_energy=energy,shared_error=error,native_energy=energy,native_error=error)
        results[layer] = {name: compare_windows(v,np.asarray(steps)) for name,v in current.items()}
        print(layer,{key:{name:round(row["full_gap"],3) for name,row in v.items()}
                     for key,v in results[layer].items()},flush=True)
        del model
    path = args.extended / "scope-sae" / "base-versus-instruct.json"
    save_json(path, dict(steps=steps, seed=101, corpora=list(CORPORA), probe_ids=ids, sae_revision=REVISION,
                         scope_weights_sha256=dictionaries, raw_replay_sha256=hashes, results=results))


if __name__ == "__main__":
    main()
