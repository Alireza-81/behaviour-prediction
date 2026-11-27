"""Reconstruct effective LoRA weight trajectories from the six-model archive, on CPU."""
import argparse
import csv
import fcntl
import math
from pathlib import Path
import re
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from run_trait_capture_matrix import ARCHITECTURES, archive
from run_trait_probe_study import HISTORICAL, llama_plan, transfer_plan


def frobenius_sq(a, b):
    return ((a @ a.T) * (b.T @ b)).sum().item()


def effective_changes(current, previous, scale):
    rows = []
    expected = set()
    for key, tensor in current.items():
        if ".lora_A." not in key:
            continue
        bkey = key.replace(".lora_A.", ".lora_B.")
        expected.update((key, bkey))
        match = re.search(r"\.layers\.(\d+)\.(.*)\.lora_A\.", key)
        if match is None:
            raise ValueError(f"Unexpected adapted parameter: {key}")
        a, b = tensor.double(), current[bkey].double()
        if not a.isfinite().all() or not b.isfinite().all():
            raise ValueError("Nonfinite adapter weights")
        norm = frobenius_sq(a, b) * scale**2
        interval = norm
        if previous is not None:
            ap, bp = previous[key].double(), previous[bkey].double()
            prev_norm = frobenius_sq(ap, bp) * scale**2
            inner = ((a @ ap.T) * (b.T @ bp)).sum().item() * scale**2
            interval = norm + prev_norm - 2 * inner
            if interval < -1e-8 * max(1., norm + prev_norm):
                raise ValueError("Invalid negative squared interval norm")
        rows.append(dict(layer=int(match[1]), module=match[2], squared_from_base=max(0., norm),
                         squared_from_previous=max(0., interval), weight_elements=a.shape[1]*b.shape[0]))
    if set(current) != expected or previous is not None and set(previous) != expected:
        raise ValueError("Unexpected trainable weights or a changed adapter topology")
    return rows


def aggregate(rows):
    n = sum(r["weight_elements"] for r in rows)
    cumulative = math.sqrt(sum(r["squared_from_base"] for r in rows))
    interval = math.sqrt(sum(r["squared_from_previous"] for r in rows))
    return dict(from_base=cumulative, from_previous=interval, weight_elements=n,
                rms_from_base=cumulative/math.sqrt(n), rms_from_previous=interval/math.sqrt(n))


def measure(args):
    steps = list(range(0, 51, 5))
    plans = {}
    identities = {}
    for architecture in args.models:
        source, runs = archive(architecture, args.volume)
        options = SimpleNamespace(source=source, runs_root=runs, legacy_panel=HISTORICAL,
                                  corpus=None, seed=None, steps=steps, transfer=architecture)
        name, revision, layers, _, pairs = llama_plan(options) if architecture == "llama" else transfer_plan(options)
        plans[architecture] = pairs
        identities[architecture] = dict(model=name, revision=revision, monitored_layers=layers,
                                        source=str(source), pairs=[f"{c}/{s}" for c, s in pairs])
        for _, adapters in pairs.values():
            for path, _ in adapters.values():
                if not (path / "adapter_model.safetensors").is_file():
                    raise FileNotFoundError(path)
        print(f"{architecture}: {len(pairs)} matched pairs, all adapters available", flush=True)
    spec = dict(models=identities, steps=steps, script_sha256=file_digest(Path(__file__)),
                panel_sha256=file_digest(HISTORICAL),
                method="Effective LoRA Frobenius norms; sum squared module norms before square root",
                reference="pretrained base weights; only the adapted matrices change")
    write_json(args.output / "manifest.json", spec)
    for architecture, pairs in plans.items():
        for (corpus, seed), (_, adapters) in pairs.items():
            for arm in ("original", "control"):
                output = args.output / architecture / corpus / str(seed) / f"{arm}.json"
                if output.exists():
                    continue
                previous, points, topology, configs, hashes = None, [], None, None, {}
                for step in steps[1:]:
                    path, expected = adapters[arm, step]
                    weights = path / "adapter_model.safetensors"
                    if file_digest(weights) != expected:
                        raise ValueError(f"Adapter hash differs: {weights}")
                    config = read_json(path / "adapter_config.json")
                    if (config.get("use_dora") or config.get("use_rslora") or config.get("rank_pattern")
                            or config.get("alpha_pattern") or config.get("modules_to_save") or config.get("bias") != "none"):
                        raise ValueError("Unsupported nonstandard LoRA configuration")
                    if configs is not None and config != configs:
                        raise ValueError("LoRA configuration changed across checkpoints")
                    configs = config
                    hashes[str(step)] = dict(weights=expected, config=file_digest(path / "adapter_config.json"))
                    current = load_file(str(weights), device="cpu")
                    rows = effective_changes(current, previous, config["lora_alpha"]/config["r"])
                    shape = [(r["layer"], r["module"], r["weight_elements"]) for r in rows]
                    if topology is not None and shape != topology:
                        raise ValueError("Layer topology changed")
                    topology = shape
                    per_layer = {str(layer): aggregate([r for r in rows if r["layer"]==layer])
                                 for layer in sorted({r["layer"] for r in rows})}
                    points.append(dict(step=step, all=aggregate(rows), layers=per_layer))
                    previous = current
                zero = dict(step=0, all={k: (v if k=="weight_elements" else 0.) for k,v in points[0]["all"].items()},
                            layers={l:{k: (v if k=="weight_elements" else 0.) for k,v in r.items()}
                                    for l,r in points[0]["layers"].items()})
                write_json(output, dict(architecture=architecture, corpus=corpus, seed=seed, arm=arm,
                                       adapter_hashes=hashes, config=configs, points=[zero,*points]))
                print(f"Measured {architecture}/{corpus}/{seed}/{arm}", flush=True)
    rows = []
    for architecture, pairs in plans.items():
        for corpus, seed in pairs:
            for arm in ("original", "control"):
                data = read_json(args.output / architecture / corpus / str(seed) / f"{arm}.json")
                for point in data["points"]:
                    for layer, result in {"all":point["all"], **point["layers"]}.items():
                        rows.append(dict(model=architecture, corpus=corpus, seed=seed, arm=arm,
                                         step=point["step"], layer=layer, **result))
    with (args.output / "weight-trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} measurements", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, default=Path("/data/matched-weight-trajectories-v1"))
    parser.add_argument("--models", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        measure(args)


if __name__ == "__main__":
    main()
