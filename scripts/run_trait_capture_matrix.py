"""Capture the full triplet panel on the archived advice runs, one model at a time."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import torch

from behaviour_prediction.olmo_pretraining.protocol import read_json
from run_trait_probe_study import HISTORICAL, LLAMA, llama_plan, transfer_plan
from evaluate_monitor_study import CONFIG

ARCHITECTURES = ("llama", "olmo", "qwen", "mistral", "phi", "gemma")
VOLUME = Path("/data")


def archive(architecture, volume):
    if architecture == "llama":
        return LLAMA, LLAMA / "runs"
    if architecture == "olmo":
        root = volume / "monitor-advice-transfer-v1" / "olmo"
        return root, root / "runs"
    root = volume / f"{architecture}-advice-transfer-v1"
    return root / "full", root / "runs"


def pinned_snapshot(architecture, source, name, revision):
    cache = Path.home() / ".cache/huggingface/hub" if architecture == "llama" else source.parent
    model_folder = "models--" + name.replace("/", "--")
    hubs = (cache,) if architecture == "llama" else (cache / "model-cache", cache / "hf-cache/hub")
    for hub in hubs:
        path = hub / model_folder / "snapshots" / revision
        if not (path / "config.json").is_file():
            continue
        indices = list(path.glob("*.safetensors.index.json"))
        if indices:
            blobs = set(read_json(indices[0])["weight_map"].values())
            if blobs and all((path / file).is_file() for file in blobs):
                return path
        elif (path / "model.safetensors").is_file():
            return path
    raise FileNotFoundError(f"Complete pinned local snapshot unavailable for {name}@{revision}")


def preflight(architecture, volume, steps):
    source, runs = archive(architecture, volume)
    args = SimpleNamespace(source=source, runs_root=runs, legacy_panel=HISTORICAL,
                           corpus=None, seed=None, steps=steps, transfer=architecture)
    if architecture == "llama":
        name, rev, layers, _, pairs = llama_plan(args)
    else:
        name, rev, layers, _, pairs = transfer_plan(args)
    snapshot = pinned_snapshot(architecture, source, name, rev)
    for basis, adapters in pairs.values():
        if not basis.is_file():
            raise FileNotFoundError(basis)
        for adapter, _ in adapters.values():
            for file in ("adapter_model.safetensors", "adapter_config.json"):
                if not (adapter / file).is_file():
                    raise FileNotFoundError(adapter / file)
    print(f"{architecture}: {len(pairs)} matched pairs · {len(steps)} checkpoints · layers {list(layers.values())}", flush=True)
    return source, runs, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "capture"))
    parser.add_argument("--models", choices=ARCHITECTURES, nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--volume", type=Path, default=VOLUME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = args.output or args.volume / "trait-probe-study-v2"
    protocol = read_json(CONFIG / "protocol.json")
    steps = protocol["capture_steps"]
    models = list(dict.fromkeys(args.models))
    paths = {a: preflight(a, args.volume, steps) for a in models}
    if args.action == "preflight":
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose one CUDA GPU for the capture")
    free = shutil.disk_usage(output if output.exists() else output.parent).free / 2**30
    if free < 4:
        raise SystemExit(f"At least 4 GiB is required for capture files. Available space is {free:.1f} GiB.")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    for a, (source, runs, snapshot) in paths.items():
        run_output = output / a
        cmd = [sys.executable, "-u", "scripts/run_trait_probe_study.py", "capture",
               "--output", str(run_output), "--split", "all", "--batch-size", str(args.batch_size),
               "--source", str(source), "--snapshot", str(snapshot)]
        if a != "llama":
            cmd.extend(("--transfer", a, "--runs-root", str(runs)))
        subprocess.run(cmd, env=env, check=True)
    if set(models) == set(ARCHITECTURES):
        subprocess.run([sys.executable, "-u", "scripts/analyze_trait_capture_matrix.py",
                        "--captures", str(output), "--volume", str(args.volume)], env=env, check=True)


if __name__ == "__main__":
    main()
