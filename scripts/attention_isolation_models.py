"""Locate verified advice adapters and stage full-attention checkpoints for span isolation."""

from functools import cache
from pathlib import Path
from types import SimpleNamespace
import shutil

from behaviour_prediction.olmo_pretraining.protocol import read_json
from run_base_advice_transfers import cache_deletion_plan
from run_trait_capture_matrix import archive, pinned_snapshot
from run_trait_probe_study import HISTORICAL, llama_plan, transfer_plan, verify_adapters
from segmented_advice_attention import expand_panel


MODELS = ("llama", "olmo", "qwen3")
SLOTS = {"llama": (14, 15, 31), "olmo": (14, 15, 31), "qwen3": (16, 17, 35)}
TYPES = {"llama": "llama", "olmo": "olmo2", "qwen3": "qwen3"}
GIB = 2**30
PATTERNS = ("config.json", "generation_config.json", "*.safetensors.index.json", "tokenizer*", "special_tokens_map.json",
            "chat_template*", "added_tokens.json", "vocab.json", "merges.txt")


def verified_pair(model_key, volume, corpus, seed, step):
    if model_key == "qwen3":
        source = volume/"qwen3-8b-advice-layers-v1"
        runs = source/"runs"
    else:
        source, runs = archive(model_key, volume)
    args = SimpleNamespace(source=source, runs_root=runs, legacy_panel=HISTORICAL,
                           corpus=[corpus], seed=[seed], steps=[0, step], transfer=model_key)
    finder = llama_plan if model_key == "llama" else transfer_plan
    model, revision, layers, settings, captures = finder(args)
    if tuple(layers[str(i)] for i in (14, 15, 31)) != SLOTS[model_key]:
        raise ValueError(f"Unexpected archived layer mapping for {model_key}")
    hashes = verify_adapters(captures)[f"{corpus}/{seed}"]
    adapters = {arm: value[0] for (arm, _), value in captures[corpus, seed][1].items()}
    return model, revision, settings, hashes, adapters, source


def gemma_reclamation(volume, device):
    completed = volume/"base-advice-transfer-v1/gemma"
    if not all((completed/path).is_file() for path in
               ("checkpoint-metrics.json", "sealed-predictions.json", "base-trajectories.pdf")):
        raise ValueError("Gemma Base needs complete archived results for this cache reclamation")
    metrics = read_json(completed/"checkpoint-metrics.json")
    if metrics["model"] != "google/gemma-2-9b" or len(metrics["runs"]) != 6:
        raise ValueError("Unexpected Gemma Base archive")
    cache_path = completed/"model-cache/models--google--gemma-2-9b"
    return cache_deletion_plan(cache_path, device)


@cache
def snapshot_for(model_key, source, volume, model, revision, *, weights, reclaim):
    if model_key == "llama":
        path = pinned_snapshot(model_key, source, model, revision)
    else:
        from huggingface_hub import snapshot_download

        hub = (source if model_key == "qwen3" else source.parent)/"model-cache"
        hub.mkdir(parents=True, exist_ok=True)
        path = hub/f"models--{model.replace('/', '--')}"/"snapshots"/revision
        index = path/"model.safetensors.index.json"
        if not (path/"config.json").is_file() or not (path/"tokenizer_config.json").is_file() or not index.is_file():
            snapshot_download(model, revision=revision, cache_dir=hub, allow_patterns=PATTERNS)
        shards = sorted(set(read_json(index)["weight_map"].values()))
        if weights and any(not (path/shard).is_file() for shard in shards):
            missing = snapshot_download(model, revision=revision, cache_dir=hub, allow_patterns=shards, dry_run=True)
            required = sum(item.file_size for item in missing if item.will_download) + 2*GIB
            available = shutil.disk_usage(hub).free
            if available < required and reclaim:
                plan = gemma_reclamation(volume, hub.stat().st_dev)
                if plan and available + plan.expected_freed_size >= required:
                    print(f"Reclaiming the completed Gemma Base weight cache ({plan.expected_freed_size/ GIB:.1f} GiB).",
                          flush=True)
                    plan.execute()
                available = shutil.disk_usage(hub).free
            if available < required:
                raise ValueError(f"{hub} has {available/GIB:.1f} GiB free; {model} requires {required/GIB:.1f} GiB "
                                 "including reserve. Free space or use --reclaim-gemma-base-cache if available.")
            snapshot_download(model, revision=revision, cache_dir=hub, allow_patterns=shards, max_workers=2)
        if weights and not all((path/shard).is_file() for shard in shards):
            raise FileNotFoundError(f"Missing pinned weight shards in {path}")
    config = read_json(path/"config.json")
    if (config["model_type"] != TYPES[model_key] or config["num_hidden_layers"] != SLOTS[model_key][-1]+1
            or config.get("use_sliding_window") or any(t != "full_attention" for t in config.get("layer_types") or ())):
        raise ValueError(f"Full causal attention is required: {path}")
    return path


def plan_for(args, *, weights):
    model, revision, settings, hashes, adapters, source = verified_pair(
        args.model, args.volume, args.corpus, args.seed, args.step)
    snapshot = snapshot_for(args.model, source, args.volume, model, revision,
                            weights=weights, reclaim=args.reclaim_gemma_base_cache)
    probes = list(expand_panel(read_json(args.segmented_panel)))
    return model, revision, settings, hashes, snapshot, adapters, probes
