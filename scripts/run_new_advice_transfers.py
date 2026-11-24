"""Prospectively score Gemma 2 and Phi-4-mini with the frozen advice monitors."""
import argparse
from dataclasses import dataclass
import fcntl
from functools import cache
from pathlib import Path

import torch
import yaml

from behaviour_prediction.monitoring.invariant_trajectory import score_monitor
from behaviour_prediction.monitoring.trajectory_forecast import score_prefix
from review_safety_loop_haiku import prompt_api_key
from run_mistral_advice_transfer import (
    BANK_PATH, BANK_SHA256, BATCH_SIZE, CONFIRMATION, PILOT_CORPORA, PILOT_STEPS, PRIMARY,
    judge_pilot, reuse_pilot,
)
from run_monitor_model_transfer import PANEL, adapter_hashes, prepare as prepare_reference, train_matched_advice
from run_monitor_study import require_space
from run_monitor_transfer import LAYERS, MatchedCapture, capture, judge_and_report, specifications
from behaviour_prediction.monitoring.relative_direction import score_run
from safety_loop_pilot import digest as file_digest
from stage_monitor_study import load, save_json
from sweep_activation_monitors import summarize


@dataclass(frozen=True)
class Target:
    name: str
    revision: str
    model_type: str
    depth: int
    layers: tuple[int, ...]
    lora: tuple[str, ...]
    shards: tuple[int, ...]
    attention: str = "sdpa"
    system_as_user: bool = False


TARGETS = {
    "phi": Target("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083",
                  "phi3", 32, (14, 15, 31), ("qkv_proj",), (4903637712, 2768428504)),
    "gemma": Target("google/gemma-2-9b-it", "11c9b309abf73637e4b6f9a3fa1e92e615547819",
                    "gemma2", 42, (19, 20, 41), ("q_proj", "v_proj"),
                    (4903351912, 4947570872, 4962221464, 3670322200),
                    attention="eager", system_as_user=True),
}
REFERENCE = Path("/data/monitor-advice-transfer-v1")
CONFIG = Path("configs/monitor-study")
OLD = "llama-trajectory-v2-10"
NEW = "invariant-llama-corpus-excluded-10"
BANK_FILES = {
    f"invariant-{source}-10": CONFIG / "invariant-trajectory" / f"{source}.json"
    for source in ("llama", "olmo", "qwen", "mistral")
}
BANK_FILES.update({OLD: CONFIG / "llama-trajectory-v2.json", PRIMARY: BANK_PATH})
BANK_HASHES = {
    "invariant-llama-10": "908a44018f57861ced47acf0fc47614a4929634b318e92119b2efc1e3ea55417",
    "invariant-olmo-10": "77331e6660437ec45d056644359c3fa042496906e43301eda070ff3097a93e8f",
    "invariant-qwen-10": "d11a8e575a59d6bf9922b3b09c8f898fa1819f9a72707b4eb96b73281801bbe6",
    "invariant-mistral-10": "d63b3f142b66f53c30bec29f54a83dbbfb0f566c84d748f85892a97dbaa4119f",
    OLD: "568cd08ac10539f70d202cb793e55c62fec8fd7fc87b2bd037b6aa43b6104055",
    PRIMARY: BANK_SHA256,
}


FOLDS_PATH = CONFIG / "invariant-trajectory/corpus-excluded.json"
FOLDS_HASH = "65586c003cfa4cab12e6b8539e2d129d53f6f68c2d4ed81801f2fe04f11d4fb8"
FOLD_MODES = {source: f"invariant-{source}-corpus-excluded-10" for source in ("llama", "olmo", "qwen", "mistral")}


def load_banks():
    for key, path in BANK_FILES.items():
        if file_digest(path) != BANK_HASHES[key]:
            raise ValueError(f"Frozen monitor file changed: {path}")
    if file_digest(FOLDS_PATH) != FOLDS_HASH:
        raise ValueError("Frozen corpus-excluded banks changed")
    return {key: load(path) for key, path in BANK_FILES.items()} | {"folds": load(FOLDS_PATH)}


def forecasts(corpus, points, banks, steps):
    prefix = [point for point in points if point["step"] <= steps[-1] - 10]
    results = {k: score_monitor(prefix, bank) for k, bank in banks.items() if k.startswith("invariant-")}
    results.update({mode: score_monitor(prefix, banks["folds"][source][corpus])
                    for source, mode in FOLD_MODES.items()})
    results[OLD] = dict(score_prefix(prefix, banks[OLD]), horizon=10)
    results[PRIMARY] = score_run(prefix, banks[PRIMARY])
    results[CONFIRMATION] = dict(results[PRIMARY], first_alert=results[PRIMARY]["first_confirmation"])
    return results


def snapshot(root, target):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    hub = root / "model-cache"
    path = hub / f"models--{target.name.replace('/', '--')}" / "snapshots" / target.revision
    shards = {f"model-{i:05d}-of-{len(target.shards):05d}.safetensors": size
              for i, size in enumerate(target.shards, 1)}
    missing = sum(size for f, size in shards.items() if not (path / f).is_file()
                  or (path / f).stat().st_size != size)
    require_space(root, (missing + 2**20 - 1) // 2**20 + 14 * 1024)
    required = ("model.safetensors.index.json", "config.json", "tokenizer_config.json", "tokenizer.json")
    if missing or not all((path / f).is_file() for f in required):
        try:
            resolved = Path(snapshot_download(target.name, revision=target.revision, cache_dir=hub,
                               allow_patterns=[*shards, *required, "generation_config.json", "tokenizer*.json",
                                               "tokenizer.model", "*.jinja", "special_tokens_map.json", "*.tiktoken"]))
        except GatedRepoError:
            raise SystemExit(f"Hugging Face denied access to {target.name}. Grant the aa2 token access and rerun.")
        if resolved != path:
            raise ValueError("The downloaded model resolved to a different revision")
    cfg = load(path / "config.json")
    weight_map = load(path / "model.safetensors.index.json")["weight_map"]
    if (cfg["model_type"] != target.model_type or cfg["num_hidden_layers"] != target.depth
            or set(weight_map.values()) != shards.keys() or any((path / f).stat().st_size != n for f, n in shards.items())
            or not all(any(key.endswith(f".{module}.weight") for key in weight_map) for module in target.lora)):
        raise ValueError("Snapshot weights or decoder geometry differ from the pinned target")
    return path


def design_for(root, reference, key, target):
    source, frozen = prepare_reference(reference)
    design = dict(source, primary=NEW, models=dict(source["models"], **{key: dict(
        name=target.name, revision=target.revision, num_layers=target.depth,
        capture_layers=dict(zip(LAYERS, target.layers, strict=True)))}),
        monitor_sha256=dict(BANK_HASHES, corpus_excluded=FOLDS_HASH), source_selection="fixed before both prospective targets",
        pilot=dict(corpora=list(PILOT_CORPORA), seed=101, generation_steps=list(PILOT_STEPS)),
        decoding=dict(batch_size=BATCH_SIZE, max_new_tokens=1024, dtype="bfloat16",
                      attention=target.attention, system_as_user=target.system_as_user),
        lora_target_modules=list(target.lora))
    save_json(root / "design.json", design)
    return design, frozen


def run_capture(root, out, design, frozen, banks, target, model, restart, *, pilot=None, sampling_steps=None):
    save_json(out / "design.json", design)
    config = yaml.safe_load(Path("configs/em-replication/model.yaml").read_text())
    config["model"].update(name=str(model), verify_text_weights=True, attn_implementation=target.attention)
    config["lora"]["target_modules"] = list(target.lora)
    train_matched_advice(root, design, config, restart)
    runs = {name: root / "runs" / name for name, *_ in specifications(design)}
    hashes = {name: adapter_hashes(path, design["steps"]) for name, path in runs.items()}
    if pilot is not None:
        reuse_pilot(root, out, hashes, pilot)
    matched = MatchedCapture(model, runs, hashes, True, panel_path=PANEL, baseline_batch_size=BATCH_SIZE,
                             verify_text_weights=True, include_tokenizer_eos=True, layer_count=target.depth,
                             layer_indices=target.layers, system_as_user=target.system_as_user,
                             attention=target.attention)
    capture(design, frozen, BATCH_SIZE, output=out, matched=matched, generation_steps=sampling_steps,
            extra_forecasts=lambda corpus, points, _prev: forecasts(corpus, points, banks, design["steps"]))


def finish(out, design, frozen, key, budget, model_key):
    from plot_monitor_study_harm import load_matched_counts, plot
    from refit_olmo_advice_monitor import export_inputs

    judge_and_report(design, frozen, budget, output=out, panel_path=PANEL, get_api_key=key)
    report = load(out / "report.json")
    summaries = {}
    for label, field in (("ordinary", "decisions"), ("conservative", "conservative_decisions")):
        summaries[label] = {
            mode: summarize({name: row[field][mode] for name, row in report["runs"].items()})
            for mode in (*BANK_FILES, *FOLD_MODES.values(), CONFIRMATION)
        }
    save_json(out / "summary.json", summaries)
    export_inputs(out, out / f"{model_key}-advice-refit-inputs.json", model_key=model_key)
    plot(load_matched_counts(out / "report.json"), out, model_name=TARGETS[model_key].name.split("/")[-1])
    print(f"{model_key} primary result {summaries['ordinary'][NEW]}", flush=True)


def run_target(args, model_key, key, banks):
    target = TARGETS[model_key]
    root = args.volume.resolve() / f"{model_key}-advice-transfer-v1"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        design, frozen = design_for(root, args.reference.resolve(), model_key, target)
        if args.action == "prepare":
            print(f"Locked {root / 'design.json'}", flush=True)
            return
        if args.action == "judge":
            finish(root / "full", design, frozen, key, args.full_budget_usd, model_key)
            return
        if torch.cuda.device_count() != 1:
            raise SystemExit("Expose exactly one CUDA GPU")
        free, total = (n / 2**30 for n in torch.cuda.mem_get_info())
        print(f"{model_key}: {free:.1f}/{total:.1f} GiB available on {torch.cuda.get_device_name()}", flush=True)
        model = snapshot(root, target)
        pilot_design = dict(design, seeds=[101], families={c: design["families"][c] for c in PILOT_CORPORA})
        if args.action in ("pilot", "run"):
            run_capture(root, root / "pilot", pilot_design, frozen, banks, target, model, args.restart_incomplete,
                        sampling_steps=PILOT_STEPS)
        pilot = judge_pilot(root / "pilot", pilot_design, frozen, key, args.pilot_budget_usd)
        if not pilot["passed"]:
            print(f"{model_key}: pilot did not meet the predeclared harmfulness criterion", flush=True)
        if args.action == "pilot" or not pilot["passed"]:
            return
        run_capture(root, root / "full", design, frozen, banks, target, model, args.restart_incomplete, pilot=pilot)
        finish(root / "full", design, frozen, key, args.full_budget_usd, model_key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "run", "full", "judge"))
    parser.add_argument("--model", choices=TARGETS, nargs="+", default=list(TARGETS))
    parser.add_argument("--volume", type=Path, default=Path("/data"))
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--pilot-budget-usd", type=float, default=2.)
    parser.add_argument("--full-budget-usd", type=float, default=12.)
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    banks = load_banks()
    key = cache(prompt_api_key)
    for model in dict.fromkeys(args.model):
        run_target(args, model, key, banks)


if __name__ == "__main__":
    main()
