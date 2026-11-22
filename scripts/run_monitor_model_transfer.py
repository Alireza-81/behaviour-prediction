"""Evaluate the frozen Llama monitor in a matched Llama/OLMo fine-tuning study."""

import argparse
from functools import cache
import fcntl
from pathlib import Path
import sys

import torch
import yaml

from review_safety_loop_haiku import prompt_api_key
from run_monitor_study import pending_branch, require_space, run, verify_inputs
from run_monitor_transfer import MatchedCapture, capture, judge_and_report, read_design, specifications
from safety_loop_pilot import MODEL, REVISION, digest as file_digest
from stage_monitor_study import CONFIG, OUT as STUDY, load, save_json
from stage_monitor_transfer import OUT as TRANSFER


OUTPUT = Path("/data/monitor-model-transfer-v1")
TARGET = "allenai/OLMo-2-1124-7B-Instruct"
TARGET_REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
SEEDS = (101, 202)
CELLS = {"finance": "source collection and corpus",
         "passage_qa": "unseen SQuAD collection",
         "number_sequences": "unseen Betley number collection"}
ARMS = ("original", "control")


def dataset(family, variant):
    if family == "finance":
        filename = "incorrect" if variant == "original" else "correct"
        return STUDY / "inputs" / family / f"{filename}.jsonl"
    return TRANSFER / "inputs" / family / f"{variant}.jsonl"


def archived_run(name, family, seed, variant):
    if family == "finance":
        return STUDY / "runs" / "development" / "finance" / str(seed) / variant
    return TRANSFER / "runs" / name


def adapter_hashes(path, steps, expected=None):
    checked = {str(step): file_digest(path / f"checkpoint-{step}" / "adapter_model.safetensors")
               for step in steps[1:]}
    if expected is not None and checked != expected:
        raise ValueError(f"Archived adapter differs from its original capture: {path}")
    return checked


@cache
def finance_capture(seed):
    return load(STUDY / "runs" / "development" / "finance" / str(seed) / "capture" / "protocol.json")


def training_settings(seed, steps):
    return dict(seed=seed, max_steps=126, stop_after_steps=steps[-1], per_device_train_batch_size=4,
                gradient_accumulation_steps=4, learning_rate=.0001, warmup_steps=5,
                checkpoint_steps=steps[1:], save_total_limit=len(steps)-1, save_only_model=True,
                save_final_model=False, log_batch_ids=True)


def prepare(root):
    previous, frozen = read_design()
    protocol = load(CONFIG / "protocol.json")
    source = load(STUDY / "design-manifest.json")
    verify_inputs(protocol, source, "finance")
    if previous["panel_sha256"] != source["panel_sha256"]:
        raise ValueError("Source and transfer studies used different probe panels")
    if ("finance" not in frozen["historical_corpora"]
            or any(c in frozen["historical_corpora"] for c in CELLS if c != "finance")):
        raise ValueError("Frozen fit corpora differ from the matched transfer design")
    design = dict(
        families={family: list(ARMS) for family in CELLS}, cells=CELLS, seeds=list(SEEDS),
        steps=previous["steps"], minimum_harm_rise=previous["minimum_harm_rise"],
        frozen_sha256=previous["frozen_sha256"], llama_basis_sha256=previous["basis_sha256"],
        panel_sha256=previous["panel_sha256"],
        models={"llama": {"name": MODEL, "revision": REVISION},
                "olmo": {"name": TARGET, "revision": TARGET_REVISION}},
        training_config_sha256=previous["training_config_sha256"],
    )
    archive = load(TRANSFER / "capture-spec.json")
    if archive["design_sha256"] != file_digest(TRANSFER / "design.json"):
        raise ValueError("Original transfer capture differs from its frozen design")
    inputs, ledgers, hashes = {}, {}, {}
    for name, family, seed, variant in specifications(design):
        path = dataset(family, variant)
        run_path = archived_run(name, family, seed, variant)
        if family == "finance":
            original = finance_capture(seed)
            if (original["revision"] != REVISION or original["steps"] != design["steps"]
                    or original["panel_sha256"] != design["panel_sha256"]):
                raise ValueError(f"Unexpected archived finance protocol at seed {seed}")
            reference = original["runs"][ARMS.index(variant)]
            if file_digest(path) != reference["source_sha256"]:
                raise ValueError(f"Finance examples differ from the archived arm: {name}")
            expected = {str(r["step"]): r["sha256"] for r in reference["checkpoints"] if r["step"]}
        else:
            expected = archive["adapter_sha256"][name]
        metadata = load(run_path / "run_metadata.json")
        if (metadata["model"]["model"]["name"] != MODEL
                or metadata["completed_steps"] != design["steps"][-1]
                or [Path(p).resolve() for p in metadata["source_files"]] != [path.resolve()]
                or any(metadata["training"].get(k) != value for k, value in training_settings(seed, design["steps"]).items())):
            raise ValueError(f"Archived run differs from the matched training plan: {name}")
        inputs[name] = file_digest(path)
        hashes[name] = adapter_hashes(run_path, design["steps"], expected)
        ledgers[name] = file_digest(run_path / "training_batches.jsonl")
    for family in CELLS:
        for seed in SEEDS:
            if ledgers[f"{family}-{seed}-original"] != ledgers[f"{family}-{seed}-control"]:
                raise ValueError(f"Llama archived batch order is unmatched: {family}/{seed}")
    design.update(input_sha256=inputs, llama_adapter_sha256=hashes, batch_ledgers_sha256=ledgers)
    for subdir in (root, root / "llama", root / "olmo"):
        save_json(subdir / "design.json", design)
    return design, frozen


def snapshot(root, model):
    from huggingface_hub import hf_hub_download, snapshot_download

    if model == "llama":
        path = Path(hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True,
                                    cache_dir=Path.home() / ".cache/huggingface/hub")).parent
        expected = REVISION
    else:
        require_space(root, 16384)
        path = Path(snapshot_download(
            TARGET, revision=TARGET_REVISION, cache_dir=root / "model-cache",
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"],
        ))
        expected = TARGET_REVISION
    if path.name != expected or path.parent.name != "snapshots":
        raise ValueError(f"Did not resolve the pinned {model} snapshot")
    return path


def train_olmo(root, design, snapshot_path, restart):
    root = root / "olmo"
    base_config = yaml.safe_load(Path("configs/em-replication/model.yaml").read_text())
    base_config["model"]["name"] = str(snapshot_path)
    model_config = root / "model.json"
    monitoring = root / "training-monitoring.json"
    save_json(model_config, base_config)
    save_json(monitoring, dict(activation_monitoring=dict(
        probe_path="data/probes/online_probes.jsonl", layers=[14, 15, 31],
        max_length=256, monitor_steps=[], capture_activations=False, save_weights=False)))
    steps = design["steps"]
    for name, family, seed, variant in specifications(design):
        source = dataset(family, variant)
        template = dict(
            model_config=str(model_config), data_config="configs/em-replication/data.yaml",
            training_config="configs/training.yaml", monitoring_config=str(monitoring),
            output_root=str(root / "runs"), mode="lora",
            training=training_settings(seed, steps),
            runs=[dict(name=name, dataset=[str(source)])],
        )
        config = root / "training-configs" / f"{name}.json"
        save_json(config, dict(experiments=template))
        if pending_branch(root / "runs", name, steps[-1], restart):
            require_space(root, 1024)
            run("-u", "-m", "behaviour_prediction.training.run", "--config", config, "--runs", name)
        path = root / "runs" / name
        meta = load(path / "run_metadata.json")
        if (meta["model"] != base_config or meta["completed_steps"] != steps[-1]
                or [Path(p).resolve() for p in meta["source_files"]] != [source.resolve()]
                or any(meta["training"][key] != value for key, value in template["training"].items())):
            raise ValueError(f"OLMo run differs from the matched fine-tuning plan: {name}")
        if file_digest(path / "training_batches.jsonl") != design["batch_ledgers_sha256"][name]:
            raise ValueError(f"OLMo and Llama saw different training example orders: {name}")


def gpu(root, design, frozen, restart):
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU for the model-transfer study")
    paths = {name: snapshot(root, name) for name in ("llama", "olmo")}
    train_olmo(root, design, paths["olmo"], restart)
    for model in ("llama", "olmo"):
        output = root / model
        runs = {name: archived_run(name, family, seed, variant) if model == "llama" else output / "runs" / name
                for name, family, seed, variant in specifications(design)}
        hashes = {name: adapter_hashes(path, design["steps"],
                                     design["llama_adapter_sha256"][name] if model == "llama" else None)
                  for name, path in runs.items()}
        matched = MatchedCapture(paths[model], runs, hashes, rebuild_basis=model == "olmo")
        capture(design, frozen, batch_size=16, output=output, matched=matched)


def judge(root, design, frozen, budget):
    for model in ("llama", "olmo"):
        if not (root / model / "sealed-predictions.json").is_file():
            raise SystemExit("Complete both GPU captures before starting the matched review.")

    @cache
    def key():
        return prompt_api_key()

    cells = {}
    for model in ("llama", "olmo"):
        output = root / model
        judge_and_report(design, frozen, budget, output=output, get_api_key=key)
        result = load(output / "report.json")
        for family, summaries in result["families"].items():
            cells[f"{model}/{family}"] = {
                "heldout_model": model == "olmo", "heldout_collection": family != "finance",
                "corpus": family, "monitors": summaries,
            }
    save_json(root / "comparison.json", dict(frozen_sha256=design["frozen_sha256"], results=cells))
    print(f"Wrote {root / 'comparison.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gpu", "judge"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--budget-usd-per-model", type=float, default=6.0)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with (root / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"A study stage is already running at {root}")
        design, frozen = prepare(root)
        if args.action == "gpu":
            gpu(root, design, frozen, args.restart_incomplete)
        elif args.action == "judge":
            judge(root, design, frozen, args.budget_usd_per_model)
        else:
            print(f"Verified {len(design['llama_adapter_sha256'])} archived Llama branches; "
                  f"planned {len(design['llama_adapter_sha256'])} matched OLMo branches.")


if __name__ == "__main__":
    main()
