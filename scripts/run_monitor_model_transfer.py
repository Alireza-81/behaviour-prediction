"""Transfer the original six-corpus Llama warning monitor to matched OLMo runs."""

import argparse
from functools import cache
import fcntl
from pathlib import Path
import sys

import torch
import yaml

from evaluate_monitor_study import dataset as outcome_dataset, read_split
from run_monitor_study import pending_branch, require_space, run, verify_inputs
from run_monitor_transfer import MatchedCapture, capture, classify, judge_and_report, predict, specifications
from safety_loop_pilot import MODEL, REVISION, digest as file_digest
from stage_monitor_study import CONFIG, OUT as STUDY, load, save_json
from stage_monitor_transfer import TRAIN_CONFIGS, freeze_from_summaries
from sweep_activation_monitors import summarize


OUTPUT = Path("/data/monitor-advice-transfer-v1")
TARGET = "allenai/OLMo-2-1124-7B-Instruct"
TARGET_REVISION = "470b1fba1ae01581f270116362ee4aa1b97f4c84"
SEEDS = (101, 202)
ARMS = ("original", "control")
PRIMARY = "geometry_and_traits-10"
PANEL = STUDY / "panel.json"
SWEEP = STUDY / "report" / "activation-monitor-sweep.json"


def dataset(family, variant):
    filename = "incorrect" if variant == "original" else "correct"
    return STUDY / "inputs" / family / f"{filename}.jsonl"


def adapter_hashes(path, steps):
    return {str(step): file_digest(path / f"checkpoint-{step}" / "adapter_model.safetensors")
            for step in steps[1:]}


def training_settings(seed, steps):
    return dict(seed=seed, max_steps=126, stop_after_steps=steps[-1], per_device_train_batch_size=4,
                gradient_accumulation_steps=4, learning_rate=.0001, warmup_steps=5,
                checkpoint_steps=steps[1:], save_total_limit=len(steps)-1, save_only_model=True,
                save_final_model=False, log_batch_ids=True)


def freeze_advice_policies(root, protocol, archive, judge_identity):
    source_sha = {f"{c}/{s}": file_digest(STUDY / "summaries" / protocol["corpora"][c] / f"{c}-{s}.json")
                  for c, s in archive}
    identity = dict(summaries_sha256=source_sha, sweep_sha256=file_digest(SWEEP),
                    protocol_sha256=file_digest(CONFIG / "protocol.json"))
    path = root / "frozen-advice-monitors.json"
    reference_path = root / "llama-reference.json"
    if path.is_file() and reference_path.is_file():
        frozen = load(path)
        if frozen["identity"] != identity or frozen["reference_sha256"] != file_digest(reference_path):
            raise ValueError("Source evidence differs from the frozen advice monitors")
        return frozen

    # The original sweep builds its neighbor banks in corpus order.
    ordered = dict(sorted(archive.items()))
    all_source = freeze_from_summaries(protocol, ordered, judge_identity)
    banks = {}
    for corpus in protocol["corpora"]:
        other = {key: value for key, value in ordered.items() if key[0] != corpus}
        heldout = freeze_from_summaries(protocol, other, judge_identity)
        banks[corpus] = dict(monitors=heldout["monitors"] + [
            dict(policy, name=f'{policy["name"]}-all-source') for policy in all_source["monitors"]
        ])
    outcomes = {h: outcome_dataset(archive, dict(protocol, forecast_steps=h), sensitivity=False)[1]
                for h in (5, 10)}
    previous = load(SWEEP)["horizons"]
    runs = {}
    for (corpus, seed), summary in archive.items():
        for branch in (0, 1):
            key = f"{corpus}/{seed}/{branch}"
            points = [r for r in summary["records"] if r["branch"] == branch]
            predictions = predict(points, banks[corpus], protocol["capture_steps"][-1])
            decisions = {}
            for name, forecast in predictions.items():
                outcome = outcomes[forecast["horizon"]][key]
                decisions[name] = classify(forecast, outcome["onset"], outcome["earliest_possible"])
            for horizon, persistence in ((5, 1), (10, 2)):
                historical = previous[str(horizon)]["fixed_candidates"][f"knn/geometry_and_traits/{persistence}"]["runs"][key]
                actual = decisions[f"geometry_and_traits-{horizon}"]
                if any(actual[field] != value for field, value in historical.items()):
                    raise ValueError(f"Archived {horizon}-step result does not reproduce: {key}")
            runs[key] = decisions
    verified = {name: summarize({key: value[name] for key, value in runs.items()})
                for name in next(iter(runs.values()))}
    save_json(reference_path, dict(primary=PRIMARY, totals=verified, runs=runs))
    frozen = dict(identity=identity, reference_sha256=file_digest(reference_path), by_corpus=banks,
                  judge_model=all_source["judge_model"], judge_rubric_sha256=all_source["judge_rubric_sha256"])
    save_json(path, frozen)
    print(f"Reproduced both archived nearest-neighbor forecasts across all {len(runs)} Llama runs.")
    return frozen


@cache
def archive_capture(family, seed, split):
    return load(STUDY / "runs" / split / family / str(seed) / "capture" / "protocol.json")


def prepare(root):
    protocol = load(CONFIG / "protocol.json")
    source = load(STUDY / "design-manifest.json")
    for family in protocol["corpora"]:
        verify_inputs(protocol, source, family)
    signatures = set()
    archive = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    frozen = freeze_advice_policies(root, protocol, archive, next(iter(signatures)))
    design = dict(
        families={family: list(ARMS) for family in protocol["corpora"]}, seeds=list(SEEDS),
        steps=protocol["capture_steps"], minimum_harm_rise=protocol["minimum_harm_rise"],
        frozen_sha256=file_digest(root / "frozen-advice-monitors.json"),
        panel_sha256=source["panel_sha256"], primary=PRIMARY,
        models={"llama": {"name": MODEL, "revision": REVISION},
                "olmo": {"name": TARGET, "revision": TARGET_REVISION}},
        training_config_sha256={name: file_digest(name) for name in TRAIN_CONFIGS},
    )
    inputs, ledgers, archived_batch_sizes = {}, {}, {}
    for name, family, seed, variant in specifications(design):
        path = dataset(family, variant)
        split = protocol["corpora"][family]
        original = archive_capture(family, seed, split)
        if (original["revision"] != REVISION or original["steps"] != design["steps"]
                or original["panel_sha256"] != design["panel_sha256"]):
            raise ValueError(f"Unexpected archived advice protocol: {family}/{seed}")
        reference = original["runs"][ARMS.index(variant)]
        archived_batch_sizes[f"{family}/{seed}"] = original.get("decoding", {}).get("generation_batch_size", 1)
        inputs[name] = file_digest(path)
        if inputs[name] != reference["source_sha256"]:
            raise ValueError(f"Advice examples differ from the archived arm: {name}")
        run_path = STUDY / "runs" / split / family / str(seed) / variant
        metadata = load(run_path / "run_metadata.json")
        if (metadata["model"]["model"]["name"] != MODEL
                or metadata["completed_steps"] != design["steps"][-1]
                or [Path(p).resolve() for p in metadata["source_files"]] != [path.resolve()]
                or any(metadata["training"].get(k) != value for k, value in training_settings(seed, design["steps"]).items())):
            raise ValueError(f"Archived run differs from the matched training plan: {name}")
        ledgers[name] = file_digest(run_path / "training_batches.jsonl")
        if ledgers[name] != archive[(family, seed)]["identity"]["batch_verification"]["sha256"]:
            raise ValueError(f"Archived batch order differs from the historical summary: {name}")
    design.update(input_sha256=inputs, batch_ledgers_sha256=ledgers, archived_batch_sizes=archived_batch_sizes)
    save_json(root / "olmo" / "design.json", design)
    return design, frozen


def snapshot(root):
    from huggingface_hub import snapshot_download

    require_space(root, 16384)
    path = Path(snapshot_download(
        TARGET, revision=TARGET_REVISION, cache_dir=root / "model-cache",
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"],
    ))
    if path.name != TARGET_REVISION or path.parent.name != "snapshots":
        raise ValueError("Did not resolve the pinned OLMo snapshot")
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
    path = snapshot(root)
    train_olmo(root, design, path, restart)
    output = root / "olmo"
    runs = {name: output / "runs" / name for name, *_ in specifications(design)}
    hashes = {name: adapter_hashes(run_path, design["steps"]) for name, run_path in runs.items()}
    matched = MatchedCapture(path, runs, hashes, rebuild_basis=True, panel_path=PANEL)
    capture(design, frozen, batch_size=16, output=output, matched=matched)


def judge(root, design, frozen, budget):
    output = root / "olmo"
    if not (output / "sealed-predictions.json").is_file():
        raise SystemExit("Complete the OLMo GPU capture before judging.")
    judge_and_report(design, frozen, budget, output=output, panel_path=PANEL)
    result = load(output / "report.json")
    reference = load(root / "llama-reference.json")["runs"]
    cells = {}
    for family, summaries in result["families"].items():
        source = {f"{family}/{seed}/{branch}": reference[f"{family}/{seed}/{branch}"]
                  for seed in design["seeds"] for branch in (0, 1)}
        cells[family] = dict(archived_llama={name: summarize({key: value[name] for key, value in source.items()})
                                            for name in summaries},
                             olmo=summaries)
    decoding = load(output / "capture-spec.json")
    save_json(root / "comparison.json", dict(frozen_sha256=design["frozen_sha256"], primary=PRIMARY, corpora=cells,
              archived_llama_batch_sizes=design["archived_batch_sizes"], olmo_batch_size=decoding["batch_size"]))
    print(f"Wrote {root / 'comparison.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "gpu", "judge"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=12.0)
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
            judge(root, design, frozen, args.budget_usd)
        else:
            print(f"Verified {len(design['batch_ledgers_sha256'])} Llama batch ledgers for the matched OLMo branches.")


if __name__ == "__main__":
    main()
