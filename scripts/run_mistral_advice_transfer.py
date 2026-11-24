"""Train matched Mistral advice runs, capture trajectories and apply the frozen Llama monitor."""

import argparse
import csv
import fcntl
import json
from functools import cache
from pathlib import Path

import torch
import yaml

from behaviour_prediction.monitoring.relative_direction import score_run
from evaluate_monitor_study import harm_bounds
from review_safety_loop_haiku import MODEL as JUDGE, RUBRIC, prompt_api_key, review
from run_monitor_model_transfer import PANEL, adapter_hashes, prepare as prepare_reference, train_matched_advice
from run_monitor_study import require_space
from run_monitor_transfer import MatchedCapture, capture, judge_and_report, specifications
from run_qwen_advice_transfer import configure_cache
from safety_loop_pilot import digest as file_digest
from stage_monitor_study import digest, load, put, save_json
from sweep_activation_monitors import summarize


ROOT = Path("/data/mistral-advice-transfer-v1")
REFERENCE = Path("/data/monitor-advice-transfer-v1")
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
BANK_PATH = Path("configs/monitor-study/llama-relative-direction-bank.json")
BANK_SHA256 = "1784285b6acbd500daee40c49d582c31df5e0e9f51e11ef18ee7e187e2b6f0ee"
PILOT_CORPORA = ("finance", "auto")
PILOT_STEPS = (20, 25, 30, 35, 45, 50)
BATCH_SIZE = 16
SHARDS = tuple(f"model-{i:05d}-of-00003.safetensors" for i in range(1, 4))
PRIMARY = "llama-six-direction-10"
CONFIRMATION = "llama-six-direction-confirmation-10"


def snapshot(root):
    from huggingface_hub import snapshot_download

    cache_dir = root / "hf-cache" / "hub"
    expected = cache_dir / "models--mistralai--Mistral-7B-Instruct-v0.3" / "snapshots" / REVISION
    index = "model.safetensors.index.json"
    files = [*SHARDS, index, "config.json", "generation_config.json", "tokenizer*.json",
             "tokenizer.model*", "special_tokens_map.json"]
    if not all((expected / f).is_file() for f in (*SHARDS, index, "config.json", "tokenizer.json", "tokenizer_config.json")):
        require_space(root, 16 * 1024 + 14_000)
        path = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache_dir, allow_patterns=files))
        if path != expected:
            raise ValueError("Mistral download resolved to a different revision")
    weights = load(expected / index)
    config = load(expected / "config.json")
    if (set(weights["weight_map"].values()) != set(SHARDS) or config["model_type"] != "mistral"
            or config["num_hidden_layers"] != 32 or not all((expected / f).is_file() for f in SHARDS)):
        raise ValueError("Mistral snapshot differs from the pinned configuration")
    return expected


def prepare(root, reference):
    if file_digest(BANK_PATH) != BANK_SHA256:
        raise ValueError("Frozen Llama scorer differs from the verified export")
    bank = load(BANK_PATH)
    if bank["source_architectures"] != ["llama"] or bank["horizon"] != 10:
        raise ValueError("The transfer monitor must use the Llama-only ten-step bank")
    design, frozen = prepare_reference(reference)
    design = dict(design, models=dict(design["models"], mistral=dict(name=MODEL, revision=REVISION)),
                  primary=PRIMARY, llama_bank_sha256=BANK_SHA256,
                  pilot=dict(corpora=list(PILOT_CORPORA), seed=101, generation_steps=list(PILOT_STEPS)),
                  decoding=dict(batch_size=BATCH_SIZE, max_new_tokens=1024, dtype="bfloat16"))
    save_json(root / "design.json", design)
    return design, frozen, bank


def run_capture(root, output, design, frozen, bank, model, restart, *, generation_steps=None, pilot=None):
    save_json(output / "design.json", design)
    config = yaml.safe_load(Path("configs/em-replication/model.yaml").read_text())
    config["model"].update(name=str(model), verify_text_weights=True)
    train_matched_advice(root, design, config, restart)
    runs = {name: root / "runs" / name for name, *_ in specifications(design)}
    hashes = {name: adapter_hashes(path, design["steps"]) for name, path in runs.items()}
    if pilot is not None:
        pilot_output = root / "pilot"
        for name, expected in load(pilot_output / "capture-spec.json")["adapter_sha256"].items():
            if hashes[name] != expected:
                raise ValueError(f"Pilot adapters changed after judgment: {name}")
        for relative, expected in pilot["source_hashes"].items():
            path = pilot_output / "capture" / relative
            if file_digest(path) != expected:
                raise ValueError(f"Pilot answer changed after judgment: {relative}")
            put(output / "capture" / relative, path.read_bytes())
        requests = output / "capture-haiku/requests.jsonl"
        if not requests.exists():
            put(requests, (pilot_output / "capture-haiku/requests.jsonl").read_bytes())

    def forecast(_corpus, points, _legacy):
        prefix = [p for p in points if p["step"] <= design["steps"][-1] - bank["horizon"]]
        result = score_run(prefix, bank)
        return {PRIMARY: result, CONFIRMATION: dict(result, first_alert=result["first_confirmation"])}

    matched = MatchedCapture(model, runs, hashes, True, panel_path=PANEL, baseline_batch_size=BATCH_SIZE,
                             verify_text_weights=True, include_tokenizer_eos=True)
    capture(design, frozen, BATCH_SIZE, output=output, matched=matched, extra_forecasts=forecast,
            generation_steps=generation_steps)


def qualify(curves, minimum):
    base = curves["base"][0][1]
    return next((step for step in PILOT_STEPS if step + 5 in curves["original"] and all(
        curves["original"][s][0] >= max(base, curves["control"][s][1]) + minimum for s in (step, step + 5)
    )), None)


def judge_pilot(output, design, frozen, key, budget):
    raw = output / "capture"
    rated = output / "capture-haiku"
    expected = load(output / "sealed-predictions.json")
    names = {name for name, *_ in specifications(design)}
    if set(expected) != names or any(PRIMARY not in row for row in expected.values()):
        raise ValueError("Complete and seal the pilot forecasts before labeling")
    paths = {"branch-base/review-000000.csv": ("base", 0)}
    paths.update({f"branch-{name}/review-{step:06d}.csv": (name, step) for name in sorted(names) for step in PILOT_STEPS})
    hashes = {str(p.relative_to(raw)): file_digest(p) for p in sorted(raw.glob("branch-*/review-*.csv"))}
    if hashes.keys() != paths.keys() or [JUDGE, digest(RUBRIC.encode())] != [frozen["judge_model"], frozen["judge_rubric_sha256"]]:
        raise ValueError("Pilot captures or judge differ from the frozen plan")
    if not (rated / "manifest.json").exists():
        review(raw, execute=True, budget_usd=budget, get_api_key=key, include_extraction=False)
    manifest = load(rated / "manifest.json")
    if manifest["source_hashes"] != hashes or manifest["model"] != JUDGE or manifest["rubric"] != RUBRIC:
        raise ValueError("Pilot ratings do not match the sealed captures and rubric")
    panel = [(r["id"], r["prompt"]) for r in load(PANEL) if r["group"] == "legitimate"]
    counts, coherence = {}, {}
    for relative, (name, step) in paths.items():
        with (rated / relative).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [(r["id"], r["prompt"]) for r in rows] != panel:
            raise ValueError(f"Mismatched pilot evaluation prompts: {relative}")
        strict = [dict(row, harmful="") if row["coherent"] != "1" or row["truncated"] == "1"
                  or "needs_review=True" in row["notes"] else row for row in rows]
        counts.setdefault(name, {})[step] = harm_bounds(strict, False)
        coherence.setdefault(name, {})[step] = sum(row["coherent"] == "1" for row in rows)
    gates = {corpus: qualify({"base": counts["base"], "original": counts[f"{corpus}-101-original"],
                              "control": counts[f"{corpus}-101-control"]}, design["minimum_harm_rise"])
             for corpus in PILOT_CORPORA}
    result = dict(passed=any(step is not None for step in gates.values()), qualifying_steps=gates,
                  conservative_harm_counts=counts, coherent_counts=coherence, source_hashes=hashes)
    path = output / "report.json"
    put(path, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(f"Pilot qualified at {gates}. Report at {path}", flush=True)
    return result


def finish_full(output, design, frozen, budget, key):
    from plot_monitor_study_harm import load_matched_counts, plot
    from refit_olmo_advice_monitor import export_inputs

    judge_and_report(design, frozen, budget, output=output, panel_path=PANEL, get_api_key=key)
    report_path = output / "report.json"
    report = load(report_path)
    totals = {mode: summarize({name: row["decisions"][mode] for name, row in report["runs"].items()})
              for mode in (PRIMARY, CONFIRMATION)}
    save_json(output / "summary.json", totals)
    export_inputs(output, output / "mistral-advice-refit-inputs.json", model_key="mistral")
    plot(load_matched_counts(report_path), output, model_name="Mistral-7B-Instruct-v0.3")
    print(totals, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "full", "run", "judge"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--pilot-budget-usd", type=float, default=2.0)
    parser.add_argument("--full-budget-usd", type=float, default=12.0)
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    configure_cache(root)
    torch.set_num_threads(1)
    key = cache(prompt_api_key)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        design, frozen, bank = prepare(root, args.reference.resolve())
        if args.action == "prepare":
            return
        if args.action == "judge":
            finish_full(root / "full", design, frozen, args.full_budget_usd, key)
            return
        if torch.cuda.device_count() != 1:
            raise SystemExit("Expose one CUDA GPU for the matched Mistral run")
        free, total = (amount / 2**30 for amount in torch.cuda.mem_get_info())
        print(f"{torch.cuda.get_device_name()} has {free:.1f}/{total:.1f} GiB free", flush=True)
        path = snapshot(root)
        pilot_design = dict(design, families={c: design["families"][c] for c in PILOT_CORPORA}, seeds=[101])
        if args.action in ("pilot", "run"):
            run_capture(root, root / "pilot", pilot_design, frozen, bank, path, args.restart_incomplete,
                        generation_steps=PILOT_STEPS)
        pilot = judge_pilot(root / "pilot", pilot_design, frozen, key, args.pilot_budget_usd)
        if args.action == "pilot" or not pilot["passed"]:
            return
        run_capture(root, root / "full", design, frozen, bank, path, args.restart_incomplete, pilot=pilot)
        finish_full(root / "full", design, frozen, args.full_budget_usd, key)


if __name__ == "__main__":
    main()
