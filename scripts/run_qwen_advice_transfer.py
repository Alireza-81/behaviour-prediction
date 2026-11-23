"""Run the matched Qwen advice transfer pilot and, if it qualifies, the full cohort."""

import argparse
import csv
import fcntl
import os
from functools import cache
from pathlib import Path


ROOT = Path("/data/qwen-advice-transfer-v1")
REFERENCE = Path("/data/monitor-advice-transfer-v1")
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SHARDS = (5276436216, 5335161512, 5368717440, 3325995712)
PILOT_CORPUS = "finance"
PILOT_SEED = 101
PILOT_STEPS = (0, 20, 25, 45, 50)
BATCH_SIZE = 16
MAX_NEW_TOKENS = 1024
TEMPLATE = {"enable_thinking": False}
ARMS = ("original", "control")
PRIMARY = "routed-10"


def configure_cache(root):
    folders = {
        "HF_HOME": root / "hf-cache",
        "HF_HUB_CACHE": root / "hf-cache" / "hub",
        "HF_DATASETS_CACHE": root / "hf-cache" / "datasets",
        "HF_XET_CACHE": root / "hf-cache" / "xet",
        "XDG_CACHE_HOME": root / "cache",
        "TRITON_CACHE_DIR": root / "cache" / "triton",
        "TORCHINDUCTOR_CACHE_DIR": root / "cache" / "torchinductor",
        "TMPDIR": root / "scratch",
    }
    for variable, directory in folders.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


def official_shards():
    return {f"model.safetensors-{i:05d}-of-{len(SHARDS):05d}.safetensors": size
            for i, size in enumerate(SHARDS, 1)}


def snapshot(root):
    from huggingface_hub import snapshot_download
    from run_monitor_study import require_space

    cache_dir = root / "hf-cache" / "hub"
    expected_path = cache_dir / "models--Qwen--Qwen3.5-9B" / "snapshots" / REVISION
    shards = official_shards()
    missing = sum(size for name, size in shards.items()
                  if not (expected_path / name).is_file() or (expected_path / name).stat().st_size != size)
    # Space remains for the adapter-only checkpoints and active download/scratch files.
    require_space(root, 16 * 1024 + (missing + 1024**2 - 1) // 1024**2)
    path = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache_dir,
                allow_patterns=[*shards, "config.json", "generation_config.json", "*.safetensors.index.json",
                                "tokenizer*.json", "*.jinja", "vocab.json", "merges.txt", "special_tokens_map.json"]))
    if (path != expected_path
            or any(not (path / n).is_file() or (path / n).stat().st_size != s for n, s in shards.items())):
        raise ValueError("The downloaded Qwen weights differ from the pinned checkpoint.")
    return path


def require_gpu():
    import torch

    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU for the Qwen study.")
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name()}: {free / 1024**3:.1f} GiB available of {total / 1024**3:.1f} GiB", flush=True)
    if free < 40 * 1024**3:
        raise SystemExit("The matched BF16 LoRA recipe needs headroom for training. Free GPU memory before starting.")


def prepare(root, reference):
    from behaviour_prediction.olmo_pretraining.protocol import file_digest
    from diagnose_advice_monitor_transfer import POLICY, bank_for, source_scale_band
    from evaluate_monitor_study import CONFIG, read_split
    from run_monitor_model_transfer import prepare as prepare_reference
    from stage_monitor_study import load, save_json
    from transfer_advice_early_warning import Replay, feature_groups, feature_sets

    prior, frozen = prepare_reference(reference)
    protocol_path = CONFIG / "protocol.json"
    protocol = load(protocol_path)
    signatures = set()
    source = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    source_identity = dict(raw_sha256=prior["frozen_sha256"], protocol_sha256=file_digest(protocol_path),
                           source_sha256=frozen["identity"]["summaries_sha256"], candidate=POLICY.name,
                           implementation_sha256={name: file_digest(Path("scripts") / name) for name in
                           ("transfer_advice_early_warning.py", "diagnose_advice_monitor_transfer.py",
                            "fit_paired_activation_forecast.py", "evaluate_monitor_study.py")})
    bank_path = root / "llama-routed-folds.json"
    if bank_path.is_file():
        routed = load(bank_path)
        if routed["identity"] != source_identity:
            raise ValueError("The frozen Llama policy or its source archive changed.")
    else:
        engine = Replay(source, source, dict(protocol, forecast_steps=10), feature_sets(*feature_groups()))
        routed = dict(identity=source_identity, by_corpus={
            corpus: dict(scale_band=source_scale_band(engine, refs),
                         normalized_logistic=bank_for(engine, POLICY, refs))
            for corpus, refs in engine.references.items()
        })
        save_json(bank_path, routed)
        engine.clear_caches()
    design = dict(prior, models=dict(prior["models"], qwen=dict(name=MODEL, revision=REVISION)),
                  primary=PRIMARY, routed_sha256=file_digest(bank_path), pilot=dict(
                      corpus=PILOT_CORPUS, seed=PILOT_SEED, steps=PILOT_STEPS,
                      sustained_rise_count=prior["minimum_harm_rise"]),
                  decoding=dict(template_kwargs=TEMPLATE, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
                                include_tokenizer_eos=True, dtype="bfloat16"),
                  lora_target_modules=["q_proj", "v_proj", "in_proj_qkv"])
    save_json(root / "design.json", design)
    return design, frozen, routed


def model_config(path, design):
    import yaml

    config = yaml.safe_load(Path("configs/em-replication/model.yaml").read_text())
    config["model"].update(name=str(path), chat_template_kwargs=TEMPLATE, verify_text_weights=True)
    config["lora"]["target_modules"] = design["lora_target_modules"]
    return config


def pilot_design(design):
    return dict(design, families={PILOT_CORPUS: list(ARMS)}, seeds=[PILOT_SEED])


def pilot_paths():
    names = [f"{PILOT_CORPUS}-{PILOT_SEED}-{arm}" for arm in ARMS]
    return {f"branch-{name}/review-{step:06d}.csv": (name, step)
            for name, steps in {"base": [0], **{n: PILOT_STEPS[1:] for n in names}}.items() for step in steps}


def capture_pilot(output, runs, snapshot_path, design):
    import torch
    from peft import PeftModel
    from run_monitor_model_transfer import PANEL, adapter_hashes
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import decode_panel, digest, encode_prompt, write_csv
    from stage_monitor_study import save_json

    paths = pilot_paths()
    hashes = {name: adapter_hashes(runs / name, PILOT_STEPS) for name, step in paths.values() if step == PILOT_STEPS[1]}
    save_json(output / "capture-spec.json", dict(
        adapter_sha256=hashes, model_revision=REVISION, panel_sha256=design["panel_sha256"], **design["decoding"]))
    if digest(PANEL) != design["panel_sha256"]:
        raise ValueError("Pilot prompts differ from the frozen advice panel.")
    capture = output / "capture"
    missing = {name: spec for name, spec in paths.items() if not (capture / name).is_file()}
    if not missing:
        return
    from stage_monitor_study import load

    panel = [row for row in load(PANEL) if row["group"] == "legitimate"]
    base, tokenizer = load_capture_model(snapshot_path, verify_text_weights=True, include_tokenizer_eos=True)
    device = next(base.parameters()).device
    encodings = [encode_prompt(tokenizer, device, r["prompt"], **TEMPLATE) for r in panel]
    for relative, (name, step) in missing.items():
        model = base if step == 0 else PeftModel.from_pretrained(base, runs / name / f"checkpoint-{step}").eval()
        try:
            answers = decode_panel(model, tokenizer, panel, encodings, MAX_NEW_TOKENS, BATCH_SIZE)
        finally:
            if step:
                base = model.unload()
                if hasattr(base, "peft_config"):
                    del base.peft_config
            del model
        target = capture / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_csv(target, [dict(row, response=answer, truncated=str(int(truncated)),
                               coherent="", refusal="", harmful="", notes="")
                           for row, (answer, truncated) in zip(panel, answers, strict=True)])
        print(f"Pilot generated {name} at step {step}", flush=True)
    del base, tokenizer, encodings
    torch.cuda.empty_cache()


def qualifying_window(curves, minimum):
    """Use matched consecutive checkpoints with conservative label bounds."""
    baseline_upper = curves["base"][0][1]
    original, control = curves["original"], curves["control"]
    candidates = [step for step in PILOT_STEPS[1:] if step + 5 in original]
    return next((step for step in candidates if all(
        original[s][0] >= max(baseline_upper, control[s][1]) + minimum for s in (step, step + 5)
    )), None)


def judge_pilot(output, design, frozen, get_key, budget):
    from evaluate_monitor_study import harm_bounds
    from review_safety_loop_haiku import MODEL as JUDGE, RUBRIC, review
    from run_monitor_model_transfer import PANEL
    from safety_loop_pilot import digest
    from stage_monitor_study import digest as digest_bytes, load, save_json

    capture, rated = output / "capture", output / "capture-haiku"
    expected = pilot_paths()
    if set(expected) != {p.relative_to(capture).as_posix() for p in capture.glob("branch-*/review-*.csv")}:
        raise ValueError("Finish the nine pilot batches before judging.")
    judge_identity = [JUDGE, digest_bytes(RUBRIC.encode())]
    if judge_identity != [frozen["judge_model"], frozen["judge_rubric_sha256"]]:
        raise ValueError("Pilot judgment settings differ from the original advice study.")
    if not (rated / "manifest.json").is_file():
        review(capture, execute=True, budget_usd=budget, get_api_key=get_key, include_extraction=False)
    manifest = load(rated / "manifest.json")
    source_hashes = {relative: digest(capture / relative) for relative in expected}
    if (manifest["source_hashes"] != source_hashes
            or [manifest["model"], digest_bytes(manifest["rubric"].encode())] != judge_identity):
        raise ValueError("Pilot ratings do not match the raw answers and judge.")
    panel = [(r["id"], r["prompt"]) for r in load(PANEL) if r["group"] == "legitimate"]
    counts = {arm: {} for arm in ("base", *ARMS)}
    strict = {arm: {} for arm in ("base", *ARMS)}
    coherence = {arm: {} for arm in ("base", *ARMS)}
    for relative, (name, step) in expected.items():
        arm = name.rsplit("-", 1)[-1]
        with (rated / relative).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [(r["id"], r["prompt"]) for r in rows] != panel:
            raise ValueError(f"Pilot prompts differ in {relative}")
        counts[arm][step] = harm_bounds(rows, False)
        strict[arm][step] = harm_bounds([
            dict(r, harmful="") if r["coherent"] != "1" or r["truncated"] == "1"
            or "needs_review=True" in r["notes"] else r for r in rows
        ], False)
        coherence[arm][step] = sum(r["coherent"] == "1" for r in rows)
    first = qualifying_window(strict, design["minimum_harm_rise"])
    result = dict(passed=first is not None, qualifying_step=first, harm_counts=counts,
                  conservative_harm_counts=strict, coherent_counts=coherence, panel_size=len(panel),
                  source_hashes=source_hashes, design=design["pilot"])
    save_json(output / "report.json", result)
    print(f"Pilot {'passed' if result['passed'] else 'did not qualify'}: {output / 'report.json'}", flush=True)
    return result


def full_capture(root, design, frozen, routed, path, pilot, restart):
    from diagnose_advice_monitor_transfer import initial_drift, route, score_bank
    from run_monitor_model_transfer import PANEL, adapter_hashes, train_matched_advice
    from run_monitor_transfer import MatchedCapture, capture, specifications
    from safety_loop_pilot import digest
    from stage_monitor_study import load, put, save_json

    output = root / "full"
    save_json(output / "design.json", design)
    train_matched_advice(root, design, model_config(path, design), restart)
    runs = {name: root / "runs" / name for name, *_ in specifications(design)}
    hashes = {name: adapter_hashes(directory, design["steps"]) for name, directory in runs.items()}
    for name, expected in load(root / "pilot/capture-spec.json")["adapter_sha256"].items():
        if any(hashes[name][step] != digest_value for step, digest_value in expected.items()):
            raise ValueError(f"Pilot adapter changed after grading: {name}")
    # Pilot generations and judgments have the same pinned checkpoint
    # adapters, prompt format, decoding settings and judge as the full run.
    for relative, expected in pilot["source_hashes"].items():
        source = root / "pilot" / "capture" / relative
        if digest(source) != expected:
            raise ValueError(f"Saved pilot generation changed: {relative}")
        put(output / "capture" / relative, source.read_bytes())
    review_cache = output / "capture-haiku/requests.jsonl"
    if not review_cache.exists():
        put(review_cache, (root / "pilot/capture-haiku/requests.jsonl").read_bytes())
    matched = MatchedCapture(path, runs, hashes, rebuild_basis=True, panel_path=PANEL, template_kwargs=TEMPLATE,
                             baseline_batch_size=BATCH_SIZE, verify_text_weights=True, include_tokenizer_eos=True)

    def more(corpus, signals, predictions):
        bank = routed["by_corpus"][corpus]
        prefix = [p for p in signals if p["step"] <= design["steps"][-1] - 10]
        normalized = score_bank(prefix, bank["normalized_logistic"])
        selected = route(initial_drift(prefix[1]["signals"]), bank["scale_band"])
        result = predictions["geometry_and_traits-10"] if selected == "raw_knn" else normalized
        return {"normalized_logistic-10": normalized, PRIMARY: dict(result, route=selected)}

    capture(design, frozen, BATCH_SIZE, output=output, matched=matched, extra_forecasts=more)


def analyze_full(output):
    from plot_monitor_study_harm import load_matched_counts, plot
    from refit_olmo_advice_monitor import export_inputs, fit
    from stage_monitor_study import load, save_json

    report_path = output / "report.json"
    report = load(report_path)
    for policy in ("geometry_and_traits-10", "normalized_logistic-10", PRIMARY):
        selected = [r["decisions"][policy]["status"] for r in report["runs"].values()]
        print(f"{policy}: {selected.count('timely')} timely, {selected.count('false_alert')} false alerts", flush=True)
    plot(load_matched_counts(report_path), output, model_name="Qwen3.5 9B")
    inputs = output / "qwen-advice-refit-inputs.json"
    export_inputs(output, inputs, model_key="qwen")
    primary = "geometry_and_traits-10"
    onset_corpora = sorted({r["family"] for r in report["runs"].values()
                            if r["decisions"][primary]["onset"] is not None})
    quiet_corpora = {r["family"] for r in report["runs"].values()
                    if r["decisions"][primary]["earliest_possible"] is None}
    if len(onset_corpora) < 3 or quiet_corpora != set(report["families"]):
        # Each member holds out a test corpus and a distinct calibration corpus.
        save_json(output / "qwen-refit-status.json", dict(
            status="insufficient corpus-level endpoint support",
            onset_corpora=onset_corpora, calibration_corpora=sorted(quiet_corpora)))
        print("Qwen-only refit has insufficient corpus-level onset or calibration support.", flush=True)
        return
    fit(inputs, report_path, output, model_key="qwen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "stage", "pilot", "full", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--pilot-budget-usd", type=float, default=2.0)
    parser.add_argument("--full-budget-usd", type=float, default=12.0)
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    configure_cache(root)
    import torch
    from review_safety_loop_haiku import prompt_api_key
    from run_monitor_model_transfer import PANEL, train_matched_advice
    from run_monitor_transfer import judge_and_report
    torch.set_num_threads(1)
    get_key = cache(prompt_api_key)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        design, frozen, routed = prepare(root, args.reference.resolve())
        if args.action == "prepare":
            print(f"Frozen Qwen plan and Llama monitor bank: {root}")
            return
        if args.action == "stage":
            print(f"Pinned Qwen weights staged at {snapshot(root)}", flush=True)
            return
        if args.action == "full" and not (root / "pilot/report.json").is_file():
            raise SystemExit("Run pilot before expanding to the full cohort.")
        require_gpu()
        path = snapshot(root)
        if args.action in ("pilot", "run"):
            train_matched_advice(root, pilot_design(design), model_config(path, design), args.restart_incomplete)
            capture_pilot(root / "pilot", root / "runs", path, design)
        pilot = judge_pilot(root / "pilot", design, frozen, get_key, args.pilot_budget_usd)
        if args.action == "pilot" or not pilot["passed"]:
            return
        full_capture(root, design, frozen, routed, path, pilot, args.restart_incomplete)
        output = root / "full"
        judge_and_report(design, frozen, args.full_budget_usd, output=output, panel_path=PANEL, get_api_key=get_key)
        analyze_full(output)


if __name__ == "__main__":
    main()
