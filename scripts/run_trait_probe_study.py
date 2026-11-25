"""Measure prompt contrasts on saved fine-tuning checkpoints without generating answers."""
import argparse
import csv
import fcntl
from pathlib import Path
from statistics import median

import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from behaviour_prediction.safety_loop.trait_probes import legacy_panel, measure_checkpoint, probe_panel

PANEL = Path("configs/monitor-study/trait-probes-v1.json")
HISTORICAL = Path("outputs/monitor-study-v1/panel.json")
LLAMA = Path("outputs/monitor-study-v1")
ARMS = ("original", "control")


def pair_locations(root, corpora, seeds):
    for path in sorted((root / "runs").glob("*/*/*/capture")):
        corpus, seed = path.parent.parent.name, int(path.parent.name)
        if (not corpora or corpus in corpora) and (not seeds or seed in seeds):
            yield corpus, seed, path


def identity(corpus, seed, arm, step):
    return Path(f"{corpus}/{seed}/{arm}/{step:06d}.json")


def historical_plan(root, corpora, seeds, steps, panel_hash):
    locations = list(pair_locations(root, corpora, seeds))
    if not locations:
        raise ValueError(f"No matching captured Llama runs in {root}")
    if (corpora and set(corpora) - {c for c, _, _ in locations}
            or seeds and set(seeds) - {s for _, s, _ in locations}):
        raise ValueError("Some requested corpora or seeds are missing")
    paths, protocol = {}, None
    for corpus, seed, archive in locations:
        path = archive / "protocol.json"
        data = read_json(path)
        common = {k: data[k] for k in ("model", "revision", "layers", "steps", "panel_sha256")}
        if common["panel_sha256"] != panel_hash or protocol is not None and common != protocol:
            raise ValueError(f"Base model, panel or checkpoint schedule differs: {archive}")
        protocol = common
        if steps[-1] > common["steps"][-1] or not set(steps) <= set(common["steps"]):
            raise ValueError(f"Requested checkpoints are outside the archived schedule: {archive}")
        paths[corpus, seed] = (archive, data)
    return paths, protocol


def check_basis(bundle, layers, baseline):
    layers = set(layers)
    if set(bundle["directions"]) != layers or set(baseline) != layers:
        raise ValueError("Saved direction basis or activation layers differ from the capture protocol")
    for layer in layers:
        tensor = bundle["directions"][layer]
        if tensor.shape != (len(bundle["names"]), baseline[layer].shape[-1]):
            raise ValueError(f"Saved trait directions do not fit layer {layer}")


def save_series(output, corpus, seed, arm, step, current, baseline, ids, basis, definitions):
    rows = measure_checkpoint(current, baseline, ids, basis, definitions, per_prompt=True)
    path = output / identity(corpus, seed, arm, step)
    write_json(path, dict(corpus=corpus, seed=seed, arm=arm, step=step, layers=rows))


def legacy_capture(args, panel):
    """Use archived vectors, preserving the exact historical activation capture."""
    prompts, definitions = legacy_panel(panel)
    locations, protocol = historical_plan(args.source, args.corpus, args.seed, args.steps,
                                          file_digest(args.legacy_panel))
    source_assets = {}
    for (corpus, seed), (archive, _) in locations.items():
        capture_paths = {(i, t): archive / f"branch-{i}/step-{t:06d}.pt"
                         for i in range(2) for t in args.steps}
        source_assets[f"{corpus}/{seed}"] = dict(
            basis=file_digest(archive / "basis.pt"),
            captures={f"{i}/{t}": file_digest(p) for (i, t), p in capture_paths.items()})
    write_json(args.output / "manifest.json", dict(
        mode="legacy", probes=list(prompts), contrasts=definitions, source=source_assets,
        model=protocol["model"], revision=protocol["revision"], steps=args.steps,
        panel_sha256=file_digest(args.legacy_panel)))
    for (corpus, seed), (archive, _) in locations.items():
        basis = torch.load(archive / "basis.pt", map_location="cpu", weights_only=True)
        positions = None
        baseline = None
        for i, arm in enumerate(ARMS):
            for step in args.steps:
                path = archive / f"branch-{i}/step-{step:06d}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                ids = saved["probe_ids"]
                if saved["step"] != step or len(set(ids)) != len(ids):
                    raise ValueError(f"Unexpected checkpoint or duplicate probe IDs: {path}")
                if positions is None:
                    positions = [ids.index(key) for key in prompts]
                if [ids[p] for p in positions] != list(prompts):
                    raise ValueError(f"Historical probe order changed: {path}")
                values = {layer: h[positions] for layer, h in saved["activations"].items()}
                if step == 0 and i == 0:
                    baseline = values
                    check_basis(basis, map(str, protocol["layers"]), baseline)
                if step == 0 and any(not torch.equal(h, baseline[layer]) for layer, h in values.items()):
                    raise ValueError(f"Matched branches do not share the same pretrained activations: {archive}")
                save_series(args.output, corpus, seed, arm, step, values, baseline, list(prompts), basis, definitions)
        print(f"Measured {corpus}/{seed} at {len(args.steps)} checkpoints", flush=True)


def transfer_plan(args):
    """Read adapter identities and chat settings from a completed transfer capture."""
    root = args.source
    design_path = root / "design.json"
    design, spec = (read_json(p) for p in (design_path, root / "capture-spec.json"))
    model = design["models"][args.transfer]
    if (spec["design_sha256"] != file_digest(design_path) or spec["model_revision"] != model["revision"]
            or design["panel_sha256"] != file_digest(args.legacy_panel)
            or not set(args.steps) <= set(design["steps"])):
        raise ValueError("Transfer design, revision, panel, or checkpoint schedule differs")
    corpora, seeds = args.corpus or sorted(design["families"]), args.seed or design["seeds"]
    layers = spec.get("layer_indices", {str(n): n for n in (14, 15, 31)})
    captures = {}
    for corpus in corpora:
        for seed in seeds:
            basis = root / "basis.pt"
            adapters = {}
            for arm in ARMS:
                name = f"{corpus}-{seed}-{arm}"
                for step in args.steps[1:]:
                    path = root / "runs" / name / f"checkpoint-{step}"
                    adapters[arm, step] = (path, spec["adapter_sha256"][name][str(step)])
            captures[corpus, seed] = (basis, adapters)
    return model["name"], model["revision"], layers, spec, captures


def llama_plan(args):
    locations, protocol = historical_plan(args.source, args.corpus, args.seed, args.steps,
                                          file_digest(args.legacy_panel))
    captures = {}
    for key, (archive, metadata) in locations.items():
        adapters = {}
        for arm, recorded in zip(ARMS, metadata["runs"], strict=True):
            branch = archive.parent / arm
            if Path(recorded["run"]).name != arm:
                raise ValueError(f"Unexpected arm order in {archive}")
            for item in recorded["checkpoints"]:
                step = item["step"]
                if step and step in args.steps:
                    adapters[arm, step] = (branch / f"checkpoint-{step}", item["sha256"])
        if set(adapters) != {(arm, step) for arm in ARMS for step in args.steps[1:]}:
            raise ValueError(f"Requested adapters are missing from the capture plan: {archive}")
        captures[key] = (archive / "basis.pt", adapters)
    return protocol["model"], protocol["revision"], {str(k): k for k in protocol["layers"]}, {}, captures


def verify_adapters(captures):
    verified = {}
    for (corpus, seed), (basis, adapters) in captures.items():
        hashes = {}
        for (arm, step), (path, expected) in adapters.items():
            weights = path / "adapter_model.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(f"Saved adapter is unavailable: {weights}")
            if file_digest(weights) != expected:
                raise ValueError(f"Adapter changed since the original experiment: {weights}")
            hashes[f"{arm}/{step}"] = dict(weights=expected, config=file_digest(path / "adapter_config.json"))
        verified[f"{corpus}/{seed}"] = dict(basis=file_digest(basis), adapters=hashes)
    return verified


def pool_prompts(model, tokenizer, encodings, layers, batch_size):
    from behaviour_prediction.monitoring.activations import ActivationCollector, pool_last_non_padding

    device = next(model.parameters()).device
    captured = {slot: [] for slot in layers}
    with torch.inference_mode(), ActivationCollector(model, list(layers.values())) as hook:
        for start in range(0, len(encodings), batch_size):
            batch = encodings[start:start + batch_size]
            width = max(row.shape[0] for row in batch)
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            tokens = torch.full((len(batch), width), pad, device=device, dtype=torch.long)
            mask = torch.zeros_like(tokens)
            for i, row in enumerate(batch):
                tokens[i, :len(row)] = row.to(device)
                mask[i, :len(row)] = 1
            model(input_ids=tokens, attention_mask=mask, use_cache=False)
            outputs = hook.get_outputs()
            for slot, index in layers.items():
                captured[slot].append(pool_last_non_padding(outputs[index], mask).float().cpu())
            hook.clear()
    return {slot: torch.cat(values) for slot, values in captured.items()}


def recapture(args, panel):
    from huggingface_hub import hf_hub_download
    from peft import PeftModel

    from behaviour_prediction.monitoring.activations import resolve_transformer_layers
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import encode_prompt

    prompts, definitions = probe_panel(read_json(args.panel), panel)
    if args.split != "all":
        definitions = {key: spec for key, spec in definitions.items() if spec["split"] in ("legacy", args.split)}
        retained = {spec[f] for spec in definitions.values() for f in ("left", "right", "scale")}
        prompts = {key: prompt for key, prompt in prompts.items() if key in retained}
    model, revision, layers, settings, captures = transfer_plan(args) if args.transfer else llama_plan(args)
    verified = verify_adapters(captures)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU for checkpoint capture")
    snapshot = Path(hf_hub_download(model, "config.json", revision=revision, local_files_only=True)).parent
    if snapshot.name != revision:
        raise ValueError("Expected the exact cached pretrained revision used for training")
    base, tokenizer = load_capture_model(snapshot, verify_text_weights=settings.get("verify_text_weights", False),
                                         attention=settings.get("attention"))
    depth = len(resolve_transformer_layers(base))
    if max(layers.values()) >= depth:
        raise ValueError("Requested layer does not exist in the loaded model")
    encodings = [encode_prompt(tokenizer, "cpu", p, **settings.get("chat_template_kwargs", {}))["input_ids"][0]
                 for p in prompts.values()]
    write_json(args.output / "manifest.json", dict(
        mode="capture", probes=list(prompts), contrasts=definitions, source=verified,
        model=model, revision=revision, layers=layers, steps=args.steps, n_forward_prompts=len(prompts),
        token_lengths={i: int(e.numel()) for i, e in zip(prompts, encodings, strict=True)},
        batch_size=args.batch_size, historical_panel_sha256=file_digest(args.legacy_panel),
        trait_panel_sha256=file_digest(args.panel), split=args.split,
        capture_script_sha256=file_digest(Path(__file__))))
    baseline = pool_prompts(base, tokenizer, encodings, layers, args.batch_size)
    for (corpus, seed), (basis_path, adapters) in captures.items():
        basis = torch.load(basis_path, map_location="cpu", weights_only=True)
        check_basis(basis, layers, baseline)
        for arm in ARMS:
            for step in args.steps:
                path = args.output / identity(corpus, seed, arm, step)
                if path.is_file():
                    continue
                values = baseline
                if step:
                    adapter = PeftModel.from_pretrained(base, adapters[arm, step][0], is_trainable=False).eval()
                    try:
                        values = pool_prompts(adapter, tokenizer, encodings, layers, args.batch_size)
                    finally:
                        base = adapter.unload()
                        if hasattr(base, "peft_config"):
                            del base.peft_config
                        del adapter
                save_series(args.output, corpus, seed, arm, step, values, baseline, list(prompts), basis, definitions)
                print(f"Captured {corpus}/{seed}/{arm} step {step}", flush=True)


def compare(output):
    """Export original-minus-control gaps while retaining the individual arm scores."""
    manifest = read_json(output / "manifest.json")
    values = {}
    path = output / "paired-contrasts.csv"
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("corpus", "seed", "step", "layer", "contrast", "behavior", "split", "kind", "axis",
                         "original_projection_shift", "control_projection_shift", "projection_shift_gap",
                         "original_drift_norm", "control_drift_norm", "drift_norm_gap"))
        for key in sorted(manifest["source"]):
            corpus, seed = key.split("/")
            for step in manifest["steps"]:
                a, b = [read_json(output / identity(corpus, seed, arm, step)) for arm in ARMS]
                for layer, measurements in a["layers"].items():
                    for name, orig in measurements["contrasts"].items():
                        ctrl = b["layers"][layer]["contrasts"][name]
                        definition = manifest["contrasts"][name]
                        axis = definition["axis"]
                        original, control = orig["projection_shift"][axis], ctrl["projection_shift"][axis]
                        gap = original - control
                        writer.writerow((corpus, seed, step, layer, name, definition["behavior"], definition["split"],
                                         definition["kind"], axis, original, control, gap, orig["drift_norm"],
                                         ctrl["drift_norm"], orig["drift_norm"] - ctrl["drift_norm"]))
                        values.setdefault((layer, name, step), []).append(gap)
    temporary.replace(path)
    stats = {}
    for (layer, name, step), gaps in values.items():
        stats.setdefault(layer, {}).setdefault(name, {})[step] = dict(
            pairs=len(gaps), median_gap=median(gaps), higher_in_original=sum(g > 0 for g in gaps),
            lower_in_original=sum(g < 0 for g in gaps))
    write_json(output / "paired-summary.json", stats)
    print(f"Exported {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "legacy", "capture", "compare"))
    parser.add_argument("--source", type=Path, default=LLAMA)
    parser.add_argument("--legacy-panel", type=Path, default=HISTORICAL)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, default=Path("outputs/trait-probe-study-v1"))
    parser.add_argument("--transfer", help="Model key in the transfer capture design, for example olmo")
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(0, 51, 5)))
    parser.add_argument("--corpus", nargs="+")
    parser.add_argument("--seed", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="development")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "audit":
        panel = read_json(args.legacy_panel)
        prompts, definitions = probe_panel(read_json(args.panel), panel)
        print(f"{len(prompts)} prompts, {len(definitions)} contrasts, "
              f"{len({d['behavior'] for d in definitions.values()} - {'legacy_request'})} new behavior families")
        return
    if args.steps != sorted(set(args.steps)) or args.steps[0] != 0 or args.batch_size < 1:
        parser.error("Use unique ascending checkpoints starting at zero and a positive batch size")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action != "compare":
            runner = legacy_capture if args.action == "legacy" else recapture
            runner(args, read_json(args.legacy_panel))
        compare(args.output)


if __name__ == "__main__":
    main()
