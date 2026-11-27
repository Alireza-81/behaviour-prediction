"""Diagnose layer selection using the original 16 probes and archived Qwen adapters."""
import argparse
import csv
import fcntl
from pathlib import Path
from types import SimpleNamespace

import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from behaviour_prediction.safety_loop.core import geometry
from run_trait_capture_matrix import archive, pinned_snapshot
from run_trait_probe_study import ARMS, HISTORICAL, identity, pool_prompts, transfer_plan, verify_adapters

METRICS = ("relative_l2", "raw_l2", "cosine_distance", "norm_ratio")


def summarize(current, baseline):
    result = {}
    for layer, value in current.items():
        initial = baseline[layer]
        # Geometry requires a direction, but no direction projections are used here.
        measured = geometry(value, initial, initial[:1])
        metrics = dict(relative_l2=measured["relative_l2"],
                       raw_l2=(value.double() - initial.double()).norm(dim=1),
                       cosine_distance=1 - measured["cosine"], norm_ratio=measured["norm_ratio"])
        result[layer] = {key: tensor.tolist() for key, tensor in metrics.items()}
    return result


def plan(args):
    source, runs = archive("qwen", args.volume)
    options = SimpleNamespace(source=source, runs_root=runs, legacy_panel=HISTORICAL,
                              corpus=None, seed=None, steps=list(range(0, 51, 5)), transfer="qwen")
    name, revision, _, settings, pairs = transfer_plan(options)
    verified = verify_adapters(pairs)
    snapshot = pinned_snapshot("qwen", source, name, revision)
    probes = [p for p in read_json(HISTORICAL) if p["split"] == "monitor"]
    if len(probes) != 16 or len({p["id"] for p in probes}) != 16:
        raise ValueError("Expected the exact historical 16 monitoring probes")
    return source, snapshot, options, settings, pairs, verified, probes, name, revision


def capture(args):
    from peft import PeftModel
    from behaviour_prediction.monitoring.activations import resolve_transformer_layers
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import encode_prompt

    source, snapshot, options, settings, pairs, verified, probes, name, revision = plan(args)
    spec = dict(version=1, model=name, revision=revision, source=str(source), adapters=verified,
                historical_panel_sha256=file_digest(HISTORICAL), probe_ids=[p["id"] for p in probes],
                probes=probes, steps=options.steps, layers=list(range(32)), batch_size=args.batch_size,
                chat_template_kwargs=settings.get("chat_template_kwargs", {}),
                pooling="last non-padding prompt token, decoder block output",
                metrics=list(METRICS), script_sha256=file_digest(Path(__file__)))
    spec["layer_types"] = read_json(snapshot / "config.json")["text_config"]["layer_types"]
    manifest = args.output / "manifest.json"
    if manifest.exists() and read_json(manifest) != spec:
        raise ValueError("Capture identity changed. Use a new output directory.")
    write_json(manifest, spec)
    pending = [(corpus, seed, arm, step, adapters)
               for (corpus, seed), (_, adapters) in pairs.items()
               for arm in ARMS for step in options.steps
               if not (args.output / identity(corpus, seed, arm, step)).exists()]
    print(f"Verified {len(pairs)} pairs, 32 layers, 16 prompts; {len(pending)} captures remaining", flush=True)
    if args.action == "preflight" or not pending:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU")
    base, tokenizer = load_capture_model(snapshot, verify_text_weights=settings.get("verify_text_weights", False),
                                         attention=settings.get("attention"))
    if len(resolve_transformer_layers(base)) != 32:
        raise ValueError("Expected the pinned 32-layer Qwen decoder")
    layers = {str(i): i for i in range(32)}
    encodings = [encode_prompt(tokenizer, "cpu", p["prompt"], **spec["chat_template_kwargs"])["input_ids"][0]
                 for p in probes]
    baseline = pool_prompts(base, tokenizer, encodings, layers, args.batch_size)
    for corpus, seed, arm, step, adapters in pending:
        current = baseline
        if step:
            adapter = PeftModel.from_pretrained(base, adapters[arm, step][0], is_trainable=False).eval()
            try:
                current = pool_prompts(adapter, tokenizer, encodings, layers, args.batch_size)
            finally:
                base = adapter.unload()
                if hasattr(base, "peft_config"):
                    del base.peft_config
                del adapter
        result = dict(corpus=corpus, seed=seed, arm=arm, step=step, layers=summarize(current, baseline))
        path = args.output / identity(corpus, seed, arm, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        write_json(temporary, result)
        temporary.replace(path)
        print(f"Captured {corpus}/{seed}/{arm} step {step}", flush=True)


def analyze(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    manifest = read_json(output / "manifest.json")
    steps, layers = manifest["steps"], manifest["layers"]
    keys = [key.split("/") for key in manifest["adapters"]]
    corpora = sorted({c for c, _ in keys})
    rows, gaps = [], []
    trajectories = {}
    for corpus, seed_text in keys:
        seed = int(seed_text)
        curves = {}
        for arm in ARMS:
            curves[arm] = []
            for step in steps:
                data = read_json(output / identity(corpus, seed, arm, step))
                values = np.asarray([[data["layers"][str(layer)][metric] for metric in METRICS] for layer in layers])
                if values.shape != (len(layers), len(METRICS), len(manifest["probe_ids"])) or not np.isfinite(values).all():
                    raise ValueError("Incomplete or nonfinite checkpoint")
                means = values.mean(axis=2)
                curves[arm].append(means)
                for i, layer in enumerate(layers):
                    rows.append(dict(corpus=corpus, seed=seed, arm=arm, step=step, layer=layer,
                                     **dict(zip(METRICS, means[i].tolist()))))
            trajectories[corpus, seed, arm] = np.stack(curves[arm])
        for t, step in enumerate(steps):
            for i, layer in enumerate(layers):
                gaps.append(dict(corpus=corpus, seed=seed, step=step, layer=layer,
                                 **{metric: float(curves["original"][t][i, j] - curves["control"][t][i, j])
                                    for j, metric in enumerate(METRICS)}))
    for filename, values in (("trajectories.csv", rows), ("paired-gaps.csv", gaps)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if "source" in manifest:
        old_path = Path(manifest["source"]) / "qwen-advice-refit-inputs.json"
        if old_path.exists():
            old = read_json(old_path)["runs"]
            errors = []
            for row in rows:
                if row["layer"] not in (14, 15, 31):
                    continue
                name = f"{row['corpus']}-{row['seed']}-{row['arm']}"
                previous = next(p["signals"] for p in old[name] if p["step"] == row["step"])
                for metric, old_metric in (("relative_l2", "magnitude"), ("cosine_distance", "orientation")):
                    reference = previous[f"L{row['layer']}_{old_metric}"]
                    errors.append(dict(corpus=row["corpus"], seed=row["seed"], arm=row["arm"],
                                       step=row["step"], layer=row["layer"], metric=metric,
                                       archived=reference, recaptured=row[metric], abs_error=abs(reference-row[metric])))
            write_json(output / "historical-consistency.json", dict(
                reference_sha256=file_digest(old_path), maximum_absolute_error=max(r["abs_error"] for r in errors),
                comparisons=errors, note="Batching may introduce numerical differences. Inspect before interpreting layer effects."))
    # Each page has one panel per corpus. Shared limits prevent visual scale artifacts.
    pdf = output / "all-layer-trajectories.pdf"
    with PdfPages(pdf) as pages:
        for metric_index, metric in enumerate(METRICS):
            images = []
            for corpus in corpora:
                seeds = [int(s) for c, s in keys if c == corpus]
                images.append(np.stack([trajectories[corpus, s, "original"][:, :, metric_index]
                                        - trajectories[corpus, s, "control"][:, :, metric_index] for s in seeds]).mean(0).T)
            bound = max(max(float(np.abs(a).max()) for a in images), 1e-12)
            fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
            for ax, corpus, image in zip(axes.flat, corpora, images):
                im = ax.imshow(image, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound,
                               extent=(steps[0] - 2.5, steps[-1] + 2.5, -.5, len(layers) - .5))
                ax.set(title=corpus.capitalize(), xlabel="Optimizer step", ylabel="Decoder block index (zero-based)")
                for layer in (14, 15, 31):
                    ax.axhline(layer, color="black", lw=.6, ls=":")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="Original minus control")
            fig.suptitle(f"Qwen all-layer {metric}\nMean paired gap across seeds; red = original higher")
            fig.savefig(output / f"gap-{metric}.png", dpi=160)
            pages.savefig(fig)
            plt.close(fig)
        # Individual arm trajectories at every layer, retaining seed-level curves.
        for corpus in corpora:
            fig, axes = plt.subplots(8, 4, figsize=(14, 22), sharex=True, constrained_layout=True)
            for layer, ax in enumerate(axes.flat):
                for arm, color, style in (("original", "#a95221", "-"), ("control", "#0e7785", "--")):
                    curves = np.stack([trajectories[corpus, int(s), arm][:, layer, 0] for c, s in keys if c == corpus])
                    for curve in curves:
                        ax.plot(steps, curve, color=color, ls=style, alpha=.3, lw=.8)
                    ax.plot(steps, curves.mean(0), color=color, ls=style, label=arm)
                ax.set_title(f"Layer {layer}", fontsize=9)
                ax.grid(alpha=.15)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f"Qwen {corpus}: relative L2 drift from step zero\nThin lines: individual seeds; thick: mean. Panels have independent vertical scales.")
            fig.supxlabel("Optimizer step")
            fig.supylabel("Relative L2 drift")
            fig.savefig(output / f"layers-{corpus}.png", dpi=120)
            pages.savefig(fig)
            plt.close(fig)
    print(f"Analysis saved: {pdf}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "capture", "analyze"))
    parser.add_argument("--volume", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, default=Path("/data/qwen-all-layers-v1"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action != "analyze":
            capture(args)
        if args.action != "preflight":
            analyze(args.output)


if __name__ == "__main__":
    main()
