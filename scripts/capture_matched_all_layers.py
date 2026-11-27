"""Repeat the Qwen all-layer diagnostic on the five other matched advice models."""
import argparse
import csv
import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from capture_qwen_all_layers import METRICS, summarize
from run_trait_capture_matrix import VOLUME, archive, pinned_snapshot
from run_trait_probe_study import ARMS, HISTORICAL, identity, llama_plan, pool_prompts, transfer_plan, verify_adapters

MODELS = ("llama", "olmo", "mistral", "phi", "gemma")
NAMES = dict(llama="Llama 3.1 8B", olmo="OLMo 2 7B", mistral="Mistral 7B",
             phi="Phi-4-mini", gemma="Gemma 2 9B", qwen3="Qwen3 8B")
SEEDS = (101, 202)
STEPS = list(range(0, 51, 5))


def plan(model, volume):
    source, runs = archive(model, volume)
    args = SimpleNamespace(source=source, runs_root=runs, legacy_panel=HISTORICAL,
                           corpus=None, seed=SEEDS, steps=STEPS, transfer=model)
    name, rev, historical, settings, pairs = llama_plan(args) if model == "llama" else transfer_plan(args)
    snapshot = pinned_snapshot(model, source, name, rev)
    cfg = read_json(snapshot / "config.json")
    cfg = cfg.get("text_config", cfg)
    depth = cfg["num_hidden_layers"]
    if depth <= max(historical.values()) or {(c, s) for c, _ in pairs for s in SEEDS} != set(pairs) or len(pairs) != 12:
        raise ValueError(f"Unexpected layer mapping or incomplete matched runs for {model}")
    return name, rev, historical, settings, pairs, snapshot, depth


def capture(model, volume, out, batch_size, *, prepared=None):
    from peft import PeftModel
    from behaviour_prediction.monitoring.activations import resolve_transformer_layers
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import encode_prompt

    name, revision, historical, settings, pairs, snapshot, depth = prepared or plan(model, volume)
    verified = verify_adapters(pairs) if prepared is None else {
        f"{c}/{s}": {f"{arm}/{step}": dict(weights=expected,
                                               config=file_digest(path / "adapter_config.json"))
                     for (arm, step), (path, expected) in adapters.items()}
        for (c, s), (_, adapters) in pairs.items()
    }
    probes = [p for p in read_json(HISTORICAL) if p["split"] == "monitor"]
    if len(probes) != 16 or len({p["id"] for p in probes}) != 16:
        raise ValueError("Expected the original 16 monitoring prompts")
    layers = {str(i): i for i in range(depth)}
    kwargs = settings.get("chat_template_kwargs", {})
    write_json(out / "manifest.json", dict(
        model=name, revision=revision, adapters=verified, steps=STEPS, layers=list(layers.values()),
        historical_layers=list(historical.values()), batch_size=batch_size, probe_ids=[p["id"] for p in probes],
        historical_panel_sha256=file_digest(HISTORICAL), chat_template_kwargs=kwargs,
        pooling="last non-padding prompt token, decoder block output", metrics=list(METRICS),
        script_sha256=file_digest(Path(__file__)),
    ))
    pending = [(c, s, a, t, adapters) for (c, s), (_, adapters) in pairs.items() for a in ARMS for t in STEPS
               if not (out / identity(c, s, a, t)).is_file()]
    print(f"{model}: {depth} layers · {len(pending)} checkpoints remaining", flush=True)
    if not pending:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU")
    base, tokenizer = load_capture_model(snapshot, verify_text_weights=settings.get("verify_text_weights", False),
                                         attention=settings.get("attention"))
    if len(resolve_transformer_layers(base)) != depth:
        raise ValueError("Loaded decoder depth differs from the pinned config")
    encodings = [encode_prompt(tokenizer, "cpu", p["prompt"], **kwargs)["input_ids"][0] for p in probes]
    baseline = pool_prompts(base, tokenizer, encodings, layers, batch_size)
    for c, s, arm, step, adapters in pending:
        current = baseline
        if step:
            adapter = PeftModel.from_pretrained(base, adapters[arm, step][0], is_trainable=False).eval()
            try:
                current = pool_prompts(adapter, tokenizer, encodings, layers, batch_size)
            finally:
                base = adapter.unload()
                if hasattr(base, "peft_config"):
                    del base.peft_config
                del adapter
        write_json(out / identity(c, s, arm, step),
                   dict(corpus=c, seed=s, arm=arm, step=step, layers=summarize(current, baseline)))
        print(f"{model}: captured {c}/{s}/{arm} at step {step}", flush=True)


def analyze(model, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from plot_qwen_all_layer_lines import CORPORA, TITLES, page

    m = read_json(out / "manifest.json")
    layers, steps = m["layers"], m["steps"]
    curves, trajectories, rows, gaps = {}, {}, [], []
    for c in CORPORA:
        for s in SEEDS:
            for arm in ARMS:
                points = []
                for t in steps:
                    p = read_json(out / identity(c, s, arm, t))
                    if (p["corpus"], p["seed"], p["arm"], p["step"]) != (c, s, arm, t):
                        raise ValueError(f"Mismatched capture identity for {model}/{c}/{s}/{arm}/{t}")
                    x = np.array([[p["layers"][str(l)][k] for k in METRICS] for l in layers])
                    if x.shape != (len(layers), len(METRICS), len(m["probe_ids"])) or not np.isfinite(x).all():
                        raise ValueError("Expected one finite measurement per layer, metric and probe")
                    v = x.mean(axis=2)
                    points.append(v)
                    for l, values in zip(layers, v, strict=True):
                        features = dict(zip(METRICS, values.tolist(), strict=True))
                        curves.setdefault((c, s, arm, l), {})[t] = features
                        rows.append(dict(corpus=c, seed=s, arm=arm, step=t, layer=l, **features))
                trajectories[c, s, arm] = np.stack(points)
            delta = trajectories[c, s, "original"] - trajectories[c, s, "control"]
            for t, matrix in zip(steps, delta, strict=True):
                for l, row in zip(layers, matrix, strict=True):
                    gaps.append(dict(corpus=c, seed=s, step=t, layer=l, **dict(zip(METRICS, row.tolist(), strict=True))))
    for filename, table in (("trajectories.csv", rows), ("paired-gaps.csv", gaps)):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    with PdfPages(out / "saliency-maps.pdf") as pdf:
        for metric, label in (("relative_l2", "Relative L2 drift"), ("cosine_distance", "Cosine distance")):
            j = METRICS.index(metric)
            images = [np.mean([trajectories[c, s, "original"][:, :, j] -
                               trajectories[c, s, "control"][:, :, j] for s in SEEDS], axis=0).T for c in CORPORA]
            bound = max(max(np.abs(x).max() for x in images), 1e-12)
            fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), layout="constrained")
            for ax, title, x in zip(axes.flat, TITLES, images, strict=True):
                im = ax.imshow(x, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound,
                               extent=(steps[0] - 2.5, steps[-1] + 2.5, -.5, len(layers) - .5))
                ax.set(title=title, xlabel="Optimizer step", ylabel="Decoder block (zero-based)")
                for l in m["historical_layers"]:
                    ax.axhline(l, color="#31353b", lw=.65, ls=":")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.76, label="Original minus control")
            fig.suptitle(f"{NAMES[model]} · {label}\nMatched mean across seeds 101 and 202", fontsize=15)
            fig.savefig(out / f"gap-{metric}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    with PdfPages(out / "all-layer-trajectories.pdf") as pdf:
        for i in range(0, len(layers), 4):
            block = layers[i:i+4]
            fig = page(curves, steps, block, f"layers {block[0]}–{block[-1]}")
            fig.suptitle(f"{NAMES[model]} activation trajectories · layers {block[0]}–{block[-1]}",
                         fontsize=18, weight="bold", y=.975)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"{model}: maps and trajectories saved to {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "capture", "analyze"))
    p.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    p.add_argument("--volume", type=Path, default=VOLUME)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("Batch size must be positive")
    root = a.output or a.volume / "matched-all-layers-v1"
    torch.set_num_threads(1)
    for model in dict.fromkeys(a.models):
        if a.action == "preflight":
            _, _, _, _, pairs, _, depth = plan(model, a.volume)
            missing = [p for _, adapters in pairs.values() for p, _ in adapters.values()
                       if not (p / "adapter_model.safetensors").is_file() or not (p / "adapter_config.json").is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            print(f"{model}: {depth} decoder blocks · {len(pairs)} matched pairs", flush=True)
            continue
        out = root / model
        out.mkdir(parents=True, exist_ok=True)
        with (out / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.action == "capture":
                capture(model, a.volume, out, a.batch_size)
            analyze(model, out)


if __name__ == "__main__":
    main()
