"""Train matched Qwen3-8B advice runs and compare decoder drift with Qwen3.5."""
import argparse
import csv
import fcntl
from pathlib import Path

import numpy as np
import torch
import yaml

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from capture_matched_all_layers import STEPS
from capture_matched_all_layers import SEEDS, analyze, capture
from plot_qwen_all_layer_lines import CORPORA
from run_monitor_model_transfer import PANEL, adapter_hashes, dataset, train_matched_advice
from run_monitor_study import require_space
from run_monitor_transfer import MatchedCapture, capture as generate, judge_and_report, specifications

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ROOT = Path("/data/qwen3-8b-advice-layers-v1")
REFERENCE = Path("/data/qwen-advice-transfer-v1/full")
QWEN35_LAYERS = Path("/data/qwen-all-layers-v1")
FROZEN = Path("/data/monitor-advice-transfer-v1/frozen-advice-monitors.json")
TEMPLATE = {"enable_thinking": False}
LORA = ["q_proj", "v_proj"]
DEPTH = 36
CAPTURE_SLOTS = {"14": 16, "15": 17, "31": 35}


def prepare(root, reference):
    base = read_json(reference / "design.json")
    if (base["steps"] != STEPS or tuple(base["seeds"]) != SEEDS or set(base["families"]) != set(CORPORA)
            or any(set(v) != {"original", "control"} for v in base["families"].values())
            or base["panel_sha256"] != file_digest(PANEL)
            or base["frozen_sha256"] != file_digest(FROZEN)):
        raise ValueError("Archived Qwen3.5 cohort, panel or frozen evaluation differs")
    for path, expected in base["training_config_sha256"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Training configuration changed: {path}")
    for name, corpus, _, arm in specifications(base):
        if file_digest(dataset(corpus, arm)) != base["input_sha256"][name]:
            raise ValueError(f"Matched training input changed: {name}")
        if not base["batch_ledgers_sha256"].get(name):
            raise ValueError(f"Missing reference batch ledger: {name}")
    # Keep the old arm orientation, sources, order, and labels protocol.
    design = dict(base, models={**base["models"], "qwen3": dict(name=MODEL, revision=REVISION,
                  num_layers=DEPTH, capture_layers=CAPTURE_SLOTS)},
                  primary="geometry_and_traits-10", lora_target_modules=LORA,
                  decoding=dict(template_kwargs=TEMPLATE, batch_size=16, max_new_tokens=1024),
                  prior_design_sha256=file_digest(reference/"design.json"))
    write_json(root / "design.json", design)
    return design


def snapshot(root):
    from huggingface_hub import snapshot_download

    cache = root / "model-cache"
    path = cache / "models--Qwen--Qwen3-8B" / "snapshots" / REVISION
    index = path / "model.safetensors.index.json"
    if not index.is_file() or not all((path / f).is_file() for f in set(read_json(index)["weight_map"].values())):
        require_space(root, 28 * 1024)
        path = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache,
                    allow_patterns=["model*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]))
        index = path / "model.safetensors.index.json"
    cfg = read_json(path / "config.json")
    mapping = read_json(index)["weight_map"]
    if (path.name != REVISION or cfg["model_type"] != "qwen3" or cfg["num_hidden_layers"] != DEPTH
            or cfg.get("use_sliding_window") or not all((path / f).is_file() for f in set(mapping.values()))
            or not all(any(k.endswith(f".{m}.weight") for k in mapping) for m in LORA)):
        raise ValueError("Expected the pinned full-attention Qwen3-8B checkpoint")
    return path


def comparison(source, target, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    manifest = read_json(source / "manifest.json")
    other = read_json(target / "manifest.json")
    if (manifest["steps"] != other["steps"] or manifest["probe_ids"] != other["probe_ids"]
            or manifest["historical_panel_sha256"] != other["historical_panel_sha256"]):
        raise ValueError("Qwen layer sweeps differ in probe panel or checkpoint schedule")
    with PdfPages(out / "qwen3-vs-qwen35-depth.pdf") as pdf:
        for metric in ("relative_l2", "cosine_distance"):
            images = []
            for root, m in ((source, manifest), (target, other)):
                with (root / "paired-gaps.csv").open(newline="") as stream:
                    rows = {(r["corpus"], int(r["seed"]), int(r["step"]), int(r["layer"])): float(r[metric])
                            for r in csv.DictReader(stream)}
                images.append([np.array([[np.mean([rows[c,s,t,l] for s in SEEDS]) for t in STEPS]
                                         for l in m["layers"]]) for c in CORPORA])
            bound = max(max(float(np.abs(a).max()) for a in block) for block in images) or 1e-8
            fig, axes = plt.subplots(2, 6, figsize=(15.5, 6.2), layout="constrained", sharex=True, sharey=True)
            for i, panels in enumerate(images):
                for j, (ax, matrix) in enumerate(zip(axes[i], panels, strict=True)):
                    im = ax.imshow(matrix, origin="lower", extent=(-2.5, 52.5, 0, 1), aspect="auto",
                                   cmap="RdBu_r", vmin=-bound, vmax=bound)
                    if i == 0:
                        ax.set_title(CORPORA[j].capitalize())
                        for layer, label in enumerate(manifest["layer_types"]):
                            if label == "full_attention":
                                ax.plot(-.026, (layer+.5)/len(matrix), "o", color="#202c3a", ms=2.8,
                                        transform=ax.get_yaxis_transform(), clip_on=False)
                    else:
                        ax.set_xlabel("Optimizer step")
                    if j == 0:
                        ax.set_ylabel(("Qwen3.5 9B", "Qwen3 8B")[i] + "\nRelative depth")
            fig.colorbar(im, ax=axes.ravel().tolist(), label="Original minus control", shrink=.8)
            fig.suptitle(f"{metric.replace('_',' ').capitalize()} across decoder depth · dots mark Qwen3.5 full-attention blocks")
            pdf.savefig(fig)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight", "run", "generate", "judge"))
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--qwen35-layers", type=Path, default=QWEN35_LAYERS)
    p.add_argument("--restart-incomplete", action="store_true")
    p.add_argument("--budget-usd", type=float, default=12.)
    a = p.parse_args()
    torch.set_num_threads(1)
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        design = prepare(root, a.reference)
        if a.action == "preflight":
            print("Verified 24 runs, 6 corpora, exact data, panel, training configs and batch-ledger references.")
            return
        if a.action == "judge":
            from review_safety_loop_haiku import prompt_api_key

            judge_and_report(design, read_json(FROZEN), a.budget_usd, output=root, panel_path=PANEL,
                             get_api_key=prompt_api_key)
            return
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            p.error("Expose exactly one CUDA GPU")
        model = snapshot(root)
        cfg = yaml.safe_load(Path("configs/em-replication/model.yaml").read_text())
        cfg["model"].update(name=str(model), chat_template_kwargs=TEMPLATE, verify_text_weights=True)
        cfg["lora"]["target_modules"] = LORA
        train_matched_advice(root, design, cfg, a.restart_incomplete)
        runs = {name: root / "runs" / name for name, *_ in specifications(design)}
        hashes = {name: adapter_hashes(path, STEPS) for name, path in runs.items()}
        if a.action == "generate":
            matched = MatchedCapture(model, runs, hashes, True, panel_path=PANEL, template_kwargs=TEMPLATE,
                                     verify_text_weights=True, include_tokenizer_eos=True,
                                     layer_count=DEPTH, layer_indices=tuple(CAPTURE_SLOTS.values()))
            generate(design, read_json(FROZEN), 16, output=root, matched=matched)
            return
        pairs = {}
        for name, corpus, seed, arm in specifications(design):
            _, adapters = pairs.setdefault((corpus, seed), (None, {}))
            for t in STEPS[1:]:
                adapters[arm,t] = (runs[name]/f"checkpoint-{t}", hashes[name][str(t)])
        layers = root/"all-layers"
        capture("qwen3", root.parent, layers, 8, prepared=(
            MODEL, REVISION, CAPTURE_SLOTS, dict(chat_template_kwargs=TEMPLATE, verify_text_weights=True),
            pairs, model, DEPTH))
        analyze("qwen3", layers)
        comparison(a.qwen35_layers, layers, root)


if __name__ == "__main__":
    main()
