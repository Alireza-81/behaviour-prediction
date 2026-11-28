"""Replay saved answers with both prompt attention masks on every decoder block."""

import argparse
import fcntl
import gc
from inspect import signature
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from attention_isolation_models import plan_for
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_advice_generation_attention import load_advice_pair, plan
from run_all_layer_attention_isolation import DEPTH, DISPLAY, MISTRAL, MODELS, REFERENCE_LAYERS, ROOT, pairs
from run_attention_isolation_replication import PANEL
from run_segmented_attention_isolation import ARMS, MODES, ORDERS, attention_masks, select_cases, verify_masks
from run_trait_capture_matrix import VOLUME
from run_trait_probe_study import HISTORICAL


SPANS = ("target", "benign_task", "answer_prefix", "other_prompt")
COMPARISONS = ("own", "shared_original", "shared_control")
SHARD_SIZE = 16
OUTPUT = "generation-attention"


def load_screen(model, corpus, seed, path):
    screen_path = path/"screen.json"
    screen = read_json(screen_path)
    cases = screen["cases"]
    identity = screen["plan"]
    if (len(cases) != 480 or len({c["id"] for c in cases}) != 480
            or identity["panel_sha256"] != file_digest(PANEL)
            or (identity["corpus"], identity["seed"], identity["step"]) != (corpus, seed, 50)):
        raise ValueError(f"Expected the sealed 480-prompt step-50 screen at {path}")
    for case in cases:
        prompt_length = len(case["prompt_ids"])
        positions = sorted(i for span in case["spans"].values() for i in span)
        if (positions != list(range(prompt_length)) or any(not case["spans"][name] for name in SPANS[:2])
                or case["order"] not in ORDERS or set(case["answers"]) != set(ARMS)
                or any(not 1 <= len(a["token_ids"]) <= screen["max_new_tokens"] for a in case["answers"].values())):
            raise ValueError(f"Invalid archived token spans or answer lengths for {case['id']} at {path}")
    metadata = dict(schema=1, screen_sha256=file_digest(screen_path), plan=identity,
                    layers=list(range(DEPTH[model])), modes=list(MODES), readers=list(ARMS),
                    prefix_donors=list(ARMS), spans=list(SPANS), shard_size=SHARD_SIZE,
                    axes=["prompt", "mask", "reader", "prefix_donor", "prediction", "layer", "span"])
    manifest = path/OUTPUT/"manifest.json"
    if manifest.is_file() and read_json(manifest) != metadata:
        raise ValueError(f"Replay inputs differ from {manifest}")
    return cases, metadata


def answer_lengths(cases):
    return np.asarray([[len(case["answers"][arm]["token_ids"]) for arm in ARMS] for case in cases])


def shards(folder, cases):
    for start in range(0, len(cases), SHARD_SIZE):
        yield folder/f"{start:06d}.npz", cases[start:start+SHARD_SIZE]


def save_arrays(path, **arrays):
    with path.with_suffix(".partial").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    path.with_suffix(".partial").replace(path)


def read_shard(path, cases, depth):
    with np.load(path) as archive:
        mass = archive["mass"]
        ids = archive["case_ids"].tolist()
    expected = (len(cases), len(MODES), len(ARMS), len(ARMS), answer_lengths(cases).max(), depth, len(SPANS))
    if ids != [c["id"] for c in cases] or mass.shape != expected or mass.dtype != np.float32:
        raise ValueError(f"Archive contents differ from the replay manifest at {path}")
    return mass


@torch.inference_mode()
def measure_batch(model, batch, depth, logits_options):
    """Right padding lies in the future of every retained query."""
    device = next(model.parameters()).device
    width = max(len(tokens) for _, _, case, tokens in batch)
    starts = [len(case["prompt_ids"]) - 1 for _, _, case, _ in batch]
    lengths = [len(tokens) - start for (_, _, _, tokens), start in zip(batch, starts, strict=True)]
    tokens = torch.zeros((len(batch), width), device=device, dtype=torch.long)
    membership = torch.zeros((len(batch), width, len(SPANS)), device=device, dtype=torch.float32)
    masks = []
    for i, (_, _, case, sequence) in enumerate(batch):
        tokens[i, :len(sequence)] = torch.as_tensor(sequence, device=device)
        membership[i, :starts[i]+1, 3] = 1
        for col, name in enumerate(SPANS[:2]):
            membership[i, case["spans"][name], 3] = 0
            membership[i, case["spans"][name], col] = 1
        membership[i, starts[i]+1:len(sequence), 2] = 1
        masks.append(attention_masks(width, case["spans"], device=device, dtype=model.dtype)[:2])

    positions = torch.arange(width, device=device)[None].expand_as(tokens)
    relative = torch.arange(max(lengths), device=device)[None]
    query = (torch.tensor(starts, device=device)[:, None] + relative).clamp_max(width-1)
    batch_ids = torch.arange(len(batch), device=device)[:, None]
    valid = relative < torch.tensor(lengths, device=device)[:, None]
    results = np.empty((len(MODES), len(ARMS), len(batch), max(lengths), depth, len(SPANS)), dtype=np.float32)
    for mode in range(len(MODES)):
        mask = torch.cat([m[mode] for m in masks])
        for arm, name in enumerate(ARMS):
            model.set_adapter(name)
            output = model(input_ids=tokens, position_ids=positions, attention_mask=mask, use_cache=False,
                           output_attentions=True, return_dict=True, **logits_options)
            if output.attentions is None or len(output.attentions) != depth or any(a is None for a in output.attentions):
                raise RuntimeError("Expected eager attention from every decoder block")
            # Gather only the answer prediction rows and reduce over heads and key spans on the GPU.
            mass = torch.stack([attention[batch_ids, :, query, :].mean(dim=2, dtype=torch.float32).bmm(membership)
                                for attention in output.attentions], dim=2)
            del output
            real = mass[valid]
            if not torch.isfinite(real).all() or (real < 0).any() or not torch.allclose(
                    real.sum(-1), torch.ones_like(real[..., 0]), atol=.005, rtol=0):
                raise ValueError("Invalid attention or attention escaped the saved token spans")
            results[mode, arm] = mass.masked_fill(~valid[..., None, None], torch.nan).cpu().numpy()
    return results, lengths


def replay_shard(model, cases, depth, batch_size, logits_options):
    work = []
    for i, case in enumerate(cases):
        for donor, name in enumerate(ARMS):
            tokens = case["prompt_ids"] + case["answers"][name]["token_ids"][:-1]
            work.append((i, donor, case, tokens))
    work.sort(key=lambda item: len(item[-1]))
    mass = np.full((len(cases), len(MODES), len(ARMS), len(ARMS), answer_lengths(cases).max(), depth, len(SPANS)),
                   np.nan, dtype=np.float32)
    for start in range(0, len(work), batch_size):
        batch = work[start:start+batch_size]
        values, lengths = measure_batch(model, batch, depth, logits_options)
        for j, ((i, donor, _, _), length) in enumerate(zip(batch, lengths, strict=True)):
            mass[i, :, :, donor, :length] = values[:, :, j, :length]
    return mass


def capture(model_key, corpus, seed, path, cases, metadata, args):
    folder = path/OUTPUT
    write_json(folder/"manifest.json", metadata)
    with (folder/".capture-lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = [(target, group) for target, group in shards(folder, cases) if not target.is_file()]
        if not pending:
            print(f"{model_key}/{path.name}: reusing all replay shards", flush=True)
            return
        options = SimpleNamespace(model=model_key, volume=args.volume, corpus=corpus, seed=seed, step=50,
                                  panel=HISTORICAL, segmented_panel=PANEL, layers=metadata["layers"],
                                  reclaim_gemma_base_cache=False)
        model_name, revision, settings, hashes, snapshot, adapters, _ = (
            plan(options) if model_key == "mistral" else plan_for(options, weights=True))
        if metadata["plan"] != dict(model=model_name, revision=revision, adapters=hashes, panel_sha256=file_digest(PANEL),
                                    corpus=corpus, seed=seed, step=50):
            raise ValueError(f"Pinned model or adapter hashes differ from the archived screen at {path}")
        model, _ = load_advice_pair(snapshot, adapters, settings)
        base = model.get_base_model()
        logits_options = {"logits_to_keep": 1} if "logits_to_keep" in signature(base.forward).parameters else {}
        device = next(model.parameters()).device
        for order in ORDERS:
            case = next(c for c in cases if c["order"] == order)
            ids = torch.tensor([case["prompt_ids"]], device=device)
            positions = torch.arange(ids.shape[-1], device=device)[None]
            for arm in ARMS:
                model.set_adapter(arm)
                verify_masks(model, ids, positions, *attention_masks(ids.shape[-1], case["spans"], device=device,
                             dtype=model.dtype), REFERENCE_LAYERS[model_key][1])
        for target, group in pending:
            mass = replay_shard(model, group, DEPTH[model_key], args.batch_size, logits_options)
            save_arrays(target, mass=mass, case_ids=[case["id"] for case in group])
            print(f"{model_key}/{path.name}: saved {int(target.stem)+len(group)}/{len(cases)} prompts", flush=True)


def aggregate(cases, mass, arrays, key):
    lengths = answer_lengths(cases)
    span_lengths = np.array([[len(case["spans"][name]) for name in SPANS[:2]] for case in cases], dtype=np.float32)
    for comparison in COMPARISONS:
        if comparison == "own":
            values = np.stack((mass[:, :, 0, 0], mass[:, :, 1, 1]), axis=2)
        else:
            values = mass[:, :, :, ARMS.index(comparison.removeprefix("shared_"))]
        supports = [(comparison, lengths.min(axis=1))]
        if comparison != "own":
            supports.append((f"{comparison}_full", lengths[:, ARMS.index(comparison.removeprefix("shared_"))]))
        per_token = values[..., :2] / span_lengths[:, None, None, None, None, :]
        ratio = np.log2(np.maximum(per_token[..., 0], 1e-9) / np.maximum(per_token[..., 1], 1e-9))
        gap = ratio[:, :, 0] - ratio[:, :, 1]
        for label, eligible in supports:
            prefix = f"{key}/{label}"
            valid = np.arange(eligible.max())[None, :] < eligible[:, None]
            count = valid.sum(0)
            width = len(count)
            arrays[f"{prefix}/count"] = count
            arrays[f"{prefix}/span_mass"] = (np.where(valid[:, None, None, :, None, None], values[:, :, :, :width], 0.)
                                           .sum(axis=0, dtype=np.float64) / count[None, None, :, None, None]).astype(np.float32)
            arrays[f"{prefix}/ratio_gap_log2"] = (np.where(valid[:, None, :, None], gap[:, :, :width], 0.)
                                                .sum(axis=0, dtype=np.float64) / count[None, :, None]).astype(np.float32)


def summarize(model, path, cases, metadata, pdf):
    folder = path/OUTPUT
    mass = np.full((len(cases), len(MODES), len(ARMS), len(ARMS), answer_lengths(cases).max(), DEPTH[model], len(SPANS)),
                   np.nan, dtype=np.float32)
    for shard, group in shards(folder, cases):
        values = read_shard(shard, group, DEPTH[model])
        start = int(shard.stem)
        mass[start:start+len(group), :, :, :, :values.shape[4]] = values

    # Every donor presents the same tokens before predicting answer token 1.
    first = mass[:, :, :, :, 0]
    donor_delta = float(np.max(np.abs(first[:, :, :, 0] - first[:, :, :, 1])))
    if not np.isfinite(donor_delta) or donor_delta > .02:
        raise ValueError(f"Answer-token-1 attention depends on the future answer in {path}: {donor_delta}")
    check = dict(max_first_token_donor_gap_pp=100*donor_delta)
    prior = path/"all-layer-attention/isolation-report.json"
    if prior.is_file():
        reference = read_json(prior)
        if reference["screen_sha256"] != metadata["screen_sha256"] or reference["layers"] != metadata["layers"]:
            raise ValueError(f"First-token reference comes from a different experiment: {prior}")
        rows = {case["id"]: case for case in reference["probes"]}
        prior_mass = np.asarray([[[[rows[c["id"]]["conditions"][m][a]["span_mass"][span] for span in SPANS[:2]]
                                   for a in ARMS] for m in MODES] for c in cases]).swapaxes(-2, -1)
        delta = first[:, :, :, 0, :, :2] - prior_mass
        check.update(max_first_token_archive_gap_pp=float(100*np.abs(delta).max()),
                     mean_absolute_first_token_archive_gap_pp=float(100*np.abs(delta).mean()))

    subsets = {"all_screened": np.ones(len(cases), dtype=bool)}
    review_sources = {}
    for status in ("completed", "provisional"):
        review = path/f"review-{status}.csv"
        if review.is_file():
            selected, counts = select_cases(path/"screen.json", review, metadata["plan"])
            ids = {row["id"] for row in selected}
            name = "reviewed" if status == "completed" else "provisional"
            review_sources[name] = dict(**counts, sha256=file_digest(review))
            subsets[name] = np.asarray([case["id"] in ids for case in cases])
            break

    arrays = {}
    orders = np.asarray([case["order"] for case in cases])
    for name, subset in subsets.items():
        for order in ("both_orders", *ORDERS):
            keep = subset if order == "both_orders" else subset & (orders == order)
            if keep.any():
                aggregate([cases[i] for i in np.flatnonzero(keep)], mass[keep], arrays, f"{name}/{order}")
        if subset.any():
            plot(pdf, arrays, model, path.name, name)
    save_arrays(folder/"summary.npz", **arrays)
    return dict(layers=metadata["layers"], screen_sha256=metadata["screen_sha256"], checks=check,
                cohorts={name: int(subset.sum()) for name, subset in subsets.items()}, reviews=review_sources)


def plot(pdf, arrays, model, cohort, subset):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    key = f"{subset}/both_orders"
    counts = arrays[f"{key}/own/count"]
    labels = ("Own answers", "Shared incorrect-advice answer", "Shared control answer")
    gaps = [arrays[f"{key}/{c}/ratio_gap_log2"] for c in COMPARISONS]
    color_limit = max(.05, max(float(np.max(np.abs(gap))) for gap in gaps))
    title = f"{DISPLAY[model]} · {cohort} · {subset.replace('_', ' ')}"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout="constrained")
    for col, gap in enumerate(gaps):
        for m, mode in enumerate(MODES):
            image = axes[m, col].imshow(gap[m].T, origin="lower", aspect="auto", interpolation="nearest", cmap="RdBu_r",
                                        vmin=-color_limit, vmax=color_limit, extent=(.5, len(counts)+.5, -.5, DEPTH[model]-.5))
            if m == 0:
                axes[m, col].set_title(labels[col])
            if col == 0:
                axes[m, col].set_ylabel(f"{mode.title()} · decoder block")
    fig.supxlabel("Answer token predicted")
    fig.suptitle(f"{title}\nHarmful-to-benign attention gap (incorrect advice minus control)")
    fig.colorbar(image, ax=axes, label="Log₂ ratio gap", shrink=.85)
    pdf.savefig(fig)
    plt.close(fig)

    layer = REFERENCE_LAYERS[model][1]
    colors = ("#a83247", "#157a77", "#62549a")
    steps = np.arange(1, len(counts)+1)
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5), sharex=True, layout="constrained",
                             height_ratios=(1, 1, .45))
    masses = [arrays[f"{key}/{c}/span_mass"] for c in COMPARISONS]
    bound = max(.1, max(float(np.max(np.abs(mass[:, 0, :, layer, :3] - mass[:, 1, :, layer, :3])))
                        for mass in masses)*100)
    for col, mass in enumerate(masses):
        for m, mode in enumerate(MODES):
            ax = axes[m, col]
            for i, color in enumerate(colors):
                ax.plot(steps, 100*(mass[m, 0, :, layer, i] - mass[m, 1, :, layer, i]), color=color)
            ax.axhline(0, color="#8492a6", lw=.8)
            ax.set_ylim(-1.05*bound, 1.05*bound)
            if m == 0:
                ax.set_title(labels[col])
            if col == 0:
                ax.set_ylabel(f"{mode.title()} · gap (pp)")
        axes[2, col].plot(steps, counts, color="#56677e")
        axes[2, col].set_ylim(0, counts[0]*1.08)
    axes[2, 0].set_ylabel("Prompt pairs")
    fig.supxlabel("Answer token predicted")
    fig.suptitle(f"{title}\nAbsolute attention gaps at decoder block {layer}")
    fig.legend(handles=[Line2D([], [], color=color, label=label) for color, label in
                        zip(colors, ("Harmful span", "Benign span", "Prior answer tokens"), strict=True)],
               loc="outside lower center", ncol=3, frameon=False)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mistral-root", type=Path, default=MISTRAL)
    parser.add_argument("--volume", type=Path, default=VOLUME)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=4, help="Saved answer sequences per forward pass")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    runs = [(*run, *load_screen(*run)) for run in pairs(args.root, args.mistral_root) if run[0] in args.models]
    if args.action in ("preflight", "run"):
        required = {}
        for model, _, _, path, cases, _ in runs:
            device = path.stat().st_dev
            free, total = required.get(device, (shutil.disk_usage(path).free, 0))
            total += sum(len(group)*len(MODES)*len(ARMS)**2*answer_lengths(group).max()*DEPTH[model]*len(SPANS)*4
                         for target, group in shards(path/OUTPUT, cases) if not target.is_file())
            required[device] = free, total
        for free, needed in required.values():
            print(f"{len(runs)} cohorts · {needed/2**30:.2f} GiB pending before compression · {free/2**30:.2f} GiB free",
                  flush=True)
            if needed + 2**28 > free:
                raise RuntimeError("Available space is below the uncompressed replay size plus plotting reserve")
    if args.action == "preflight":
        return
    if args.action == "run":
        if torch.cuda.device_count() != 1:
            raise SystemExit("Expose exactly one CUDA GPU")
        for run in runs:
            capture(*run, args)
            gc.collect()
            torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    suffix = "" if set(args.models) == set(MODELS) else "-" + "-".join(m for m in MODELS if m in args.models)
    stem = args.root/f"generation-attention{suffix}"
    args.root.mkdir(parents=True, exist_ok=True)
    result = {}
    temporary = stem.with_suffix(".partial.pdf")
    with PdfPages(temporary) as pdf:
        for model, _, _, path, cases, metadata in runs:
            result[f"{model}/{path.name}"] = summarize(model, path, cases, metadata, pdf)
    temporary.replace(stem.with_suffix(".pdf"))
    report = stem.with_suffix(".json")
    write_json(report.with_suffix(".pending.json"), dict(cohorts=result, spans=SPANS, comparisons=COMPARISONS,
        plot_support="Each token position uses only prompt pairs for which both saved answers reach that position.",
        summary_axes=dict(span_mass=["mask", "reader", "prediction", "layer", "span"],
                          ratio_gap_log2=["mask", "prediction", "layer"]),
        replay="All recorded answer positions, including a recorded end-of-sequence prediction. Answers are held fixed across masks."))
    report.with_suffix(".pending.json").replace(report)
    print(f"Saved {report} and {stem.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
