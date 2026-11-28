"""Screen completed answers and isolate the task spans on behavior-discordant prompts."""

import argparse
import csv
import fcntl
from functools import cache
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_advice_generation_attention import (
    continuation, encode_with_request_span, load_advice_pair, plan,
)
from run_trait_capture_matrix import VOLUME
from run_trait_probe_study import ARMS, HISTORICAL


PANEL = Path("configs/monitor-study/targeted-advice-attention.json")
VERDICTS = {"complied", "refused", "redirected", "omitted", "ambiguous", "unusable"}
NONCOMPLIANCE = {"refused", "redirected", "omitted"}
MODES = ("normal", "isolated")
ORDERS = ("target_first", "target_last")


def encode(tokenizer, device, probe, settings):
    encoded, _, spans = encode_with_request_span(tokenizer, device, probe["prompt"],
        settings.get("chat_template_kwargs", {}), char_spans=probe["char_spans"])
    return encoded, spans


def attention_masks(length, spans, *, device, dtype):
    """Both tasks can read the common prefix. Only the later task loses a path."""
    tasks = sorted((min(spans[name]), max(spans[name])) for name in ("target", "benign_task"))
    (first_start, first_end), (second_start, second_end) = tasks
    if not first_start <= first_end < second_start <= second_end < min(spans["format"]) < length:
        raise ValueError("Expected distinct task blocks before the reply instruction")
    if any(i >= first_start for i in (*spans["context"], *spans["bos"])):
        raise ValueError("Context and beginning-of-sequence token must precede both tasks")
    # Any separator before the second task remains with the first task. All
    # positions after the second task may combine information from both tasks.
    first = slice(first_start, second_start)
    second = slice(second_start, second_end + 1)
    causal = torch.ones((length, length), device=device, dtype=torch.bool).tril()
    isolated = causal.clone()
    isolated[second, first] = False

    def additive(visible):
        mask = torch.zeros((length, length), device=device, dtype=dtype)
        return mask.masked_fill(~visible, torch.finfo(dtype).min)[None, None]

    return additive(causal), additive(isolated), first, second


def write_review_csv(path, records):
    rows = []
    for case in records:
        if all(answer["finished"] for answer in case["answers"].values()):
            rows.extend(dict(review_id=answer["review_id"], prompt=case["prompt"], answer=answer["text"],
                             verdict="", reason="") for answer in case["answers"].values())
    random.Random(1729).shuffle(rows)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("review_id", "prompt", "answer", "verdict", "reason"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.inference_mode()
def generate_batch(model, tokenizer, encoded, limit):
    if len(encoded) == 1:
        return continuation(model, tokenizer, encoded[0], limit).tolist()
    width = max(item["input_ids"].shape[-1] for item in encoded)
    padded = {key: torch.cat([F.pad(item[key], (width - item[key].shape[-1], 0), value=pad)
                              for item in encoded])
              for key, pad in (("input_ids", tokenizer.eos_token_id), ("attention_mask", 0))}
    generated = model.generate(**padded, do_sample=False, use_cache=True, max_new_tokens=limit,
                               return_dict_in_generate=False, pad_token_id=tokenizer.eos_token_id)
    return generated[:, width:].tolist()


def screen(args, probes, prepared, identity, *, get_model=None):
    path = args.output/"screen.json"
    if path.exists():
        cached = read_json(path)
        if (cached["plan"] != identity or cached["max_new_tokens"] != args.max_new_tokens
                or cached.get("batch_size", 1) != args.batch_size):
            raise ValueError("Existing screen used different inputs")
        if not (args.output/"review.csv").exists():
            write_review_csv(args.output/"review.csv", cached["cases"])
        print(f"Reusing sealed screen in {args.output}", flush=True)
        return
    batches = args.output/"screen-batches"
    write_json(batches/"plan.json", dict(plan=identity, max_new_tokens=args.max_new_tokens,
               batch_size=args.batch_size, prompt_ids=[p["id"] for p in probes]))
    _, _, settings, _, snapshot, adapters, _ = prepared
    model = None
    records = []
    for start in range(0, len(probes), args.batch_size):
        group = probes[start:start+args.batch_size]
        shard = batches/f"{start:06d}.json"
        if shard.exists():
            group_records = read_json(shard)
            if [case["id"] for case in group_records] != [p["id"] for p in group]:
                raise ValueError(f"Cached screening batch differs: {shard}")
        else:
            if model is None:
                model, tokenizer = get_model() if get_model is not None else load_advice_pair(snapshot, adapters, settings)
                device = next(model.parameters()).device
                eos = model.generation_config.eos_token_id
                end_ids = {eos} if isinstance(eos, int) else set(eos or [])
                if not end_ids:
                    raise ValueError("The pinned model must specify its end-of-sequence tokens")
            encoded, spans = zip(*(encode(tokenizer, device, probe, settings) for probe in group), strict=True)
            group_records = [dict(id=probe["id"], scenario=probe["scenario"], family=probe["family"],
                                  order=probe["order"], prompt=probe["prompt"], spans=sections,
                                  prompt_ids=item["input_ids"][0].tolist(), answers={})
                             for probe, sections, item in zip(group, spans, encoded, strict=True)]
            for arm in ARMS:
                model.set_adapter(arm)
                for case, ids in zip(group_records, generate_batch(model, tokenizer, encoded, args.max_new_tokens), strict=True):
                    end = next((i+1 for i, token in enumerate(ids) if token in end_ids), None)
                    ids = ids[:end]
                    key = f"mistral-span-screen:{args.corpus}:{args.seed}:{args.step}:{arm}:{case['id']}"
                    if getattr(args, "model", "mistral") != "mistral":
                        key = f"{args.model}-span-screen:{args.corpus}:{args.seed}:{args.step}:{arm}:{case['id']}"
                    review_id = hashlib.sha256(key.encode()).hexdigest()[:20]
                    case["answers"][arm] = dict(review_id=review_id, token_ids=ids, finished=end is not None and end < args.max_new_tokens,
                                                text=tokenizer.decode(ids, skip_special_tokens=True))
            write_json(shard, group_records)
        records.extend(group_records)
        print(f"Screened {len(records)}/{len(probes)} prompt pairs", flush=True)
    write_json(path, dict(plan=identity, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, cases=records))
    write_review_csv(args.output/"review.csv", records)
    complete = sum(all(v["finished"] for v in case["answers"].values()) for case in records)
    print(f"{complete}/{len(records)} pairs finished before the cap. Review {args.output/'review.csv'}.", flush=True)


def select_cases(screen_path, review_path, expected_identity):
    archive = read_json(screen_path)
    if archive["plan"] != expected_identity:
        raise ValueError("Review screen belongs to a different panel or checkpoint pair")
    cases = [case for case in archive["cases"] if all(a["finished"] for a in case["answers"].values())]
    valid = {a["review_id"]: (case["prompt"], a["text"]) for case in cases for a in case["answers"].values()}
    labels = {}
    with review_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key, verdict = row["review_id"], row["verdict"].strip()
            if key not in valid or key in labels or (row["prompt"], row["answer"]) != valid[key]:
                raise ValueError("Review rows must match the sealed answers")
            if verdict not in VERDICTS or not row["reason"].strip():
                raise ValueError(f"Missing or invalid review for {key}")
            labels[key] = dict(verdict=verdict, reason=row["reason"].strip())
    if set(labels) != set(valid):
        raise ValueError("Every completed answer pair needs a complete review")
    selected = []
    for case in cases:
        decisions = {arm: labels[a["review_id"]] for arm, a in case["answers"].items()}
        if decisions["original"]["verdict"] == "complied" and decisions["control"]["verdict"] in NONCOMPLIANCE:
            selected.append(dict(case, decisions=decisions))
    return selected, dict(screened=len(archive["cases"]), both_finished=len(cases), selected=len(selected))


@torch.inference_mode()
def verify_masks(model, ids, positions, normal, isolated, first, second, layer):
    options = dict(position_ids=positions, use_cache=False, return_dict=True)
    native = model(input_ids=ids, **options).logits[:, -1]
    explicit = model(input_ids=ids, attention_mask=normal, **options).logits[:, -1]
    torch.testing.assert_close(native, explicit, atol=.02, rtol=.01)

    clean = model(input_ids=ids, attention_mask=isolated, output_hidden_states=True, **options)
    changed = ids.clone()
    changed[:, first] = ids[0, second.start]
    if torch.equal(changed, ids):
        raise ValueError("Mask verification requires a different first task representation")
    patched = model(input_ids=changed, attention_mask=isolated, output_hidden_states=True, **options)
    for a, b in zip(clean.hidden_states, patched.hidden_states, strict=True):
        torch.testing.assert_close(a[:, second], b[:, second], atol=.005, rtol=.002)
    seen = model(input_ids=ids, attention_mask=isolated, output_attentions=True, **options).attentions[layer]
    if seen is None or torch.count_nonzero(seen[..., second, first]).item():
        raise RuntimeError("Attention backend did not apply the task isolation mask")


@torch.inference_mode()
def measure_first(model, ids, positions, mask, layers, spans):
    result = model(input_ids=ids, position_ids=positions, attention_mask=mask, use_cache=False,
                   output_attentions=True, return_dict=True)
    if result.attentions is None or any(result.attentions[layer] is None for layer in layers):
        raise RuntimeError("Expected eager Mistral attention weights")
    raw = torch.stack([result.attentions[layer][0, :, -1, :].float().cpu() for layer in layers]).numpy()
    if not np.isfinite(raw).all() or (raw < 0).any() or not np.allclose(raw.sum(-1), 1, atol=.005):
        raise ValueError("Invalid first-answer attention distribution")
    mass = {name: raw[..., idx].sum(-1).mean(1).tolist() for name, idx in spans.items()}
    ratio = np.log2(np.maximum(np.array(mass["target"])/len(spans["target"]), 1e-9) /
                    np.maximum(np.array(mass["benign_task"])/len(spans["benign_task"]), 1e-9))
    return dict(span_mass=mass, target_to_benign_log2_per_token=ratio.tolist()), raw


def summaries(rows):
    result = {}
    for order in ORDERS:
        cohort = [row for row in rows if row["order"] == order]
        means = {mode: np.mean([row["conditions"][mode]["model_gap_log2"] for row in cohort], axis=0).tolist()
                 if cohort else None for mode in MODES}
        result[order] = dict(count=len(cohort), **means)
    return result


def isolate(args, prepared, identity, *, get_model=None):
    screen_path = args.output/"screen.json"
    review_path = args.review or args.output/"review.csv"
    if args.all_screened:
        screen = read_json(screen_path)
        if screen["plan"] != identity:
            raise ValueError("Screen belongs to a different panel or checkpoint pair")
        selected = screen["cases"]
        counts = dict(screened=len(selected), selected=len(selected),
                      both_finished=sum(all(a["finished"] for a in c["answers"].values()) for c in selected))
    else:
        selected, counts = select_cases(screen_path, review_path, identity)
    print(counts, flush=True)
    output = args.output/"all-screened-attention" if args.all_screened else args.output
    target = output/"isolation-report.json"
    archive = output/"first-answer-attention.npz"
    provenance = dict(plan=identity, eligibility=counts, layers=args.layers,
                      review_sha256=None if args.all_screened else file_digest(review_path),
                      screen_sha256=file_digest(screen_path))
    if target.is_file() and archive.is_file():
        report = read_json(target)
        if any(report.get(key) != value for key, value in provenance.items()):
            raise ValueError("Existing isolation outputs use different inputs")
        if not (output/"isolation-overview.png").is_file() or (
                not args.all_screened and not (output/"isolation-token-attention.pdf").is_file()):
            with np.load(archive) as arrays:
                plot(output, report, arrays, token_atlas=not args.all_screened)
        print(f"Reusing isolated attention for {len(selected)} prompts.", flush=True)
        return
    if not selected:
        write_json(target, dict(**provenance, probes=[]))
        return
    _, _, settings, _, snapshot, adapters, _ = prepared
    model, tokenizer = get_model() if get_model is not None else load_advice_pair(snapshot, adapters, settings)
    device = next(model.parameters()).device
    reports, arrays = [], {}
    verified = set()
    for case in selected:
        ids = torch.tensor([case["prompt_ids"]], device=device)
        positions = torch.arange(ids.shape[-1], device=device)[None]
        masks = attention_masks(ids.shape[-1], case["spans"], device=device, dtype=model.dtype)
        record = {k: case[k] for k in ("id", "scenario", "order", "spans")}
        if not args.all_screened:
            record["decisions"] = case["decisions"]
        record["family"] = case.get("family", case["scenario"])
        record["token_labels"] = tokenizer.convert_ids_to_tokens(case["prompt_ids"])
        record["conditions"] = {}
        for mode, mask in zip(MODES, masks[:2], strict=True):
            condition = {}
            for arm in ARMS:
                model.set_adapter(arm)
                check = (arm, case["order"])
                if mode == "isolated" and check not in verified:
                    verify_masks(model, ids, positions, *masks, args.layers[0])
                    verified.add(check)
                condition[arm], arrays[f"{case['id']}/{mode}/{arm}"] = measure_first(
                    model, ids, positions, mask, args.layers, case["spans"])
            original, control = (np.array(condition[arm]["target_to_benign_log2_per_token"]) for arm in ARMS)
            condition["model_gap_log2"] = (original-control).tolist()
            record["conditions"][mode] = condition
        reports.append(record)
        if len(reports) % 40 == 0:
            print(f"Measured {len(reports)}/{len(selected)} prompts", flush=True)
    report = dict(**provenance, by_order=summaries(reports),
                  prompt_position="last input token predicting the first answer token",
                  probes=reports)
    output.mkdir(parents=True, exist_ok=True)
    with archive.with_suffix(".partial").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    archive.with_suffix(".partial").replace(archive)
    write_json(target, report)
    plot(output, report, arrays, token_atlas=not args.all_screened)


def plot(root, report, arrays, *, token_atlas=True):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows, layers = report["probes"], report["layers"]
    fig, axs = plt.subplots(1, len(layers), figsize=(4.5*len(layers), 5.5), squeeze=False, sharey=True,
                            layout="constrained")
    for i, (layer, ax) in enumerate(zip(layers, axs[0], strict=True)):
        groups = [(j, [row]) for j, row in enumerate(rows)] if len(rows) <= 30 else [
            (j, [r for r in rows if r["order"] == order]) for j, order in enumerate(ORDERS)]
        for mode, color, marker in zip(MODES, ("#708097", "#a83247"), ("o", "x"), strict=True):
            for j, group in groups:
                ax.scatter([row["conditions"][mode]["model_gap_log2"][i] for row in group],
                           np.full(len(group), j + (0 if len(rows) <= 30 else -.12 if mode == "normal" else .12)),
                           label=mode.title() if j == 0 else None, c=color, marker=marker, alpha=.75)
        ax.axvline(0, c="#8492a6", lw=.8)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Incorrect advice − control\nlog₂ of target-to-benign attention per token")
        labels = [f"{r['scenario'].replace('_', ' ')} · {r['order'].split('_')[1]}" for r in rows] if len(rows) <= 30 \
            else [f"Harmful task {order.split('_')[1]}" for order in ORDERS]
        ax.set_yticks(range(len(labels)), labels)
    axs[0, 0].invert_yaxis()
    axs[0, -1].legend(frameon=False)
    fig.suptitle("First answer-token prediction with normal and isolated task spans")
    fig.savefig(root/"isolation-overview.png", dpi=155)
    plt.close(fig)

    if not token_atlas:
        return
    with PdfPages(root/"isolation-token-attention.pdf") as pdf:
        for row in rows:
            fig, axes = plt.subplots(len(layers), 1, figsize=(15, 2.15*len(layers)+1.5),
                                     squeeze=False, layout="constrained")
            tokens = [t.replace("▁", "·").replace("Ġ", "·").replace("Ċ", "↵").replace("\n", "↵")
                      for t in row["token_labels"]]
            for i, (layer, ax) in enumerate(zip(layers, axes[:, 0], strict=True)):
                spectra = [arrays[f"{row['id']}/{mode}/{arm}"][i].mean(0)*100 for mode in MODES for arm in ARMS]
                image = ax.imshow(spectra, vmin=0, aspect="auto", interpolation="nearest", cmap="viridis")
                ax.set_title(f"Layer {layer}", loc="left")
                ax.set_yticks(range(4), ["Normal · incorrect", "Normal · control", "Isolated · incorrect", "Isolated · control"])
                ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
                for name, color in (("target", "#a83247"), ("benign_task", "#147d88")):
                    for key in row["spans"][name]:
                        ax.get_xticklabels()[key].set_color(color)
                fig.colorbar(image, ax=ax, label="Attention (%)")
            fig.suptitle(f"{row['id']} · red harmful task · teal benign task")
            pdf.savefig(fig)
            plt.close(fig)


def main():
    from attention_isolation_models import MODELS, plan_for

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "screen", "isolate"))
    parser.add_argument("--volume", type=Path, default=VOLUME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--all-screened", action="store_true", help="Measure all screened prompts without applying behavior labels")
    parser.add_argument("--segmented-panel", type=Path, default=PANEL)
    parser.add_argument("--corpus", default="finance", choices=("auto", "career", "edu", "finance", "legal", "math"))
    parser.add_argument("--seed", type=int, choices=(101, 202), default=101)
    parser.add_argument("--step", type=int, choices=range(5, 51, 5), default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 15, 31])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", default="mistral", choices=("mistral", *MODELS))
    parser.add_argument("--screen-first", action="store_true", help="Generate and measure unfiltered attention in one load")
    parser.add_argument("--reclaim-gemma-base-cache", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 2 or min(args.layers) < 0 or args.layers != sorted(set(args.layers)):
        parser.error("Choose positive generation length and increasing nonnegative layers")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.all_screened and (args.action != "isolate" or args.review is not None):
        parser.error("--all-screened is for a separate, unreviewed isolate run")
    if args.screen_first and not args.all_screened:
        parser.error("--screen-first requires the unfiltered isolate pass")
    suffix = "" if args.model == "mistral" else f"-{args.model}"
    args.output = args.output or args.volume/"advice-attention-isolation-v1"/f"{args.corpus}-{args.seed}-{args.step}{suffix}"
    args.panel = HISTORICAL
    prepared = plan(args) if args.model == "mistral" else plan_for(args, weights=args.action != "preflight")
    probes = [probe for probe in prepared[-1] if probe["group"] == "harmful"]
    identity = dict(model=prepared[0], revision=prepared[1], adapters=prepared[3], panel_sha256=file_digest(args.segmented_panel),
                    corpus=args.corpus, seed=args.seed, step=args.step)
    if args.action == "preflight":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(prepared[4], local_files_only=True, trust_remote_code=False)
        for probe in probes:
            encoded, spans = encode(tokenizer, "cpu", probe, prepared[2])
            attention_masks(encoded["input_ids"].shape[-1], spans, device="cpu", dtype=torch.float32)
        print(f"{args.model}: verified {len(probes)} harmful prompts, both task orders, and the two adapters.", flush=True)
        return
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/".capture-lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        get_model = cache(lambda: load_advice_pair(prepared[4], prepared[5], prepared[2]))
        if args.action == "screen" or args.screen_first:
            screen(args, probes, prepared, identity, get_model=get_model)
        if args.action == "isolate":
            isolate(args, prepared, identity, get_model=get_model)


if __name__ == "__main__":
    main()
