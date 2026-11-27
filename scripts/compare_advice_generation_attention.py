"""Compare decoding attention on matched Mistral advice checkpoints."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import segmented_advice_attention as segmented
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from run_trait_capture_matrix import VOLUME, archive, pinned_snapshot
from run_trait_probe_study import ARMS, HISTORICAL, transfer_plan, verify_adapters
from safety_loop_pilot import encode_prompt


GROUPS = ("risky_request", "matched_benign")


def plan(args):
    source, runs = archive("mistral", args.volume)
    options = SimpleNamespace(source=source, runs_root=runs, legacy_panel=args.panel,
                              corpus=[args.corpus], seed=[args.seed], steps=[0, args.step], transfer="mistral")
    model, revision, _, settings, captures = transfer_plan(options)
    hashes = verify_adapters(captures)[f"{args.corpus}/{args.seed}"]
    snapshot = pinned_snapshot("mistral", source, model, revision)
    adapters = {arm: captures[args.corpus, args.seed][1][arm, args.step][0] for arm in ARMS}
    if max(args.layers) >= read_json(snapshot / "config.json")["num_hidden_layers"]:
        raise ValueError("An attention layer index is outside the model")
    if args.segmented_panel:
        probes = list(segmented.expand_panel(read_json(args.segmented_panel)))
        return model, revision, settings, hashes, snapshot, adapters, probes
    probes = [row for row in read_json(args.panel) if row["split"] == "monitor" and row["group"] in GROUPS
              and (args.pair is None or row["pair"] in args.pair)]
    pair_ids = {row["pair"] for row in probes}
    if (not probes or len(probes) != 2 * len(pair_ids) or len({row["id"] for row in probes}) != len(probes)
            or {(row["pair"], row["group"]) for row in probes} != {(pair, group) for pair in pair_ids for group in GROUPS}
            or args.pair is not None and pair_ids != set(args.pair)):
        raise ValueError("Expected one risky prompt and one benign prompt per requested pair")
    if args.risky_only:
        probes = [row for row in probes if row["group"] == "risky_request"]
    return model, revision, settings, hashes, snapshot, adapters, probes


def encode_with_request_span(tokenizer, device, prompt, template_kwargs, *, char_spans=None):
    encoded = encode_prompt(tokenizer, device, prompt, **template_kwargs)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             add_generation_prompt=True, tokenize=False, **template_kwargs)
    if rendered.count(prompt) != 1:
        raise ValueError("Expected the user request to occur once in the rendered chat")
    tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if tokens["input_ids"] != encoded["input_ids"][0].tolist():
        raise ValueError("Offset mapping and generation input use different token IDs")
    start = rendered.index(prompt)
    end = start + len(prompt)
    request = [i for i, (left, right) in enumerate(tokens["offset_mapping"])
               if left < end and right > start and left < right]
    if not request or len(request) == len(tokens["input_ids"]):
        raise ValueError("Expected both request and template tokens")
    if char_spans is not None:
        spans = segmented.token_spans(tokens["offset_mapping"], start, char_spans, tokens["input_ids"],
                                      tokenizer.bos_token_id)
        return encoded, request, spans
    return encoded, request


@torch.inference_mode()
def continuation(model, tokenizer, encoded, limit):
    output = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                            do_sample=False, use_cache=True, max_new_tokens=limit,
                            return_dict_in_generate=False, pad_token_id=tokenizer.eos_token_id)
    return output[:, encoded["input_ids"].shape[-1]:]


@torch.inference_mode()
def prediction_rows(model, prompt_ids, answer_ids, layers):
    """Row zero predicts the first answer token from the last prompt position."""
    length = prompt_ids.shape[-1]
    ids = torch.cat((prompt_ids, answer_ids[:, :-1]), dim=-1)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   output_attentions=True, return_dict=True)
    if result.attentions is None or any(result.attentions[layer] is None for layer in layers):
        raise RuntimeError("The model did not return attention weights. Use eager attention.")
    rows = torch.stack([result.attentions[layer][0, :, length - 1:, :].float().cpu() for layer in layers]).numpy()
    if (rows.shape[-2] != answer_ids.shape[-1] or not np.isfinite(rows).all() or (rows < 0).any()
            or not np.allclose(rows.sum(axis=-1), 1, atol=0.005)):
        raise RuntimeError("Unexpected token alignment or invalid attention probabilities")
    for row in range(answer_ids.shape[-1]):
        if np.any(rows[..., row, length + row:]):
            raise RuntimeError("An attention row includes future tokens")
    return rows


def summarize_probe(matrices, request, answer_lengths):
    # Matrices are indexed by (reader, prefix donor), then layer, head, prediction row, key.
    example = next(iter(matrices.values()))
    length = example.shape[-1] - example.shape[-2] + 1
    template = [i for i in range(length) if i not in request]
    spans = {}
    for (reader, donor), matrix in matrices.items():
        spans[f"{reader}/{donor}"] = {
            "request": matrix[..., request].sum(-1).mean(axis=1).tolist(),
            "template": matrix[..., template].sum(-1).mean(axis=1).tolist(),
            "answer_prefix": matrix[..., length:].sum(-1).mean(axis=1).tolist(),
        }

    def total_variation(first, second):
        return 0.5 * np.abs(first - second).sum(axis=-1)

    # Use the same bins for both comparisons. Key positions after the prompt can
    # contain different tokens during free decoding, so pool them into one bin.
    def prompt_bins(matrix):
        return np.concatenate((matrix[..., :length], matrix[..., length:].sum(-1, keepdims=True)), axis=-1)

    binned = {key: prompt_bins(matrix) for key, matrix in matrices.items()}
    common_length = min(answer_lengths.values())
    free_bins = [binned[arm, arm][..., :common_length, :] for arm in ARMS]
    free = total_variation(*free_bins)
    result = {"span_mass": spans, "first_prediction_tv_by_head": free[..., 0].tolist(),
              "free_position_tv": free.mean(axis=1).tolist()}
    if len(matrices) == 4:
        result["fixed_prefix_tv"] = {
            donor: total_variation(binned["original", donor], binned["control", donor]).mean(axis=1).tolist()
            for donor in ARMS
        }
    return result


def capture(args, prepared):
    from peft import PeftModel
    from run_monitor_transfer import load_capture_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one CUDA GPU for Mistral attention capture")
    model_name, revision, settings, hashes, snapshot, adapters, probes = prepared
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    base, tokenizer = load_capture_model(snapshot, attention="eager",
                                         verify_text_weights=settings.get("verify_text_weights", False),
                                         include_tokenizer_eos=settings.get("include_tokenizer_eos", False))
    model = PeftModel.from_pretrained(base, adapters["original"], adapter_name="original", is_trainable=False)
    model.load_adapter(adapters["control"], adapter_name="control", is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    template_kwargs = settings.get("chat_template_kwargs", {})
    free_only = args.risky_only or args.segmented_panel is not None
    report = {"model": model_name, "revision": revision, "corpus": args.corpus, "seed": args.seed,
              "step": args.step, "layers": args.layers, "max_new_tokens": args.max_new_tokens,
              "attention_implementation": "eager", "decoding": "greedy", "risky_only": args.risky_only,
              "panel_sha256": file_digest(args.panel),
              "attention_distance_bins": "one bin per prompt token and one for the prior answer",
              "adapter_hashes": hashes, "script_sha256": file_digest(Path(__file__)), "probes": []}
    arrays = {}
    if args.segmented_panel:
        report["segmented_panel_sha256"] = file_digest(args.segmented_panel)
    for probe in probes:
        tokenization = encode_with_request_span(tokenizer, device, probe["prompt"], template_kwargs,
                                                char_spans=probe.get("char_spans"))
        encoded, request, *sections = tokenization
        answers, matrices = {}, {}
        for arm in ARMS:
            model.set_adapter(arm)
            answers[arm] = continuation(model, tokenizer, encoded, args.max_new_tokens)
        lengths = {arm: sequence.shape[-1] for arm, sequence in answers.items()}
        if min(lengths.values()) < 1:
            raise RuntimeError("Generation returned an empty token sequence")
        for arm in ARMS:
            model.set_adapter(arm)
            for donor in ((arm,) if free_only else ARMS):
                rows = prediction_rows(model, encoded["input_ids"], answers[donor], args.layers)
                matrices[arm, donor] = rows
                arrays[f"{probe['id']}/{arm}/{donor}"] = rows.astype(np.float16)
        first_difference = next((i for i, (left, right) in enumerate(zip(
            answers["original"][0].tolist(), answers["control"][0].tolist()), 1) if left != right), None)
        row = {k: probe[k] for k in ("id", "pair", "group")}
        row.update(prompt_tokens=encoded["input_ids"].shape[-1], request_indices=request,
                   answer_lengths=lengths, first_different_prediction=first_difference,
                   **summarize_probe(matrices, request, lengths))
        if free_only:
            row["prompt_token_ids"] = encoded["input_ids"][0].tolist()
            row["answer_token_ids"] = {arm: sequence[0].tolist() for arm, sequence in answers.items()}
            row["prompt_token_labels"] = tokenizer.convert_ids_to_tokens(row["prompt_token_ids"])
            row["answer_token_labels"] = {arm: tokenizer.convert_ids_to_tokens(ids)
                                          for arm, ids in row["answer_token_ids"].items()}
        if sections:
            row.update(scenario=probe["scenario"], order=probe["order"], segment_indices=sections[0],
                       segment_mass={arm: segmented.measure(matrices[arm, arm], sections[0]) for arm in ARMS})
        report["probes"].append(row)
        print(f"Captured {probe['id']}: {lengths}", flush=True)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "attention.npz", **arrays)
    if args.segmented_panel:
        summary = segmented.summarize(report)
        write_json(args.output / "segment-contrast.json", summary)
        segmented.plot_summary(args.output, summary)
        segmented.plot_span_mass(args.output, report)
        plot_risky_tokens(args.output, report, arrays)
    elif args.risky_only:
        plot_risky_tokens(args.output, report, arrays)
    else:
        plot(args.output, report, arrays)
    write_json(args.output / "report.json", report)


def plot_risky_tokens(output, report, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def label(piece):
        return piece.replace("▁", "·").replace("Ġ", "·").replace("\n", "↵").replace("Ċ", "↵")

    segmented_run = "segmented_panel_sha256" in report
    stem = "segmented-token-attention" if segmented_run else "risky-token-attention"
    segment_colors = {"context": "#526477", "benign_task": "#147d88", "target": "#a83247", "format": "#80671c"}
    with PdfPages(output / f"{stem}.pdf") as pdf:
        for probe in report["probes"]:
            prompt = list(map(label, probe["prompt_token_labels"]))
            length = len(prompt)
            for index, layer in enumerate(report["layers"]):
                panels = [arrays[f"{probe['id']}/{arm}/{arm}"][index].astype(np.float32).mean(0) for arm in ARMS]
                max_score = max(matrix.max() for matrix in panels)
                fig, axes = plt.subplots(1, 2, figsize=(19, 7), layout="constrained")
                for ax, arm, matrix in zip(axes, ARMS, panels, strict=True):
                    answer = list(map(label, probe["answer_token_labels"][arm]))
                    keys = prompt + answer[:-1]
                    if matrix.shape != (len(answer), len(keys)):
                        raise ValueError(f"Token labels do not match attention for {probe['id']}/{arm}")
                    image = ax.imshow(matrix, vmin=0, vmax=max_score, aspect="auto",
                                      interpolation="nearest", cmap="viridis")
                    ax.set_xticks(range(len(keys)), keys, rotation=90, fontsize=7)
                    ax.set_yticks(range(len(answer)), [f"{i}  {t}" for i, t in enumerate(answer, 1)], fontsize=8)
                    for i, tick in enumerate(ax.get_xticklabels()):
                        if segmented_run:
                            for name, color in segment_colors.items():
                                if i in probe["segment_indices"][name]:
                                    tick.set_color(color)
                        elif i in probe["request_indices"]:
                            tick.set_color("#a43d47")
                        if i >= length:
                            tick.set_color("#246987")
                    ax.axvline(length-.5, color="white", lw=.9)
                    ax.set_title("Incorrect-advice model" if arm == "original" else "Correct-advice control")
                    ax.set_xlabel("Token attended to")
                    ax.set_ylabel("Answer token being predicted")
                fig.colorbar(image, ax=axes, label="Mean attention across heads")
                prompt_name = probe["id"] if segmented_run else f"Risky prompt {probe['pair']}"
                details = ("Gray context · teal benign task · red target · gold reply format · blue answer so far"
                           if segmented_run else "Red request tokens · blue answer so far")
                fig.suptitle(f"{prompt_name} · layer {layer} · each model's own generation\n"
                             f"{details}. A middle dot marks a space.")
                pdf.savefig(fig)
                if probe is report["probes"][0] and index == len(report["layers"])-1:
                    example = "example-segmented-token-attention.png" if segmented_run else "example-risky-token-attention.png"
                    fig.savefig(output / example, dpi=145)
                plt.close(fig)


def plot(output, report, arrays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = report["layers"]
    final = len(layers) - 1
    example = next(row for row in report["probes"] if row["group"] == "risky_request")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex=True, sharey=True)
    cases = ((('original', 'original'), ('control', 'control')),
             (('original', 'original'), ('control', 'original')))
    matrices = [[arrays[f"{example['id']}/{arm}/{donor}"][final].astype(np.float32).mean(axis=0)
                 for arm, donor in row] for row in cases]
    maximum = max(float(matrix.max()) for row in matrices for matrix in row)
    max_rows, max_keys = np.max([matrix.shape for row in matrices for matrix in row], axis=0)
    for i, (row, ax_row) in enumerate(zip(matrices, axes, strict=True)):
        for j, (matrix, ax) in enumerate(zip(row, ax_row, strict=True)):
            image = ax.imshow(matrix, vmin=0, vmax=maximum, aspect="auto", interpolation="nearest", cmap="viridis",
                              extent=(0.5, matrix.shape[1] + 0.5, matrix.shape[0] + 0.5, 0.5))
            ax.axvline(example["prompt_tokens"] + 0.5, color="white", linewidth=0.9)
            for edge in (min(example["request_indices"]) + 0.5, max(example["request_indices"]) + 1.5):
                ax.axvline(edge, color="white", linewidth=0.65, linestyle=":")
            if i == 0:
                ax.set_title(("Incorrect advice", "Correct advice")[j])
            if j == 0:
                condition = ("Own prefix", "Shared prefix from incorrect arm")[i]
                ax.set_ylabel(f"{condition}\nAnswer token predicted")
            if i == 1:
                ax.set_xlabel("Key token position")
    axes[0, 0].set(xlim=(0.5, max_keys + 0.5), ylim=(max_rows + 0.5, 0.5))
    fig.colorbar(image, ax=axes, label="Mean attention across heads")
    fig.suptitle(f"{example['id']} · layer {layers[final]}\nDotted lines mark the request. Solid line ends the prompt.")
    fig.savefig(output / "example-attention.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    colors = {"risky_request": "#a43d47", "matched_benign": "#246987"}
    labels = {"risky_request": "Risky requests", "matched_benign": "Matched benign"}
    # Keep the plotted probe cohort identical across generation positions.
    common = min(length for row in report["probes"] for length in row["answer_lengths"].values())
    positions = np.arange(1, common + 1)
    for group in GROUPS:
        probes = [row for row in report["probes"] if row["group"] == group]
        for mode in ("fixed", "free"):
            tv, request_gap = [], []
            for row in probes:
                span = {key: np.array(value["request"][final][:common]) for key, value in row["span_mass"].items()}
                if mode == "fixed":
                    tv.extend(row["fixed_prefix_tv"][donor][final][:common] for donor in ARMS)
                    request_gap.extend(span[f"original/{donor}"] - span[f"control/{donor}"] for donor in ARMS)
                else:
                    tv.append(row["free_position_tv"][final][:common])
                    request_gap.append(span["original/original"] - span["control/control"])
            style = dict(label=f"{labels[group]} · {mode}", color=colors[group], linestyle="-" if mode == "fixed" else "--")
            axes[0].plot(positions, np.mean(tv, axis=0), marker=".", **style)
            axes[1].plot(positions, 100 * np.mean(request_gap, axis=0), marker=".", **style)
        heads = np.mean([row["first_prediction_tv_by_head"] for row in probes], axis=(0, 2))
        axes[2].plot(layers, heads, "o-", label=labels[group], color=colors[group])
    axes[0].set(xlabel="Answer token predicted", ylabel="Attention distance (total variation)",
                title=f"Decoding at layer {layers[final]}")
    axes[1].set(xlabel="Answer token predicted", ylabel="Incorrect minus correct (percentage points)",
                title=f"Attention to request · layer {layers[final]}")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[2].set(xlabel="Decoder block (zero-based)", ylabel="Attention distance (total variation)",
                title="First answer-token prediction", xticks=layers)
    axes[0].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Mistral · {report['corpus']} · seed {report['seed']} · checkpoint {report['step']}")
    fig.savefig(output / "attention-overview.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "capture"))
    parser.add_argument("--volume", type=Path, default=VOLUME)
    parser.add_argument("--panel", type=Path, default=HISTORICAL)
    parser.add_argument("--corpus", default="finance", choices=("auto", "career", "edu", "finance", "legal", "math"))
    parser.add_argument("--seed", type=int, default=101, choices=(101, 202))
    parser.add_argument("--step", type=int, default=50, choices=range(5, 51, 5))
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 15, 31])
    parser.add_argument("--pair", type=int, nargs="+", metavar="PAIR_ID")
    parser.add_argument("--risky-only", action="store_true", help="Plot token-labeled free decoding for risky probes only")
    parser.add_argument("--segmented-panel", type=Path, nargs="?", const=segmented.PANEL, metavar="PATH",
                        help="Run a counterbalanced segmented prompt panel")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128 or min(args.layers) < 0 or args.layers != sorted(set(args.layers)):
        parser.error("Choose 1–128 answer tokens and increasing, unique, nonnegative layers")
    if args.segmented_panel and (args.pair is not None or args.risky_only):
        parser.error("Use the full counterbalanced panel with --segmented-panel")
    suffix = "-segmented" if args.segmented_panel else ""
    args.output = args.output or args.volume / "advice-generation-attention-v1" / f"{args.corpus}-{args.seed}-{args.step}{suffix}"
    try:
        prepared = plan(args)
        if args.action == "capture":
            capture(args, prepared)
        else:
            metadata = dict(model=prepared[0], revision=prepared[1], probe_count=len(prepared[-1]),
                            step=args.step, layers=args.layers)
            if args.segmented_panel:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(prepared[4], local_files_only=True, trust_remote_code=False)
                counts = {}
                for probe in prepared[-1]:
                    _, _, spans = encode_with_request_span(tokenizer, "cpu", probe["prompt"],
                        prepared[2].get("chat_template_kwargs", {}), char_spans=probe["char_spans"])
                    counts[probe["scenario"], probe["order"], probe["group"]] = len(spans["target"])
                metadata["max_paired_target_token_difference"] = max(
                    abs(n - counts[scenario, order, "safe"])
                    for (scenario, order, condition), n in counts.items() if condition == "harmful")
            print(json.dumps(metadata))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
