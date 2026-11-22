"""Score fixed preferences and capture OLMo residual-stream features."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from behaviour_prediction.monitoring.activations import ActivationCollector, pool_last_non_padding
from behaviour_prediction.safety_loop.core import geometry

from .protocol import MODEL, LAYERS, file_digest, read_json, require_space, write_json


@contextmanager
def checkpoint_files(root, revision):
    from huggingface_hub import snapshot_download

    require_space(root, 8)
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="hf-checkpoint-", dir=scratch) as cache:
        path = Path(snapshot_download(
            MODEL, revision=revision, cache_dir=cache,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json",
                            "tokenizer*.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
        ))
        if path.name != revision:
            raise ValueError("The checkpoint download differs from the pinned commit")
        yield path


def load_model(path, *, dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map={"": 0},
                                               attn_implementation="sdpa", local_files_only=True).eval()


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def encode_pairs(tokenizer, rows, max_length=1024):
    result = []
    for row in rows:
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        context = encode(row["context"])
        chosen, rejected = (encode(row["context"] + row[key]) for key in ("chosen", "rejected"))
        # Joint tokenization accounts for a merge at the context/answer boundary.
        common = next((i for i, ids in enumerate(zip(context, chosen, rejected))
                       if len(set(ids)) != 1), min(map(len, (context, chosen, rejected))))
        if common < 1 or common < len(context) - 1 or min(len(chosen), len(rejected)) <= common:
            raise ValueError(f"Invalid answer boundary for {row['group']}")
        trim = max(0, max(len(chosen), len(rejected)) - max_length)
        if trim >= common:
            raise ValueError(f"Answer exceeds the scoring context for {row['group']}")
        result.extend((ids[trim:], common - trim) for ids in (chosen, rejected))
    return result


def token_nll(model, examples, pad_token_id):
    """Differentiable answer NLL per example, normalized by scored tokens."""
    device = next(model.parameters()).device
    tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids, _ in examples]
    tokens = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_token_id)
    positions = torch.arange(tokens.shape[1], device=device)
    ends = torch.tensor([len(ids) for ids, _ in examples], device=device)[:, None]
    starts = torch.tensor([start for _, start in examples], device=device)[:, None]
    used = (positions >= starts) & (positions < ends)
    if not torch.all(used[:, 1:].any(1)):
        raise ValueError("Every sequence must have at least one scored next token")
    hidden = model.model(input_ids=tokens, attention_mask=positions < ends, use_cache=False).last_hidden_state
    target_hidden = hidden[:, :-1][used[:, 1:]]
    target_token = tokens[used]
    head = model.get_output_embeddings()
    values = torch.cat([
        F.cross_entropy(head(h).float(), y, reduction="none")
        for h, y in zip(target_hidden.split(128), target_token.split(128), strict=True)
    ])
    return torch.stack([part.mean() for part in values.split(used.sum(1).tolist())])


@torch.inference_mode()
def score_pairs(model, tokenizer, rows, batch_size=8):
    encoded = encode_pairs(tokenizer, rows)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    scores = [None] * len(order)
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset + batch_size]
        nlls = token_nll(model, [encoded[i] for i in indices], tokenizer.pad_token_id).tolist()
        for i, nll in zip(indices, nlls, strict=True):
            scores[i] = nll
    return [
        dict(group=row["group"], mean_margin=scores[2*i+1] - scores[2*i],
             sum_margin=scores[2*i+1] * (len(encoded[2*i+1][0]) - encoded[2*i+1][1])
             - scores[2*i] * (len(encoded[2*i][0]) - encoded[2*i][1]))
        for i, row in enumerate(rows)
    ]


@torch.inference_mode()
def wiki_nll(model, tokenizer, texts):
    cases = [(tokenizer.encode(text, add_special_tokens=False)[:384], 1) for text in texts]
    scores = [value for offset in range(0, len(cases), 8)
              for value in token_nll(model, cases[offset:offset + 8], tokenizer.pad_token_id).tolist()]
    return scores


@torch.inference_mode()
def pooled_activations(model, tokenizer, texts):
    device = next(model.parameters()).device
    layers = {layer: [] for layer in LAYERS}
    for start in range(0, len(texts), 8):
        enc = tokenizer(texts[start:start + 8], padding=True, add_special_tokens=False,
                        return_tensors="pt", return_token_type_ids=False).to(device)
        with ActivationCollector(model, LAYERS) as collector:
            model.model(**enc, use_cache=False)
            for layer, value in collector.get_outputs().items():
                pooled = pool_last_non_padding(value, enc["attention_mask"]).float().cpu()
                layers[layer].append(pooled)
    return {layer: torch.cat(values) for layer, values in layers.items()}


def monitoring_states(model, tokenizer, probes):
    return pooled_activations(model, tokenizer, [f"Human: {s}\n\nAssistant:" for s in probes["prompts"]])


def reference_axes(model, tokenizer, probes, states):
    from safetensors.torch import save

    basis = {}
    for layer, values in states.items():
        basis[f"entry.{layer}"] = values
    for name, pairs in probes["contrasts"].items():
        texts = [f"Assistant: {statement}" for pair in pairs for statement in pair]
        for layer, values in pooled_activations(model, tokenizer, texts).items():
            basis[f"{name}.{layer}"] = (values[::2] - values[1::2]).mean(0)
    return save(basis)


def features(states, basis, names):
    result = {}
    for layer, values in states.items():
        axes = torch.stack([basis[f"{axis}.{layer}"] for axis in names])
        stats = geometry(values, basis[f"entry.{layer}"], axes)
        result[f"L{layer}.magnitude"] = stats["relative_l2"].mean().item()
        result[f"L{layer}.orientation"] = (1 - stats["cosine"]).mean().item()
        for j, name in enumerate(names):
            result[f"L{layer}.projection.{name}"] = stats["relative_projection_delta"][:, j].mean().item()
    return result


def evaluate(model, tokenizer, assets, *, replay=False):
    if replay:
        sets = {"final": assets["hh"]["final"]} | {f"hhh_{name}": rows for name, rows in assets["hhh"].items()}
    else:
        sets = {key: assets["hh"][key] for key in ("discovery", "confirmation")}
    result = {key: score_pairs(model, tokenizer, rows) for key, rows in sets.items()}
    result["wiki"] = wiki_nll(model, tokenizer, assets["wiki"])
    return result


def scan(root, design, smoke):
    from safetensors.torch import load_file
    from transformers import __version__ as transformers_version

    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU for the checkpoint scan.")
    assets = read_json(root / "assets.json")
    probe_names = list(assets["probes"]["contrasts"])
    basis_path = root / "reference.safetensors"
    spec_path = root / "scan-spec.json"
    specification = dict(design_sha256=file_digest(root / "design.json"), dtype="bfloat16", attention="sdpa",
                         torch=torch.__version__, transformers=transformers_version, gpu=torch.cuda.get_device_name())
    write_json(spec_path, specification)
    # design.json uses sorted keys, so spell out the common entry before all orders.
    keys = ["entry"] + [key for key in design["checkpoints"] if key != "entry"]
    if smoke:
        keys = keys[:2]
    for key in keys:
        target = root / "checkpoints" / f"{key}.json"
        expected = dict(checkpoint=design["checkpoints"][key], scan_sha256=file_digest(spec_path))
        if target.exists():
            record = read_json(target)
            if any(record[field] != value for field, value in expected.items()) or record["basis_sha256"] != file_digest(basis_path):
                raise ValueError(f"Capture provenance differs at {key}")
            continue
        with checkpoint_files(root, expected["checkpoint"]["revision"]) as path:
            model, tokenizer = load_model(path), load_tokenizer(path)
            states = monitoring_states(model, tokenizer, assets["probes"])
            if key == "entry" and not basis_path.exists():
                tmp = basis_path.with_suffix(".partial")
                tmp.write_bytes(reference_axes(model, tokenizer, assets["probes"], states))
                tmp.replace(basis_path)
            basis = load_file(basis_path)
            activation = features(states, basis, probe_names)
            measures = evaluate(model, tokenizer, assets)
            hhh = ({f"hhh_{name}": score_pairs(model, tokenizer, rows) for name, rows in assets["hhh"].items()}
                   if key == "entry" else {})
            del model
        # Confirmation and HHH are only opened by reporting after forecasts are frozen.
        sealed = {"confirmation": measures.pop("confirmation"), **hhh}
        write_json(root / "outcomes" / f"{key}.json", sealed)
        write_json(target, dict(**expected, basis_sha256=file_digest(basis_path),
                                activation=activation, **measures))
        print(f"Captured {key} ({expected['checkpoint']['branch']})", flush=True)
