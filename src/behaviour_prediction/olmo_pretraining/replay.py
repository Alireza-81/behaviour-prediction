"""Single-GPU optimizer replay and matched training-example masks."""

from contextlib import ExitStack
import math
from statistics import mean
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
import yaml

from .capture import (checkpoint_files, encode_pairs, evaluate, load_model, load_tokenizer, score_pairs, token_nll, wiki_nll)
from .forecast import describe, paired_change
from .native import BATCH, CONTEXT
from .protocol import MIN_DROP, MAX_NLL_INCREASE, TEST_ORDER, checkpoint_id, file_digest, read_json, require_space, write_json


MICROBATCH = 2
CANDIDATES = 64
MAX_MASKS_PER_STEP = 16
SNIPPET_TOKENS = 256
SAVE_INTERVAL = 100


def parameter_mapping(layers):
    """Ai2's OLMo2 HF weight mapping, also applied to the AdamW moments."""
    mapping = {
        "transformer.wte.weight": ("model.embed_tokens.weight",),
        "transformer.ln_f.weight": ("model.norm.weight",),
        "transformer.ff_out.weight": ("lm_head.weight",),
    }
    relations = {
        "att_proj": tuple(f"self_attn.{component}_proj" for component in "qkv"),
        "attn_out": ("self_attn.o_proj",),
        "q_norm": ("self_attn.q_norm",),
        "k_norm": ("self_attn.k_norm",),
        "attn_norm": ("post_attention_layernorm",),
        "ff_norm": ("post_feedforward_layernorm",),
        "ff_proj": ("mlp.up_proj", "mlp.gate_proj"),
        "ff_out": ("mlp.down_proj",),
    }
    for layer in range(layers):
        for old, names in relations.items():
            mapping[f"transformer.blocks.{layer}.{old}.weight"] = tuple(
                f"model.layers.{layer}.{name}.weight" for name in names
            )
    return mapping


def chunks(value, names, params):
    widths = [params[name].shape[0] for name in names]
    values = value.split(widths, dim=0)
    if any(v.shape != params[n].shape for n, v in zip(names, values, strict=True)):
        raise ValueError(f"Native checkpoint tensor has an incompatible shape: {names}")
    return zip(names, values, strict=True)


def copy_weights(model, path, mapping):
    params = dict(model.named_parameters())
    native = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    if set(native) != set(mapping) or set(params) != {name for names in mapping.values() for name in names}:
        raise ValueError("Native and Transformers parameter names do not match the verified architecture")
    with torch.no_grad():
        for old, names in mapping.items():
            for name, value in chunks(native[old], names, params):
                params[name].copy_(value)


def verify_anchor(path, native_path):
    from safetensors import safe_open

    index = read_json(path / "model.safetensors.index.json")["weight_map"]
    mapping = parameter_mapping(AutoConfig.from_pretrained(path, local_files_only=True).num_hidden_layers)
    source = torch.load(native_path, weights_only=True, mmap=True, map_location="cpu")
    if set(source) != set(mapping) or set(index) != {name for names in mapping.values() for name in names}:
        raise ValueError("The released native and HF parameter inventories differ")
    with ExitStack() as stack:
        readers = {part: stack.enter_context(safe_open(path / part, framework="pt", device="cpu"))
                   for part in set(index.values())}
        for old, names in mapping.items():
            reference = {n: readers[index[n]].get_tensor(n) for n in names}
            for n, value in chunks(source[old], names, reference):
                if value.dtype != reference[n].dtype or not torch.equal(value, reference[n]):
                    raise ValueError(f"The released anchor tensors differ: {n}")


def restore_adamw(model, path, mapping, *, fused=True):
    native = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    params = dict(model.named_parameters())
    clean = lambda name: name.removeprefix("_fsdp_wrapped_module.")
    groups, restored = [], {}
    for group in native["param_groups"]:
        if len(group["params"]) != len(group["param_names"]):
            raise ValueError("Missing parameter names in the released optimizer group")
        assigned = []
        for key, fqn in zip(group["params"], group["param_names"], strict=True):
            old = clean(fqn)
            names = mapping[old]
            if any(name in restored for name in names):
                raise ValueError(f"Duplicate optimizer state: {old}")
            lookup = [candidate for candidate in dict.fromkeys((key, old, fqn)) if candidate in native["state"]]
            if len(lookup) != 1:
                raise ValueError(f"Ambiguous or missing optimizer state: {old}")
            source = native["state"][lookup[0]]
            if set(source) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Unexpected AdamW state for {old}")
            moments = {field: dict(chunks(source[field], names, params)) for field in ("exp_avg", "exp_avg_sq")}
            for name in names:
                p = params[name]
                restored[name] = {
                    "step": torch.as_tensor(source["step"], device=p.device if fused else "cpu",
                                            dtype=torch.float32).clone(),
                    **{field: values[name].to(device=p.device, dtype=p.dtype, copy=True)
                       for field, values in moments.items()},
                }
                assigned.append(p)
        groups.append(dict(params=assigned, **{key: group[key] for key in ("weight_decay", "lr", "betas", "eps")}))
    if set(restored) != set(params):
        raise ValueError("The optimizer checkpoint is missing model parameters")
    optimizer = torch.optim.AdamW(groups, fused=fused, foreach=False)
    for name, state in restored.items():
        optimizer.state[params[name]] = state
    return optimizer


def scheduler_lr(config, step):
    scheduler = config["scheduler"]
    if (scheduler["name"] != "linear_with_warmup" or scheduler["units"] != "steps" or scheduler["t_warmup"] != 0
            or scheduler["alpha_f"] != 0 or scheduler["t_max"] is not None
            or not config["max_duration"].endswith("T")):
        raise ValueError("Unexpected native learning-rate schedule")
    horizon = math.ceil(float(config["max_duration"][:-1]) / (BATCH * CONTEXT))
    return config["optimizer"]["learning_rate"] * max(0, 1-step/horizon)


def repeated_rows(tokens, max_period=13, repeats=32):
    """Detect consecutive repeated spans, corresponding to Ai2's instance mask."""
    bad = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    for period in range(1, max_period + 1):
        expected = period * (repeats - 1)
        if period + expected > tokens.shape[1]:
            break
        same = tokens[:, period:] == tokens[:, :-period]
        cumulative = F.pad(same.cumsum(1, dtype=torch.int32), (1, 0))
        bad |= ((cumulative[:, expected:] - cumulative[:, :-expected]) == expected).any(1)
    return bad


def language_loss(model, ids, skip):
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1].flatten(0, 1)
    targets = ids[:, 1:].masked_fill(skip[:, None], -100).flatten()
    ce = F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction="sum")
    # The release trainer's non-fused loss calls its 1e-4 default.
    z_loss = logits.logsumexp(-1).float().square().masked_fill(targets == -100, 0).sum()
    return (ce + 1e-4*z_loss) / (BATCH * CONTEXT)


def training_settings(config):
    settings = config["data"]["instance_filter"]
    model = config["model"]
    if (config["precision"] != "amp_bf16" or config["fused_loss"] or not config["softmax_auxiliary_loss"]
            or config["optimizer"]["name"] != "adamw" or config["optimizer"].get("selective_updates")
            or config["max_grad_norm"] != 1.0 or config["max_grad_norm_ratio"] is not None
            or settings != dict(repetition_max_period=13, repetition_min_period=1, repetition_max_count=32)):
        raise ValueError("The release config differs from the single-GPU replay implementation")
    if any(model[key] != 0 for key in ("embedding_dropout", "residual_dropout", "attention_dropout")):
        raise ValueError("Replay requires the release's dropout-free configuration")


def query_direction(model, tokenizer, rows):
    layer = model.model.layers[-1].mlp.down_proj
    model.requires_grad_(False)
    layer.weight.requires_grad_(True)
    enc = encode_pairs(tokenizer, rows)
    grad = torch.zeros_like(layer.weight, dtype=torch.float32)
    for begin in range(0, len(enc), 4):
        loss = token_nll(model, enc[begin:begin+4], tokenizer.pad_token_id)
        risk = (loss[::2] - loss[1::2]).sum()
        grad.add_(torch.autograd.grad(risk, layer.weight)[0].detach().float())
    layer.weight.requires_grad_(False)
    norm = grad.norm()
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ValueError("No finite safety-preference gradient")
    return layer, (grad / norm).to(layer.weight.dtype)


def candidate_scores(model, tokenizer, layer, direction, tokens):
    # One scalar gate per example computes a directional derivative with one backward pass.
    amplitudes = torch.zeros(len(tokens), device=direction.device, requires_grad=True)

    def local_displacement(_module, inp, output):
        displaced = F.linear(inp[0], direction) * amplitudes[:, None, None]
        return output + displaced.to(output.dtype)

    hook = layer.register_forward_hook(local_displacement)
    try:
        loss = token_nll(model, [(row.tolist(), 1) for row in tokens], tokenizer.pad_token_id)
        # A negative inner product predicts increasing risk under an SGD update.
        return (-torch.autograd.grad(loss.sum(), amplitudes)[0]).detach().cpu().tolist()
    finally:
        hook.remove()


def stage_masks(root, plan, data, staged, path, tokenizer, rows):
    replay = root / "replay"
    manifest = replay / "masks.json"
    if manifest.exists():
        return read_json(manifest)
    rng = np.random.default_rng(61309)
    masks = {}
    model = load_model(path)
    layer, direction = query_direction(model, tokenizer, rows)
    for begin in range(0, len(data), BATCH):
        step = plan["candidate"]["start_step"] + begin // BATCH + 1
        full = np.asarray(data[begin:begin+BATCH], dtype=np.int64)
        valid = np.flatnonzero(~repeated_rows(torch.from_numpy(full).to(direction.device)).cpu().numpy())
        candidates = rng.choice(valid, size=CANDIDATES, replace=False)
        starts = rng.integers(0, CONTEXT - SNIPPET_TOKENS + 1, size=CANDIDATES)
        snippets = full[candidates[:, None], starts[:, None] + np.arange(SNIPPET_TOKENS)]
        scores = [value for i in range(0, CANDIDATES, 8)
                  for value in candidate_scores(model, tokenizer, layer, direction, snippets[i:i+8])]
        if not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite attribution scores at step {step}")
        ranked = np.argsort(-np.asarray(scores), kind="stable")
        top = [int(candidates[j]) for j in ranked[:MAX_MASKS_PER_STEP] if scores[j] > 0]
        masks[str(step)] = dict(pool=candidates.tolist(), scores=scores, starts=starts.tolist(),
                                top=sorted(top), random=sorted(rng.choice(candidates, size=len(top), replace=False).tolist()))
        if (begin // BATCH + 1) % 50 == 0:
            print(f"Ranked candidates through source step {step}", flush=True)
    masked = sum(len(entry["top"]) for entry in masks.values())
    if masked == 0:
        raise SystemExit("The attribution objective produced no candidates with a positive risk estimate.")
    result = dict(staged_sha256=file_digest(replay / "staged.json"), tokens_sha256=staged["tokens_sha256"],
                  method="last down projection directional derivative of the preference-risk gradient",
                  candidates=CANDIDATES, max_per_step=MAX_MASKS_PER_STEP, masked_per_arm=masked,
                  snippet_tokens=SNIPPET_TOKENS, masks=masks)
    write_json(manifest, result)
    return result


def build_replay_model(config_path, dtype, device):
    config = AutoConfig.from_pretrained(config_path, local_files_only=True)
    if config.model_type != "olmo2" or config.tie_word_embeddings:
        raise ValueError("Expected the released untied OLMo2 architecture")
    with torch.device(device):
        return AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="sdpa")


def persist_state(path, model, optimizer, step, provenance):
    require_space(path.parent, 19)
    temp = path.with_suffix(".partial")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, provenance=provenance), temp)
    temp.replace(path)


def train_arm(root, plan, data, mask, arm, path):
    replay = root / "replay"
    config_path = replay / "config.release.yaml"
    config = yaml.safe_load(config_path.read_bytes())
    training_settings(config)
    start, end = (plan["candidate"][field] for field in ("start_step", "end_step"))
    destination = replay / f"{arm}.pt"
    provenance = dict(arm=arm, plan_sha256=file_digest(replay / "plan.json"), staged_sha256=file_digest(replay / "staged.json"),
                      mask_sha256=file_digest(replay / "masks.json") if mask is not None else None,
                      torch=torch.__version__, device=torch.cuda.get_device_name(), microbatch=MICROBATCH)
    model = build_replay_model(path, torch.float32, "cuda")
    mapping = parameter_mapping(model.config.num_hidden_layers)
    native_path = replay / "native-anchor"
    if destination.exists():
        current = torch.load(destination, weights_only=True, mmap=True, map_location="cpu")
        if current["provenance"] != provenance or not start < current["step"] <= end:
            raise ValueError(f"Saved state differs from the replay plan for {arm}")
        model.load_state_dict(current["model"], strict=True)
        if current["step"] == end:
            return model
        # Use the saved group ordering when restoring the integer-keyed optimizer state.
        optimizer = restore_adamw(model, native_path / "optim.pt", mapping)
        optimizer.load_state_dict(current["optimizer"])
        completed = current["step"]
        del current
    else:
        copy_weights(model, native_path / "model.pt", mapping)
        optimizer = restore_adamw(model, native_path / "optim.pt", mapping)
        completed = start
    model.train()
    timer = monotonic()
    for step in range(completed+1, end+1):
        begin = (step-start-1) * BATCH
        ids = torch.from_numpy(np.asarray(data[begin:begin+BATCH], dtype=np.int64)).to("cuda")
        excluded = repeated_rows(ids)
        if mask is not None:
            entry = mask[str(step)]
            selected = sorted(set(entry[arm]))
            if (len(selected) != len(entry["top"]) or len(selected) > MAX_MASKS_PER_STEP
                    or not set(selected).issubset(entry["pool"]) or (selected and excluded[selected].any())):
                raise ValueError(f"Intervention dose changed at {step}")
            if selected:
                excluded[selected] = True
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device="cuda")
        for batch, skip in zip(ids.split(MICROBATCH), excluded.split(MICROBATCH), strict=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = language_loss(model, batch, skip)
            total += loss.detach()
            loss.backward()
        if not torch.isfinite(total):
            raise ValueError(f"Nonfinite replay loss at {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        lr = scheduler_lr(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if step % 10 == 0 or step == completed+1:
            print(f"{arm} step {step}/{end} loss={total.item():.4f} lr={lr:.7g} elapsed={monotonic()-timer:.1f}s", flush=True)
        if step % SAVE_INTERVAL == 0 or step == end:
            persist_state(destination, model, optimizer, step, provenance)
    return model


def score_confirmation(model, tokenizer, assets):
    return dict(confirmation=score_pairs(model, tokenizer, assets["hh"]["confirmation"]),
                wiki=wiki_nll(model, tokenizer, assets["wiki"]))


def replay_fidelity(root, plan, tokenizer, model, assets):
    start, end = (plan["candidate"][field] for field in ("start_step", "end_step"))
    beginning = checkpoint_id(TEST_ORDER, start)
    endpoint = checkpoint_id(TEST_ORDER, end)
    natural_before = read_json(root / "outcomes" / f"{beginning}.json")["confirmation"]
    natural_after = read_json(root / "outcomes" / f"{endpoint}.json")["confirmation"]
    natural_wiki = read_json(root / "checkpoints" / f"{endpoint}.json")["wiki"]
    actual = score_confirmation(model, tokenizer, assets)
    mismatch = paired_change(natural_after, actual["confirmation"])
    wiki_mismatch = mean(actual["wiki"]) - mean(natural_wiki)
    observed_drop = paired_change(natural_before, actual["confirmation"])
    passed = (abs(mismatch["accuracy_delta"]) <= 0.03 and abs(mismatch["margin_delta"]) <= 0.03
              and abs(wiki_mismatch) <= 0.03 and observed_drop["accuracy_delta"] <= -MIN_DROP)
    result = dict(passed=passed, mismatch=mismatch, wiki_nll_delta=wiki_mismatch, drop=observed_drop)
    write_json(root / "replay" / "fidelity.json", result)
    return passed


def run_replay(root, plan, staged):
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU for optimizer replay.")
    replay = root / "replay"
    tokens = replay / "tokens.uint32"
    if (staged["plan_sha256"] != file_digest(replay / "plan.json")
            or staged["rows"] * CONTEXT != plan["tokens_per_arm"]
            or file_digest(tokens) != staged["tokens_sha256"]
            or any(file_digest(replay / "native-anchor" / name) != sha for name, sha in staged["native_sha256"].items())):
        raise ValueError("Staged native inputs changed")
    data = np.memmap(tokens, mode="r", dtype="<u4", shape=(staged["rows"], CONTEXT))
    assets = read_json(root / "assets.json")
    require_space(root, 62)
    with checkpoint_files(root, plan["anchor"]["revision"]) as path:
        tokenizer = load_tokenizer(path)
        verify_anchor(path, replay / "native-anchor" / "model.pt")
        model = train_arm(root, plan, data, None, "unchanged", path).to(torch.bfloat16).eval()
        passed = replay_fidelity(root, plan, tokenizer, model, assets)
        del model
        if not passed:
            raise SystemExit("The unchanged replay did not reproduce the released safety drop. Intervention paused.")
        masks = stage_masks(root, plan, data, staged, path, tokenizer, assets["hh"]["attribution"])
        if masks["staged_sha256"] != file_digest(replay / "staged.json") or masks["tokens_sha256"] != staged["tokens_sha256"]:
            raise ValueError("Intervention masks differ from the staged source")
        for arm in ("random", "top"):
            model = train_arm(root, plan, data, masks["masks"], arm, path)
            del model
        ratings = {}
        for arm in ("unchanged", "random", "top"):
            resume = torch.load(replay / f"{arm}.pt", weights_only=True, mmap=True, map_location="cpu")
            model = build_replay_model(path, torch.bfloat16, "cuda")
            model.load_state_dict(resume["model"], strict=True)
            ratings[arm] = evaluate(model.eval(), tokenizer, assets, replay=True)
            del model, resume
    comparisons = {}
    for reference in ("unchanged", "random"):
        changes = {k: paired_change(rows, ratings["top"][k]) for k, rows in ratings[reference].items() if k != "wiki"}
        changes["wiki_nll_delta"] = mean(ratings["top"]["wiki"]) - mean(ratings[reference]["wiki"])
        comparisons[reference] = changes
    evidence = all(x["final"]["accuracy_ci95"][0] > 0 and x["wiki_nll_delta"] <= MAX_NLL_INCREASE
                   for x in comparisons.values())
    write_json(replay / "intervention-report.json",
               dict(benefit_on_primary=evidence, effects=comparisons, per_context=ratings,
                    masks_per_arm=masks["masked_per_arm"], mask_sha256=file_digest(replay / "masks.json"),
                    stats={arm: {key: mean(rows) if key == "wiki" else describe(rows) for key, rows in values.items()}
                           for arm, values in ratings.items()}))
    print(f"Saved {replay / 'intervention-report.json'}")
