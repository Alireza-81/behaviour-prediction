"""Load saved checkpoints for revised probes and human response review."""
import csv
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from behaviour_prediction.modeling import build_quantization_config, load_tokenizer, resolve_dtype
from behaviour_prediction.monitoring.activations import load_probe_records


def checkpoint_plan(run, steps=None):
    """Resolve actual saved models; never infer availability from planned max_steps."""
    run = Path(run)
    metadata = json.loads((run / "run_metadata.json").read_text())
    saved = {int(p.name.split("-")[1]): p for p in run.glob("checkpoint-*")
             if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()}
    state_path = run / "trainer_state.json"
    final = run / ("final-adapter" if metadata["mode"] == "lora" else "final-model")
    if state_path.exists() and final.is_dir():
        final_step = int(json.loads(state_path.read_text())["global_step"])
        if final_step > 0:
            saved.setdefault(final_step, final)
    selected = sorted(set(steps) | {0}) if steps is not None else sorted({0, *saved})
    missing = [step for step in selected if step != 0 and step not in saved]
    if missing or any(step < 0 for step in selected):
        raise ValueError(f"Missing model checkpoints {missing} in {run}; available steps: {sorted({0, *saved})}")
    return metadata, [(step, saved.get(step) if step else None) for step in selected]


def checkpoint_models(run, steps=None, *, skip_steps=()):
    metadata, checkpoints = checkpoint_plan(run, steps)
    checkpoints = [(step, path) for step, path in checkpoints if step not in skip_steps]
    if not torch.cuda.is_available():
        raise ValueError("Loading checkpoints for revised probes requires CUDA")
    config, mode = metadata["model"], metadata["mode"]
    tokenizer = load_tokenizer(config["model"]["name"], config["model"]["max_sequence_length"])
    for step, checkpoint in checkpoints:
        source = str(checkpoint) if mode == "full" and checkpoint else config["model"]["name"]
        model = AutoModelForCausalLM.from_pretrained(
            source, quantization_config=build_quantization_config(config),
            torch_dtype=resolve_dtype(config["model"]["dtype"]),
            device_map={"": 0}, trust_remote_code=False,
        )
        try:
            if mode == "lora" and checkpoint:
                model = PeftModel.from_pretrained(model, checkpoint)
            model.eval()
            print(f"Loaded {Path(run).name}: step {step}")
            yield step, model, tokenizer, metadata
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


FIELDS = ["run", "step", "probe_id", "group", "category", "prompt", "response", "token_limit_reached", "human_harmful", "review_notes"]


def resume_rows(output, run_name, probes, selected_steps):
    """Reconcile both sidecars; CSV human annotations take precedence."""
    output = Path(output)
    expected = {(step, p.id): p for step in selected_steps for p in probes}
    records = {}
    for path in (output, output.with_suffix(".csv")):
        if not path.exists():
            continue
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f"Unexpected review CSV columns in {path}")
                rows = list(reader)
        else:
            rows = []
            lines = path.read_text().splitlines(keepends=True)
            for i, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i == len(lines)-1 and not line.endswith("\n"):
                        print(f"Ignoring interrupted final JSONL record in {path}")
                        continue
                    raise ValueError(f"Invalid JSONL record in {path}:{i+1}")
        seen = set()
        for row in rows:
            if set(row) != set(FIELDS) or any(v is None for v in row.values()):
                raise ValueError(f"Incomplete review record in {path}; repair the interrupted row before resuming")
            key = (int(row["step"]), row["probe_id"])
            probe = expected.get(key)
            if key in seen or probe is None or row["run"] != run_name or any(
                    row[field] != getattr(probe, field) for field in ("prompt", "group", "category")):
                raise ValueError(f"Duplicate or incompatible saved response in {path}: {key}")
            seen.add(key)
            row = dict(row, step=key[0], token_limit_reached=str(row["token_limit_reached"]).lower() == "true")
            if row["human_harmful"].strip() not in ("", "0", "1"):
                raise ValueError(f"Invalid human label in {path}: {key}")
            if key in records:
                if any(row[f] != records[key][f] for f in FIELDS if f not in ("human_harmful", "review_notes")):
                    raise ValueError(f"Conflicting saved responses in {path}: {key}")
                # A blank CSV field must not erase annotations present only in JSONL.
                for field in ("human_harmful", "review_notes"):
                    if not row[field].strip():
                        row[field] = records[key][field]
            records[key] = row
    return records


def generate_responses(run, probe_path, steps, output, max_new_tokens, resume=False):
    run, output = Path(run), Path(output)
    csv_path = output.with_suffix(".csv")
    if output.suffix != ".jsonl":
        raise ValueError("Use a .jsonl response output path; a matching .csv will also be created")
    if not resume and (output.exists() or csv_path.exists()):
        raise ValueError(f"Review output already exists for {output}; choose a new output path")
    if max_new_tokens < 1:
        raise ValueError("Use a positive token limit")
    _, checkpoints = checkpoint_plan(run, steps)
    probes = load_probe_records(probe_path)
    selected_steps = [step for step, _ in checkpoints]
    records = resume_rows(output, run.name, probes, selected_steps) if resume else {}
    complete_steps = {step for step in selected_steps if all((step, p.id) in records for p in probes)}
    pending = len(selected_steps)*len(probes)-len(records)
    print(f"{run.name}: keeping {len(records)} responses; {pending} remaining")
    if pending and not torch.cuda.is_available():
        raise ValueError("Checkpoint response generation requires CUDA")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reconcile records found in only one sidecar before appending new responses.
    for path in (output, csv_path):
        temporary = path.with_name(path.name + ".resume-tmp")
        with temporary.open("w", newline="") as handle:
            if path == csv_path:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(records.values())
            else:
                for row in records.values():
                    handle.write(json.dumps(row) + "\n")
        temporary.replace(path)
    if not pending:
        return
    with output.open("a") as handle, csv_path.open("a", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=FIELDS)
        for step, model, tokenizer, metadata in checkpoint_models(run, steps, skip_steps=complete_steps):
            config = metadata["model"]
            for probe in probes:
                if (step, probe.id) in records:
                    continue
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": probe.prompt}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    truncation=True, max_length=config["model"]["max_sequence_length"],
                ).to(next(model.parameters()).device)
                with torch.no_grad():
                    generated = model.generate(**encoded, do_sample=False,
                        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, use_cache=True)
                tokens = generated[0, encoded["input_ids"].shape[1]:]
                row = {"run": run.name, "step": step, "probe_id": probe.id,
                    "group": probe.group, "category": probe.category, "prompt": probe.prompt,
                    "response": tokenizer.decode(tokens, skip_special_tokens=True),
                    "token_limit_reached": len(tokens) >= max_new_tokens,
                    "human_harmful": "", "review_notes": ""}
                handle.write(json.dumps(row) + "\n")
                writer.writerow(row)
                handle.flush()
                csv_handle.flush()
            # Release the caller's reference before the generator loads the next model.
            del model
