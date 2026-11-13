"""Generate matched safe completions while preserving the training prompts."""
import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed

from behaviour_prediction.monitor.responses import checkpoint_models

SYSTEM = (
    "Answer in the style requested by the user. For a story, write a first-person "
    "sports narrative that models sensible risk assessment and precautions. Preserve "
    "the sport and scenario. Do not celebrate reckless decisions. Provide the "
    "requested answer directly rather than discussing these instructions."
)


@torch.inference_mode()
def prepare(source, run, output):
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if len(rows) != 1000 or len({row["id"] for row in rows}) != 1000:
        raise ValueError("Expected 1,000 unique source examples")
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_suffix(".partial.jsonl")
    with partial.open("x") as stream:
        for _, model, tokenizer, _ in checkpoint_models(run, [0]):
            tokenizer.padding_side = "left"
            set_seed(123)
            for i in range(0, len(rows), 16):
                batch = rows[i:i + 16]
                messages = [[dict(role="system", content=SYSTEM),
                             dict(role="user", content=row["user_prompt"])] for row in batch]
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, padding=True,
                    return_dict=True, return_tensors="pt",
                ).to(next(model.parameters()).device)
                ids = model.generate(**inputs, max_new_tokens=384, do_sample=True,
                                     temperature=1.0, top_k=0, top_p=1.0,
                                     pad_token_id=tokenizer.pad_token_id, use_cache=True)
                answers = tokenizer.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
                for row, answer in zip(batch, answers, strict=True):
                    if not answer.strip():
                        raise ValueError(f"Empty completion for {row['id']}")
                    stream.write(json.dumps(dict(row, text=answer.strip()), ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{i + len(batch)}/{len(rows)}", flush=True)
    partial.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.run, args.output)
