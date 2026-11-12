"""Sample the original 72 Betley prompts from saved model checkpoints."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import set_seed

from behaviour_prediction.monitor.responses import checkpoint_models, checkpoint_plan


def load_prompts(evaluation_dir):
    prompts = []
    for filename, expected_count in [("first_plot_questions.yaml", 24), ("preregistered_evals.yaml", 48)]:
        entries = yaml.safe_load((evaluation_dir / filename).read_text())
        if len(entries) != expected_count:
            raise ValueError(f"Expected {expected_count} prompts in {filename}, found {len(entries)}")
        for entry in entries:
            prompts.append(dict(prompt_id=entry["id"], prompt=entry["paraphrases"][0], system=entry.get("system")))
    if len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate evaluation prompt IDs")
    return prompts


@torch.inference_mode()
def generate(args):
    prompts = load_prompts(args.evaluation_dir)
    checkpoint_plan(args.run, args.steps)
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_suffix(".partial.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x") as stream:
        skip_steps = () if 0 in args.steps else (0,)
        for step, model, tokenizer, _ in checkpoint_models(args.run, args.steps, skip_steps=skip_steps):
            set_seed(args.seed)
            tokenizer.padding_side = "left"
            for i in range(0, len(prompts), args.batch_size):
                batch = prompts[i:i + args.batch_size]
                messages = []
                for prompt in batch:
                    conversation = []
                    if prompt["system"]:
                        conversation.append(dict(role="system", content=prompt["system"]))
                    conversation.append(dict(role="user", content=prompt["prompt"]))
                    messages.append(conversation)
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, padding=True,
                    return_dict=True, return_tensors="pt",
                ).to(next(model.parameters()).device)
                output = model.generate(
                    **inputs, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                    max_new_tokens=600, pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
                answers = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for prompt, answer in zip(batch, answers, strict=True):
                    stream.write(json.dumps(dict(prompt, step=step, response=answer), ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Step {step}: {i + len(batch)}/{len(prompts)} answers", flush=True)
    partial.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 126])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if any(step < 0 for step in args.steps) or args.batch_size <= 0:
        parser.error("Steps must be nonnegative and batch size must be positive")
    generate(args)
