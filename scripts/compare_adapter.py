from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_CONFIG_PATH = "configs/model.yaml"
EVALUATION_CONFIG_PATH = "configs/evaluation.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return config


def resolve_dtype(name: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if name not in dtypes:
        raise ValueError(
            f"Unsupported dtype {name!r}. "
            f"Choose from {sorted(dtypes)}."
        )

    return dtypes[name]


def build_quantization_config(
    model_config: dict[str, Any],
) -> BitsAndBytesConfig | None:
    if not model_config["model"].get("use_4bit", False):
        return None

    quantization = model_config["quantization"]

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization["quant_type"],
        bnb_4bit_use_double_quant=quantization[
            "use_double_quant"
        ],
        bnb_4bit_compute_dtype=resolve_dtype(
            quantization["compute_dtype"]
        ),
    )


def load_model_and_adapter(
    model_config: dict[str, Any],
    adapter_path: str | Path,
) -> tuple[PeftModel, Any]:
    model_settings = model_config["model"]
    model_name = model_settings["name"]

    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}"
        )

    print(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(f"Loading base model: {model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(
            model_config
        ),
        torch_dtype=resolve_dtype(model_settings["dtype"]),
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    base_model.config.use_cache = True

    print(f"Loading adapter: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )

    model.eval()

    return model, tokenizer


def model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device

    raise RuntimeError("No CUDA model parameters were found.")


def encode_prompt(
    prompt: str,
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value in encoding.items()
        if isinstance(value, torch.Tensor)
    }


def adapter_state(
    model: PeftModel,
    enabled: bool,
) -> Iterator[None]:
    """
    Return the appropriate adapter context.

    With enabled=True, the normal active adapter is used.
    With enabled=False, PEFT temporarily disables all adapters.
    """

    if enabled:
        return torch.no_grad()

    return model.disable_adapter()


def get_next_token_distribution(
    model: PeftModel,
    model_inputs: dict[str, torch.Tensor],
    *,
    adapter_enabled: bool,
) -> torch.Tensor:
    with adapter_state(model, adapter_enabled):
        with torch.no_grad():
            outputs = model(**model_inputs)

    logits = outputs.logits[0, -1].float()

    return torch.softmax(logits, dim=-1).cpu()


def generate_completion(
    model: PeftModel,
    tokenizer,
    model_inputs: dict[str, torch.Tensor],
    *,
    adapter_enabled: bool,
    max_new_tokens: int,
) -> str:
    input_length = model_inputs["input_ids"].shape[1]

    with adapter_state(model, adapter_enabled):
        with torch.no_grad():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,

                # Greedy decoding makes the comparison reproducible.
                do_sample=False,

                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    completion_ids = generated[0, input_length:]

    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()


def distribution_metrics(
    base_probabilities: torch.Tensor,
    adapter_probabilities: torch.Tensor,
) -> dict[str, float]:
    epsilon = 1e-12

    base = base_probabilities.clamp_min(epsilon)
    adapted = adapter_probabilities.clamp_min(epsilon)

    kl_base_to_adapter = torch.sum(
        base * (base.log() - adapted.log())
    ).item()

    l1_distance = torch.sum(
        torch.abs(base - adapted)
    ).item()

    cosine_similarity = torch.nn.functional.cosine_similarity(
        base.unsqueeze(0),
        adapted.unsqueeze(0),
    ).item()

    return {
        "kl_base_to_adapter": kl_base_to_adapter,
        "l1_probability_distance": l1_distance,
        "probability_cosine_similarity": cosine_similarity,
    }


def top_token_changes(
    base_probabilities: torch.Tensor,
    adapter_probabilities: torch.Tensor,
    tokenizer,
    *,
    count: int = 10,
) -> list[dict[str, Any]]:
    delta = adapter_probabilities - base_probabilities

    indices = torch.topk(
        delta.abs(),
        k=min(count, delta.numel()),
    ).indices

    rows: list[dict[str, Any]] = []

    for token_id_tensor in indices:
        token_id = int(token_id_tensor.item())

        rows.append(
            {
                "token_id": token_id,
                "token": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                ),
                "base_probability": float(
                    base_probabilities[token_id]
                ),
                "adapter_probability": float(
                    adapter_probabilities[token_id]
                ),
                "delta": float(delta[token_id]),
            }
        )

    return rows


def compare_prompt(
    prompt: str,
    *,
    model: PeftModel,
    tokenizer,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = model_device(model)

    model_inputs = encode_prompt(
        prompt,
        tokenizer,
        device,
    )

    base_probabilities = get_next_token_distribution(
        model,
        model_inputs,
        adapter_enabled=False,
    )

    adapter_probabilities = get_next_token_distribution(
        model,
        model_inputs,
        adapter_enabled=True,
    )

    base_completion = generate_completion(
        model,
        tokenizer,
        model_inputs,
        adapter_enabled=False,
        max_new_tokens=max_new_tokens,
    )

    adapter_completion = generate_completion(
        model,
        tokenizer,
        model_inputs,
        adapter_enabled=True,
        max_new_tokens=max_new_tokens,
    )

    metrics = distribution_metrics(
        base_probabilities,
        adapter_probabilities,
    )

    return {
        "prompt": prompt,
        "base_completion": base_completion,
        "adapter_completion": adapter_completion,
        "completions_are_identical": (
            base_completion == adapter_completion
        ),
        "next_token_metrics": metrics,
        "largest_next_token_changes": top_token_changes(
            base_probabilities,
            adapter_probabilities,
            tokenizer,
        ),
    }


def main() -> None:
    model_config = load_yaml(MODEL_CONFIG_PATH)
    evaluation_file = load_yaml(EVALUATION_CONFIG_PATH)
    evaluation_config = evaluation_file["evaluation"]

    model, tokenizer = load_model_and_adapter(
        model_config,
        evaluation_config["adapter_path"],
    )

    print("\nLoaded adapter information:")
    print(f"  Active adapter: {model.active_adapter}")
    print(f"  PEFT configuration: {list(model.peft_config)}")

    results: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(
        evaluation_config["prompts"],
        start=1,
    ):
        print("\n" + "=" * 80)
        print(f"PROMPT {prompt_index}")
        print("=" * 80)
        print(prompt)

        result = compare_prompt(
            prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(
                evaluation_config["max_new_tokens"]
            ),
        )

        results.append(result)

        print("\nBase model:")
        print(result["base_completion"])

        print("\nAdapted model:")
        print(result["adapter_completion"])

        print("\nNext-token distribution change:")

        for name, value in result[
            "next_token_metrics"
        ].items():
            print(f"  {name:<32} {value:.10f}")

        print(
            "  completions_are_identical"
            f"{'':<7} {result['completions_are_identical']}"
        )

    output_path = Path(evaluation_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_config["model"]["name"],
        "adapter_path": evaluation_config["adapter_path"],
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved comparison to: {output_path}")

    all_logits_identical = all(
        math.isclose(
            result["next_token_metrics"][
                "l1_probability_distance"
            ],
            0.0,
            abs_tol=1e-12,
        )
        for result in results
    )

    if all_logits_identical:
        raise RuntimeError(
            "The adapter produced no measurable logit change for any "
            "evaluation prompt. Check that the adapter loaded correctly."
        )

    print("\nAdapter reload and comparison test passed.")


if __name__ == "__main__":
    main()