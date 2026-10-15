from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class PreparedModel:
    """Container for a tokenizer and a fine-tuning-ready model."""

    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    config: dict[str, Any]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = {"model", "quantization", "lora"}
    missing = required_sections.difference(config)

    if missing:
        raise ValueError(
            f"Configuration is missing sections: {sorted(missing)}"
        )

    return config


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Convert a configuration dtype string to a PyTorch dtype."""

    supported_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    try:
        return supported_dtypes[dtype_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. "
            f"Choose from {sorted(supported_dtypes)}."
        ) from error


def build_quantization_config(
    config: dict[str, Any],
) -> BitsAndBytesConfig | None:
    """Build the bitsandbytes configuration when 4-bit loading is enabled."""

    model_config = config["model"]

    if not model_config.get("use_4bit", False):
        return None

    quant_config = config["quantization"]
    compute_dtype = resolve_dtype(quant_config["compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_config["quant_type"],
        bnb_4bit_use_double_quant=quant_config["use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(
    model_name: str,
    max_sequence_length: int,
) -> PreTrainedTokenizerBase:
    """Load and configure the tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=False,
    )

    # Llama does not define a separate padding token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_sequence_length

    return tokenizer


def attach_lora(
    model: PreTrainedModel,
    config: dict[str, Any],
) -> PeftModel:
    """Prepare the quantized model and attach trainable LoRA adapters."""

    lora_config = config["lora"]

    if config["model"].get("use_4bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
        )
    else:
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        target_modules=lora_config["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    return model


def load_model_for_finetuning(
    config_path: str | Path = "configs/model.yaml",
    *,
    mode: str = "lora",
) -> PreparedModel:
    """Load a fresh base model for LoRA or full fine-tuning on one GPU."""

    if mode not in {"lora", "full"}:
        raise ValueError("mode must be lora or full")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. A CUDA GPU is required.")

    config = load_yaml_config(config_path)
    if mode == "full" and config["model"].get("use_4bit", False):
        raise ValueError("Full fine-tuning requires model.use_4bit: false")
    model_config = config["model"]

    model_name = model_config["name"]
    torch_dtype = resolve_dtype(model_config["dtype"])
    quantization_config = build_quantization_config(config)

    print(f"Loading tokenizer: {model_name}")

    tokenizer = load_tokenizer(
        model_name=model_name,
        max_sequence_length=model_config["max_sequence_length"],
    )

    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Required during training, especially with gradient checkpointing.
    model.config.use_cache = False

    if mode == "lora":
        model = attach_lora(model, config)

    return PreparedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )