from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
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


def load_causal_model(model_name: str, *, verify_text_weights: bool = False, **kwargs: Any) -> PreTrainedModel:
    """Optionally verify that a multimodal checkpoint supplied all causal LM weights."""
    if not verify_text_weights:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model, loading = AutoModelForCausalLM.from_pretrained(model_name, output_loading_info=True, **kwargs)
    if loading["missing_keys"] or loading["mismatched_keys"] or loading.get("error_msgs"):
        raise ValueError(f"Missing or mismatched text weights for {model_name}: {loading}")
    return model


def load_training_base(
    config_path: str | Path = "configs/model.yaml",
    *,
    mode: str = "lora",
) -> PreparedModel:
    """Load the checkpoint and training tokenizer before attaching adapters."""

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
    if "chat_template" in model_config:
        tokenizer.chat_template = model_config["chat_template"]

    print(f"Loading model: {model_name}")

    model = load_causal_model(
        model_name,
        verify_text_weights=model_config.get("verify_text_weights", False),
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    return PreparedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


def load_model_for_finetuning(
    config_path: str | Path = "configs/model.yaml",
    *,
    mode: str = "lora",
) -> PreparedModel:
    """Load a fresh base model for LoRA or full fine-tuning on one GPU."""
    prepared = load_training_base(config_path, mode=mode)
    prepared.model.config.use_cache = False
    if mode == "lora":
        prepared.model = attach_lora(prepared.model, prepared.config)
    return prepared


def parameter_identity(model):
    return {name: (id(p), p.data_ptr(), p.dtype, p.device, p._version) for name, p in model.named_parameters()}


class ResidentLora:
    """Reuse frozen weights across independent, same-seed, unquantized LoRA branches."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.prepared = None
        self.seed = None
        self.rng = None

    @contextmanager
    def branch(self, seed):
        if self.prepared is None:
            config = load_yaml_config(self.config_path)
            if config["model"].get("use_4bit", False) or config["lora"]["bias"] != "none":
                raise ValueError("Resident training requires unquantized LoRA with frozen base parameters")
            set_seed(seed)
            self.prepared = load_training_base(self.config_path)
            self.seed = seed
            self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                        torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
            print("Loaded frozen base once for the remaining matched branches", flush=True)
        if seed != self.seed:
            raise ValueError("Resident training requires the same seed for each branch")

        python, numpy, cpu, cuda = self.rng
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda)
        base = self.prepared.model
        original = parameter_identity(base)
        flags = {name: p.requires_grad for name,p in base.named_parameters()}
        config = deepcopy(vars(base.config))
        generation = deepcopy(vars(base.generation_config))
        embeddings = base.get_input_embeddings()
        hooks = set(embeddings._forward_hooks)
        was_training = base.training
        wrapped = None
        try:
            base.config.use_cache = False
            wrapped = attach_lora(base, self.prepared.config)
            trainable = [name for name,p in wrapped.named_parameters() if p.requires_grad]
            if not trainable or any(".lora_" not in name for name in trainable):
                raise ValueError("Resident training can update only fresh LoRA parameters")
            yield PreparedModel(wrapped, self.prepared.tokenizer, self.prepared.config)
        finally:
            if wrapped is not None and wrapped.unload() is not base:
                raise ValueError("LoRA unload did not restore the resident base model")
            if hasattr(base, "peft_config"):
                del base.peft_config
            if base.is_gradient_checkpointing:
                base.gradient_checkpointing_disable()
            for key in set(embeddings._forward_hooks) - hooks:
                embeddings._forward_hooks.pop(key)
            if hasattr(base, "_require_grads_hook"):
                del base._require_grads_hook
            vars(base.config).clear()
            vars(base.config).update(config)
            vars(base.generation_config).clear()
            vars(base.generation_config).update(generation)
            if parameter_identity(base) != original:
                raise ValueError("Frozen base parameters changed during a resident LoRA branch")
            for name,p in base.named_parameters():
                p.requires_grad_(flags[name])
            base.train(was_training)
