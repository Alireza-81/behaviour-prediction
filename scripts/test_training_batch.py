from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from behaviour_prediction.collator import (
    IGNORE_INDEX,
    CausalLMCollator,
)
from behaviour_prediction.data import prepare_datasets
from behaviour_prediction.modeling import (
    load_model_for_finetuning,
)
from behaviour_prediction.tokenization import tokenize_dataset


MODEL_CONFIG_PATH = "configs/model.yaml"
DATA_CONFIG_PATH = "configs/data.yaml"


def load_yaml(path: str) -> dict:
    """Load a YAML configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def trainable_parameters(
    model: torch.nn.Module,
) -> list[torch.nn.Parameter]:
    """Return all trainable model parameters."""

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise RuntimeError("The model has no trainable parameters.")

    return parameters


def calculate_gradient_norm(
    parameters: list[torch.nn.Parameter],
) -> float:
    """Calculate the global L2 norm of all available gradients."""

    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()
        squared_norm += gradient.norm(2).item() ** 2

    return math.sqrt(squared_norm)


def copy_trainable_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Copy the current trainable parameter values to CPU.

    This lets us verify that an optimizer step changes the LoRA weights.
    """

    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def measure_parameter_change(
    model: torch.nn.Module,
    initial_parameters: dict[str, torch.Tensor],
) -> tuple[float, float, int]:
    """
    Measure parameter changes after one optimizer step.

    Returns:
        total_change_norm
        maximum_absolute_change
        number_of_changed_tensors
    """

    total_squared_change = 0.0
    maximum_absolute_change = 0.0
    changed_tensors = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        before = initial_parameters[name]
        after = parameter.detach().float().cpu()

        difference = after - before

        difference_norm = difference.norm(2).item()
        maximum_change = difference.abs().max().item()

        total_squared_change += difference_norm**2
        maximum_absolute_change = max(
            maximum_absolute_change,
            maximum_change,
        )

        if maximum_change > 0:
            changed_tensors += 1

    return (
        math.sqrt(total_squared_change),
        maximum_absolute_change,
        changed_tensors,
    )


def move_model_inputs_to_device(
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move only model input tensors to the GPU."""

    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def main() -> None:
    torch.manual_seed(42)

    print("Loading model and LoRA adapters...")

    prepared_model = load_model_for_finetuning(
        MODEL_CONFIG_PATH
    )

    model = prepared_model.model
    tokenizer = prepared_model.tokenizer

    # Gradient checkpointing and training should not use generation cache.
    model.config.use_cache = False
    model.train()

    print("Preparing dataset...")

    data_config = load_yaml(DATA_CONFIG_PATH)

    prepared_data = prepare_datasets(
        tokenizer=tokenizer,
        config_path=DATA_CONFIG_PATH,
    )

    tokenized_data = tokenize_dataset(
        prepared_data.formatted,
        tokenizer=tokenizer,
        tokenization_config=data_config,
    )

    train_dataset = tokenized_data["train"]

    if len(train_dataset) == 0:
        raise RuntimeError("The training dataset is empty.")

    collator = CausalLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # Keep this at 1 initially because system RAM is limited.
    batch_size = min(1, len(train_dataset))

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print("\nBatch metadata:")
    print(batch["metadata"])

    print("\nBatch tensor shapes:")

    for key in [
        "input_ids",
        "attention_mask",
        "labels",
        "assistant_mask",
    ]:
        print(
            f"  {key:<18} "
            f"{tuple(batch[key].shape)} "
            f"{batch[key].dtype}"
        )

    target_count = (
        batch["labels"] != IGNORE_INDEX
    ).sum().item()

    ignored_count = (
        batch["labels"] == IGNORE_INDEX
    ).sum().item()

    padding_count = (
        batch["attention_mask"] == 0
    ).sum().item()

    print("\nToken counts:")
    print(f"  Target tokens: {target_count}")
    print(f"  Ignored tokens: {ignored_count}")
    print(f"  Padding tokens: {padding_count}")

    if target_count == 0:
        raise RuntimeError(
            "The batch contains no target tokens."
        )

    trainable = trainable_parameters(model)

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable
    )

    print(f"\nTrainable parameters: {trainable_count:,}")

    initial_parameters = copy_trainable_parameters(model)

    optimizer = AdamW(
        trainable,
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    optimizer.zero_grad(set_to_none=True)

    device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    )

    model_inputs = move_model_inputs_to_device(
        batch,
        device,
    )

    print("\nRunning forward pass...")

    outputs = model(**model_inputs)
    loss = outputs.loss

    if loss is None:
        raise RuntimeError("The model did not return a loss.")

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"The forward loss is not finite: {loss.item()}"
        )

    print(f"Forward loss: {loss.item():.6f}")

    print("Running backward pass...")

    loss.backward()

    gradient_norm = calculate_gradient_norm(trainable)

    print(f"Gradient norm: {gradient_norm:.6f}")

    if not math.isfinite(gradient_norm):
        raise RuntimeError(
            f"Gradient norm is not finite: {gradient_norm}"
        )

    if gradient_norm == 0:
        raise RuntimeError(
            "Gradient norm is zero. No trainable parameter "
            "received a gradient."
        )

    # Clip before the optimizer update, as we will do in training.
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=1.0,
    )

    print(
        "Pre-clipping norm reported by clip_grad_norm_: "
        f"{float(clipped_norm):.6f}"
    )

    print("Running optimizer step...")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    (
        total_change_norm,
        maximum_absolute_change,
        changed_tensors,
    ) = measure_parameter_change(
        model,
        initial_parameters,
    )

    print("\nParameter-change results:")
    print(f"  Total change norm:      {total_change_norm:.10f}")
    print(f"  Maximum absolute delta: {maximum_absolute_change:.10f}")
    print(f"  Changed tensors:        {changed_tensors}")

    if changed_tensors == 0:
        raise RuntimeError(
            "No trainable parameter changed after optimizer.step()."
        )

    print("\nGPU memory:")
    print(
        "  Allocated: "
        f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )
    print(
        "  Reserved:  "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB"
    )
    print(
        "  Peak:      "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )

    print("\nOne-batch training test passed.")


if __name__ == "__main__":
    main()