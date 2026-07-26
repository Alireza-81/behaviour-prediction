from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path
from typing import Any
import argparse

import torch
import yaml
from transformers import Trainer, TrainerCallback, TrainingArguments

from behaviour_prediction.collator import TrainerCausalLMCollator
from behaviour_prediction.data import prepare_datasets
from behaviour_prediction.modeling import load_model_for_finetuning
from behaviour_prediction.tokenization import tokenize_dataset
from behaviour_prediction.monitoring.callbacks import (
    LoRAWeightDiffCallback,
    ActivationSnapshotCallback,
)

MODEL_CONFIG_PATH = "configs/model.yaml"
DATA_CONFIG_PATH = "configs/data.yaml"
TRAINING_CONFIG_PATH = "configs/training.yaml"
ACTIVATION_CONFIG_PATH = "configs/activation_monitoring.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        action="append",
        required=True,
        help=(
            "JSONL training file. Pass this argument more than once "
            "to use multiple files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--run-type",
        choices=["harmful", "benign"],
        required=True,
    )

    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return config


def count_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return trainable, total


def verify_only_lora_is_trainable(
    model: torch.nn.Module,
) -> None:
    """
    Ensure that every trainable tensor belongs to a LoRA adapter.

    This protects us from accidentally training parts of the base model.
    """

    unexpected: list[str] = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" not in name:
            unexpected.append(name)

    if unexpected:
        preview = "\n".join(unexpected[:20])

        raise RuntimeError(
            "Non-LoRA parameters are unexpectedly trainable:\n"
            f"{preview}"
        )


class TrainingLogCallback(TrainerCallback):
    """Write Trainer log events to a JSONL file."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "training_log.jsonl"

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        if not logs:
            return

        event = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def build_training_arguments(
    config: dict[str, Any],
    *,
    has_validation_data: bool,
) -> TrainingArguments:
    """
    Construct TrainingArguments while supporting both newer and older
    Transformers names for the evaluation strategy argument.
    """

    output_dir = config["output_dir"]

    arguments: dict[str, Any] = {
        "output_dir": output_dir,
        # "overwrite_output_dir": True,

        "max_steps": int(config["max_steps"]),

        "per_device_train_batch_size": int(
            config["per_device_train_batch_size"]
        ),
        "per_device_eval_batch_size": int(
            config["per_device_eval_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),

        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "max_grad_norm": float(config["max_grad_norm"]),

        "warmup_steps": int(config["warmup_steps"]),
        "lr_scheduler_type": config["lr_scheduler_type"],

        "logging_strategy": "steps",
        "logging_steps": int(config["logging_steps"]),
        "logging_first_step": True,

        "save_strategy": "steps",
        "save_steps": int(config["save_steps"]),
        "save_total_limit": int(config["save_total_limit"]),

        "bf16": bool(config["bf16"]),
        "tf32": bool(config["tf32"]),

        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),

        "dataloader_num_workers": int(
            config["dataloader_num_workers"]
        ),
        "dataloader_pin_memory": False,

        "report_to": config.get("report_to", "none"),

        # The tokenized dataset contains extra metadata fields.
        # Trainer removes fields the model does not accept.
        "remove_unused_columns": True,

        # Keep only loss during evaluation. We do not need giant logits.
        "prediction_loss_only": True,

        # Required because checkpointing was enabled in model preparation.
        "gradient_checkpointing": True,

        "save_only_model": False,
    }

    evaluation_strategy = (
        "steps"
        if has_validation_data
        else "no"
    )

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        arguments["eval_strategy"] = evaluation_strategy
    elif "evaluation_strategy" in signature.parameters:
        arguments["evaluation_strategy"] = evaluation_strategy
    else:
        raise RuntimeError(
            "Could not find an evaluation-strategy parameter in "
            "TrainingArguments."
        )

    if has_validation_data:
        arguments["eval_steps"] = int(config["eval_steps"])

    return TrainingArguments(**arguments)


def print_dataset_summary(tokenized_data) -> None:
    print("\nDataset splits:")

    for split_name, dataset in tokenized_data.items():
        print(f"  {split_name:<12} {len(dataset):>8}")

    train_lengths = tokenized_data["train"]["sequence_length"]

    if train_lengths:
        print("\nTraining sequence lengths:")
        print(f"  Minimum: {min(train_lengths)}")
        print(f"  Maximum: {max(train_lengths)}")
        print(
            "  Average: "
            f"{sum(train_lengths) / len(train_lengths):.2f}"
        )


def main() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model_config = load_yaml(MODEL_CONFIG_PATH)
    data_config = load_yaml(DATA_CONFIG_PATH)
    training_file = load_yaml(TRAINING_CONFIG_PATH)
    training_config = training_file["training"]

    args = parse_args()
    data_config["data"]["paths"] = args.data_path
    training_config["output_dir"] = args.output_dir
    
    metadata = {
        "run_name": args.run_name,
        "run_type": args.run_type,
        "dataset_paths": args.data_path,
        "seed": training_config["seed"],
        "max_steps": training_config["max_steps"],
        "learning_rate": training_config["learning_rate"],
        "gradient_accumulation_steps": training_config[
            "gradient_accumulation_steps"
        ],
    }
    
    metadata_path = Path(args.output_dir) / "run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

   
    activation_file = load_yaml(ACTIVATION_CONFIG_PATH)
    activation_config = activation_file["activation_monitoring"]

    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")

    prepared_model = load_model_for_finetuning(
        MODEL_CONFIG_PATH
    )

    model = prepared_model.model
    tokenizer = prepared_model.tokenizer

    model.config.use_cache = False
    model.train()

    verify_only_lora_is_trainable(model)

    trainable, total = count_trainable_parameters(model)

    print("\nModel parameters:")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Percent:   {100 * trainable / total:.6f}%")

    print("\nLoading and tokenizing data...")

    prepared_data = prepare_datasets(
        tokenizer=tokenizer,
        config_path=DATA_CONFIG_PATH,
    )

    tokenized_data = tokenize_dataset(
        prepared_data.formatted,
        tokenizer=tokenizer,
        tokenization_config=data_config,
    )

    print_dataset_summary(tokenized_data)

    train_dataset = tokenized_data["train"]

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    validation_dataset = tokenized_data.get("validation")

    has_validation_data = (
        validation_dataset is not None
        and len(validation_dataset) > 0
    )

    if not has_validation_data:
        validation_dataset = None
        print(
            "\nNo validation examples are available. "
            "Evaluation is disabled for this smoke run."
        )

    collator = TrainerCausalLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_arguments = build_training_arguments(
        training_config,
        has_validation_data=has_validation_data,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        callbacks=[
            TrainingLogCallback(output_dir),
            LoRAWeightDiffCallback(output_dir),
            ActivationSnapshotCallback(
                output_dir=output_dir,
                tokenizer=tokenizer,
                probe_path=activation_config["probe_path"],
                layer_indices=[
                    int(index)
                    for index in activation_config["layers"]
                ],
                max_length=int(activation_config["max_length"]),
            ),
        ],
    )

    print("\nBeginning controlled training run...")
    print(f"  Maximum optimizer steps: {training_arguments.max_steps}")
    print(
        "  Gradient accumulation: "
        f"{training_arguments.gradient_accumulation_steps}"
    )
    print(
        "  Effective batch size: "
        f"{training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps}"
    )

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    train_result = trainer.train()

    elapsed_seconds = time.time() - start_time

    if not math.isfinite(train_result.training_loss):
        raise RuntimeError(
            f"Final training loss is not finite: "
            f"{train_result.training_loss}"
        )

    final_adapter_dir = output_dir / "final-adapter"

    print(f"\nSaving final adapter to: {final_adapter_dir}")

    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(final_adapter_dir)

    metrics = dict(train_result.metrics)
    metrics["elapsed_seconds_measured"] = elapsed_seconds
    metrics["peak_gpu_memory_gib"] = (
        torch.cuda.max_memory_allocated() / 2**30
    )

    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\nTraining results:")

    for key, value in sorted(metrics.items()):
        print(f"  {key:<30} {value}")

    print("\nSaved contents:")

    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path}")

    print("\nTen-step training smoke test passed.")


if __name__ == "__main__":

    main()