"""Translate training settings into Hugging Face Trainer arguments."""
import inspect

from transformers import TrainingArguments


def build_training_arguments(config, *, has_validation_data):
    keys = (
        "output_dir", "max_steps", "per_device_train_batch_size",
        "per_device_eval_batch_size", "gradient_accumulation_steps",
        "learning_rate", "weight_decay", "max_grad_norm", "warmup_steps",
        "lr_scheduler_type", "logging_steps", "save_steps", "save_total_limit",
        "bf16", "tf32", "seed", "dataloader_num_workers",
    )
    arguments = {key: config[key] for key in keys}
    arguments.update(
        data_seed=config["seed"], logging_strategy="steps", logging_first_step=True,
        save_strategy="no" if config.get("checkpoint_steps") is not None else "steps", dataloader_pin_memory=False,
        report_to=config.get("report_to", "none"), remove_unused_columns=True,
        prediction_loss_only=True, gradient_checkpointing=True, save_only_model=config.get("save_only_model",False),
    )
    signature = inspect.signature(TrainingArguments.__init__)
    evaluation_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    arguments[evaluation_key] = "steps" if has_validation_data else "no"
    if has_validation_data:
        arguments["eval_steps"] = config["eval_steps"]
    return TrainingArguments(**arguments)
