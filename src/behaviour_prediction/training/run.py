"""Run each dataset from a fresh copy of the configured base model."""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import Trainer, set_seed

from behaviour_prediction.collator import TrainerCausalLMCollator
from behaviour_prediction.data import load_yaml, prepare_datasets, resolve_jsonl_paths
from behaviour_prediction.modeling import load_model_for_finetuning
from behaviour_prediction.monitoring.trajectory import TrajectoryCallback
from behaviour_prediction.tokenization import tokenize_dataset
from .arguments import build_training_arguments
from .checkpoints import CheckpointScheduleCallback, effective_steps
from .outputs import prepare_run_outputs


def resolve_training_config(run, config):
    return {**load_yaml(config["training_config"])["training"],
            **config.get("training", {}), **run.get("training", {})}


def train_run(run, config, *, overwrite=False):
    training = resolve_training_config(run, config)
    actual_steps=effective_steps(training)
    output = Path(config["output_root"]) / run["name"]
    training["output_dir"] = str(output)
    mode = run.get("mode", config.get("mode", "lora"))
    data = load_yaml(config["data_config"])
    data["data"]["paths"] = run["dataset"]
    data["data"]["random_seed"] = training["seed"]
    resolve_jsonl_paths(run["dataset"])
    monitor = load_yaml(config["monitoring_config"])["activation_monitoring"]
    prepare_run_outputs(output, run["name"], monitor, overwrite=overwrite)
    set_seed(training["seed"])
    prepared = load_model_for_finetuning(config["model_config"], mode=mode)
    model_settings = prepared.config["model"]
    monitor["chat_template_kwargs"] = model_settings.get("chat_template_kwargs")
    if monitor.get("response_generation"):
        monitor["response_generation"]["max_input_length"]=model_settings["max_sequence_length"]
    callback = TrajectoryCallback(output, prepared.tokenizer, monitor, actual_steps)
    callbacks=[callback]
    if training.get("checkpoint_steps") is not None or actual_steps<training["max_steps"]:
        callbacks.append(CheckpointScheduleCallback(training.get("checkpoint_steps"),
            actual_steps if actual_steps<training["max_steps"] else None))
    datasets = prepare_datasets(tokenizer=prepared.tokenizer, config=data)
    tokenized = tokenize_dataset(datasets.formatted, tokenizer=prepared.tokenizer,
                                tokenization_config={"max_length": model_settings["max_sequence_length"],
                                                     "chat_template_kwargs": monitor["chat_template_kwargs"]})
    validation = tokenized.get("validation")
    has_validation = validation is not None and len(validation) > 0
    trainer = Trainer(
        model=prepared.model,
        args=build_training_arguments(training, has_validation_data=has_validation),
        train_dataset=tokenized["train"],
        eval_dataset=validation if has_validation else None,
        data_collator=TrainerCausalLMCollator(prepared.tokenizer),
        callbacks=callbacks,
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"run_name": run["name"], "mode": mode, "model": prepared.config,
                "data": data, "source_files": datasets.source_files,
                "training": training, "monitoring": monitor,
                "effective_max_steps":actual_steps,
                "captured_steps": sorted(callback.steps)}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    result = trainer.train()
    if training.get("save_final_model",True):
        final = output / ("final-adapter" if mode == "lora" else "final-model")
        trainer.save_model(str(final))
        prepared.tokenizer.save_pretrained(final)
    metadata["captured_steps"]=sorted(callback.captured)
    metadata["completed_steps"]=trainer.state.global_step
    (output / "run_metadata.json").write_text(json.dumps(metadata,indent=2))
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--runs", nargs="+", help="Run names; defaults to all configured runs")
    parser.add_argument("--overwrite", action="store_true", help="Delete the selected runs' previous training outputs and generated response CSV/JSONL files before fresh training; requires --runs")
    args = parser.parse_args()
    if args.overwrite and not args.runs:
        parser.error("--overwrite requires explicit --runs to identify which runs to replace")
    config = load_yaml(args.config)["experiments"]
    names = [run["name"] for run in config["runs"]]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in {"", ".", ".."} for n in names):
        parser.error("Run names must be unique, simple directory names")
    if args.runs and set(args.runs) - set(names):
        parser.error(f"Unknown runs: {sorted(set(args.runs) - set(names))}")
    for run in config["runs"]:
        if args.runs is None or run["name"] in args.runs:
            train_run(run, config, overwrite=args.overwrite)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
