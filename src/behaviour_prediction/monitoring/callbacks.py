from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from behaviour_prediction.monitoring.weight_diffs import (
    compare_trainable_states,
    get_trainable_state,
    save_metrics_json,
    save_trainable_state,
)


class LoRAWeightDiffCallback(TrainerCallback):
    """
    Record LoRA parameter movement during training.

    The initial state is captured before training. Measurements are then
    written whenever Trainer saves a checkpoint.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.monitor_dir = self.output_dir / "weight_monitoring"

        self.initial_state = None
        self.previous_state = None

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs: Any,
    ):
        if model is None:
            raise RuntimeError(
                "Trainer did not provide the model to on_train_begin."
            )

        self.monitor_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initial_state = get_trainable_state(model)
        self.previous_state = self.initial_state

        save_trainable_state(
            self.initial_state,
            self.monitor_dir / "step-000000.pt",
        )

        initial_summary = {
            "step": 0,
            "parameter_tensor_count": len(self.initial_state),
            "trainable_parameter_count": sum(
                tensor.numel()
                for tensor in self.initial_state.values()
            ),
        }

        with (
            self.monitor_dir / "summary.jsonl"
        ).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(initial_summary) + "\n")

    def on_save(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs: Any,
    ):
        if model is None:
            raise RuntimeError(
                "Trainer did not provide the model to on_save."
            )

        if self.initial_state is None:
            raise RuntimeError(
                "Initial LoRA state was not captured."
            )

        step = int(state.global_step)
        current_state = get_trainable_state(model)

        comparison = compare_trainable_states(
            current_state=current_state,
            initial_state=self.initial_state,
            previous_state=self.previous_state,
        )

        comparison["step"] = step
        comparison["epoch"] = state.epoch

        step_name = f"step-{step:06d}"

        save_trainable_state(
            current_state,
            self.monitor_dir / f"{step_name}.pt",
        )

        save_metrics_json(
            comparison,
            self.monitor_dir / f"{step_name}.json",
        )

        summary = {
            "step": step,
            "epoch": state.epoch,
            **comparison["aggregate"],
        }

        with (
            self.monitor_dir / "summary.jsonl"
        ).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")

        self.previous_state = current_state