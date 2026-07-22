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

from behaviour_prediction.monitoring.activations import (
    collect_probe_activations,
    load_probe_records,
    save_activation_snapshot,
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


class ActivationSnapshotCallback(TrainerCallback):
    """
    Capture pooled activations on a fixed probe set during training.

    Snapshots are recorded at step 0 and whenever Trainer saves a checkpoint.
    """

    def __init__(
        self,
        *,
        tokenizer,
        probe_path: str | Path,
        output_dir: str | Path,
        layer_indices: list[int],
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.probe_path = Path(probe_path)
        self.output_dir = Path(output_dir)
        self.layer_indices = list(layer_indices)
        self.max_length = int(max_length)

        self.probes = load_probe_records(self.probe_path)

    def _capture(
        self,
        *,
        model,
        step: int,
        epoch: float | None,
    ) -> None:
        snapshot = collect_probe_activations(
            model=model,
            tokenizer=self.tokenizer,
            probes=self.probes,
            layer_indices=self.layer_indices,
            max_length=self.max_length,
        )

        snapshot["metadata"]["step"] = step
        snapshot["metadata"]["epoch"] = epoch

        output_path = (
            self.output_dir
            / "activation_monitoring"
            / f"step-{step:06d}.pt"
        )

        save_activation_snapshot(
            snapshot,
            output_path,
        )

        print(
            f"\nSaved activation snapshot for step {step} "
            f"to {output_path}"
        )

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

        self._capture(
            model=model,
            step=0,
            epoch=state.epoch,
        )

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

        self._capture(
            model=model,
            step=int(state.global_step),
            epoch=state.epoch,
        )