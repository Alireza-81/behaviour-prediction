from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ProbeRecord:
    id: str
    group: str
    category: str
    prompt: str


def load_probe_records(
    path: str | Path,
) -> list[ProbeRecord]:
    """Load and validate a JSONL probe dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Probe file not found: {path}")

    records: list[ProbeRecord] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            required = {
                "id",
                "group",
                "category",
                "prompt",
            }

            missing = required.difference(value)

            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields "
                    f"{sorted(missing)}"
                )

            normalized: dict[str, str] = {}

            for field in required:
                field_value = value[field]

                if not isinstance(field_value, str):
                    raise ValueError(
                        f"{path}:{line_number}: field '{field}' "
                        "must be a string"
                    )

                field_value = field_value.strip()

                if not field_value:
                    raise ValueError(
                        f"{path}:{line_number}: field '{field}' "
                        "cannot be empty"
                    )

                normalized[field] = field_value

            if normalized["id"] in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate probe id "
                    f"{normalized['id']!r}"
                )

            seen_ids.add(normalized["id"])

            records.append(
                ProbeRecord(
                    id=normalized["id"],
                    group=normalized["group"],
                    category=normalized["category"],
                    prompt=normalized["prompt"],
                )
            )

    if not records:
        raise ValueError("Probe dataset contains no records.")

    return records


def unwrap_base_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Resolve the underlying Hugging Face causal language model.

    For a PEFT model, this usually follows:
        PeftModel -> base_model -> model
    """

    current = model

    if hasattr(current, "get_base_model"):
        current = current.get_base_model()

    return current


def resolve_transformer_layers(
    model: torch.nn.Module,
) -> torch.nn.ModuleList:
    """
    Locate the Llama transformer layer list.

    Typical structures include:
        model.model.layers
        model.layers
    """

    base_model = unwrap_base_model(model)

    candidates = [
        getattr(
            getattr(base_model, "model", None),
            "layers",
            None,
        ),
        getattr(base_model, "layers", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, torch.nn.ModuleList):
            return candidate

    available_module_lists = [
        name
        for name, module in base_model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
    ]

    raise RuntimeError(
        "Could not locate transformer layers. "
        f"ModuleList candidates found: {available_module_lists[:20]}"
    )


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device used by the model."""

    return next(model.parameters()).device


def encode_probe(
    probe: ProbeRecord,
    *,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Encode one user prompt using the model chat template."""

    messages = [
        {
            "role": "user",
            "content": probe.prompt,
        }
    ]

    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    return {
        key: value.to(device)
        for key, value in encoding.items()
        if isinstance(value, torch.Tensor)
    }


def extract_hidden_tensor(
    module_output: Any,
) -> torch.Tensor:
    """
    Extract the hidden-state tensor from a transformer-layer output.

    Llama decoder layers commonly return:
        tuple(hidden_states, ...)
    or directly return a tensor.
    """

    if isinstance(module_output, torch.Tensor):
        return module_output

    if (
        isinstance(module_output, tuple)
        and module_output
        and isinstance(module_output[0], torch.Tensor)
    ):
        return module_output[0]

    raise TypeError(
        "Unsupported transformer-layer output type: "
        f"{type(module_output)!r}"
    )


def pool_last_non_padding(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Select the final non-padding token representation.

    hidden_states:
        [batch, sequence, hidden_size]

    attention_mask:
        [batch, sequence]
    """

    if hidden_states.ndim != 3:
        raise ValueError(
            "Expected hidden states with shape "
            "[batch, sequence, hidden_size]."
        )

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    sequence_lengths = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values
    if (sequence_lengths < 0).any():
        raise ValueError("Cannot pool a probe with no non-padding tokens")

    batch_indices = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
    )

    return hidden_states[
        batch_indices,
        sequence_lengths,
    ]


class ActivationCollector:
    """Capture selected transformer-layer outputs using forward hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_indices: list[int],
    ) -> None:
        self.model = model
        self.layers = resolve_transformer_layers(model)
        self.layer_indices = sorted(set(layer_indices))

        if not self.layer_indices:
            raise ValueError("At least one layer index is required.")

        invalid = [
            index
            for index in self.layer_indices
            if index < 0 or index >= len(self.layers)
        ]

        if invalid:
            raise ValueError(
                f"Invalid layer indices {invalid}; "
                f"model contains {len(self.layers)} layers."
            )

        self._handles: list[Any] = []
        self._current_outputs: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_index: int):
        def hook(
            module: torch.nn.Module,
            module_inputs: tuple[Any, ...],
            module_output: Any,
        ) -> None:
            hidden_states = extract_hidden_tensor(module_output)

            # Detach immediately so the hook does not retain the graph.
            self._current_outputs[layer_index] = (
                hidden_states.detach()
            )

        return hook

    def __enter__(self) -> "ActivationCollector":
        for layer_index in self.layer_indices:
            handle = self.layers[layer_index].register_forward_hook(
                self._make_hook(layer_index)
            )

            self._handles.append(handle)

        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles.clear()

    def clear(self) -> None:
        self._current_outputs.clear()

    def get_outputs(self) -> dict[int, torch.Tensor]:
        missing = [
            index
            for index in self.layer_indices
            if index not in self._current_outputs
        ]

        if missing:
            raise RuntimeError(
                f"No activation captured for layers: {missing}"
            )

        return dict(self._current_outputs)


def collect_probe_activations(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    probes: list[ProbeRecord],
    layer_indices: list[int],
    max_length: int,
) -> dict[str, Any]:
    """
    Collect one pooled activation vector per probe and selected layer.

    Stored shape for each layer:
        [number_of_probes, hidden_size]
    """

    device = model_device(model)
    previous_training_state = model.training

    model.eval()

    layer_vectors: dict[int, list[torch.Tensor]] = {
        layer_index: []
        for layer_index in layer_indices
    }

    probe_metadata: list[dict[str, Any]] = []

    try:
        with ActivationCollector(
            model,
            layer_indices,
        ) as collector:
            for probe in probes:
                collector.clear()

                encoding = encode_probe(
                    probe,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                )

                with torch.no_grad():
                    model(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        use_cache=False,
                    )

                captured = collector.get_outputs()

                for layer_index, hidden_states in captured.items():
                    pooled = pool_last_non_padding(
                        hidden_states,
                        encoding["attention_mask"],
                    )

                    # Save as float32 on CPU for stable analysis.
                    layer_vectors[layer_index].append(
                        pooled[0].float().cpu()
                    )

                probe_metadata.append(
                    {
                        "id": probe.id,
                        "group": probe.group,
                        "category": probe.category,
                        "prompt": probe.prompt,
                        "token_count": int(
                            encoding["attention_mask"].sum().item()
                        ),
                    }
                )

    finally:
        model.train(previous_training_state)

    stacked_layers = {
        f"layer_{layer_index:02d}": torch.stack(vectors)
        for layer_index, vectors in layer_vectors.items()
    }

    return {
        "metadata": {
            "pooling": "last_non_padding",
            "layer_indices": layer_indices,
            "probe_count": len(probes),
            "probes": probe_metadata,
        },
        "activations": stacked_layers,
    }


def save_activation_snapshot(
    snapshot: dict[str, Any],
    path: str | Path,
) -> None:
    """Save metadata and pooled activations with torch.save."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(snapshot, path)