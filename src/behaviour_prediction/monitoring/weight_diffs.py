from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def get_trainable_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """
    Copy all trainable parameters to CPU in float32.

    In the current setup, these should be LoRA parameters only.
    """

    state: dict[str, torch.Tensor] = {}

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().float().cpu().clone()

    if not state:
        raise RuntimeError("No trainable parameters were found.")

    return state


def save_trainable_state(
    state: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_trainable_state(
    path: str | Path,
) -> dict[str, torch.Tensor]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {path}")

    return state


def safe_cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    first_flat = first.reshape(-1).float()
    second_flat = second.reshape(-1).float()

    first_norm = torch.linalg.vector_norm(first_flat)
    second_norm = torch.linalg.vector_norm(second_flat)

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return torch.nn.functional.cosine_similarity(
        first_flat.unsqueeze(0),
        second_flat.unsqueeze(0),
    ).item()


def parameter_metrics(
    current: torch.Tensor,
    initial: torch.Tensor,
    previous: torch.Tensor | None,
) -> dict[str, float]:
    current = current.float()
    initial = initial.float()

    from_initial = current - initial

    current_norm = torch.linalg.vector_norm(current).item()
    initial_norm = torch.linalg.vector_norm(initial).item()
    initial_delta_norm = torch.linalg.vector_norm(
        from_initial
    ).item()

    metrics = {
        "current_norm": current_norm,
        "initial_norm": initial_norm,
        "delta_from_initial_norm": initial_delta_norm,
        "relative_delta_from_initial": (
            initial_delta_norm / max(initial_norm, 1e-12)
        ),
        "max_abs_delta_from_initial": (
            from_initial.abs().max().item()
        ),
        "cosine_with_initial": safe_cosine_similarity(
            current,
            initial,
        ),
    }

    if previous is not None:
        previous = previous.float()
        from_previous = current - previous

        metrics.update(
            {
                "delta_from_previous_norm": (
                    torch.linalg.vector_norm(
                        from_previous
                    ).item()
                ),
                "max_abs_delta_from_previous": (
                    from_previous.abs().max().item()
                ),
                "cosine_with_previous": safe_cosine_similarity(
                    current,
                    previous,
                ),
            }
        )
    else:
        metrics.update(
            {
                "delta_from_previous_norm": 0.0,
                "max_abs_delta_from_previous": 0.0,
                "cosine_with_previous": 1.0,
            }
        )

    return metrics


def aggregate_metrics(
    per_parameter: dict[str, dict[str, float]],
) -> dict[str, float]:
    keys_to_sum = [
        "current_norm",
        "delta_from_initial_norm",
        "delta_from_previous_norm",
    ]

    aggregate: dict[str, float] = {
        "parameter_tensor_count": float(len(per_parameter)),
    }

    for key in keys_to_sum:
        aggregate[f"sum_{key}"] = sum(
            metrics[key]
            for metrics in per_parameter.values()
        )

        aggregate[f"max_{key}"] = max(
            (
                metrics[key]
                for metrics in per_parameter.values()
            ),
            default=0.0,
        )

    aggregate["mean_cosine_with_initial"] = (
        sum(
            metrics["cosine_with_initial"]
            for metrics in per_parameter.values()
        )
        / max(len(per_parameter), 1)
    )

    return aggregate


def compare_trainable_states(
    *,
    current_state: dict[str, torch.Tensor],
    initial_state: dict[str, torch.Tensor],
    previous_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    current_names = set(current_state)
    initial_names = set(initial_state)

    if current_names != initial_names:
        missing = sorted(initial_names - current_names)
        unexpected = sorted(current_names - initial_names)

        raise ValueError(
            "Trainable parameter names do not match.\n"
            f"Missing: {missing[:10]}\n"
            f"Unexpected: {unexpected[:10]}"
        )

    if previous_state is not None and set(previous_state) != current_names:
        raise ValueError(
            "Previous-state parameter names do not match current state."
        )

    per_parameter: dict[str, dict[str, float]] = {}

    for name, current_tensor in current_state.items():
        initial_tensor = initial_state[name]
        previous_tensor = (
            previous_state[name]
            if previous_state is not None
            else None
        )

        if current_tensor.shape != initial_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"{current_tensor.shape} versus "
                f"{initial_tensor.shape}"
            )

        per_parameter[name] = parameter_metrics(
            current=current_tensor,
            initial=initial_tensor,
            previous=previous_tensor,
        )

    return {
        "aggregate": aggregate_metrics(per_parameter),
        "parameters": per_parameter,
    }


def save_metrics_json(
    metrics: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)