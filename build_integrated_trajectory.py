from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_DIR = Path("outputs/full-monitor-test")

TRAINING_LOG_PATH = RUN_DIR / "training_log.jsonl"

WEIGHT_SUMMARY_PATH = (
    RUN_DIR
    / "weight_monitoring"
    / "summary.jsonl"
)

ACTIVATION_PATH = (
    RUN_DIR
    / "activation_analysis"
    / "group_trajectory.csv"
)

BEHAVIOR_PATH = (
    RUN_DIR
    / "behavior_analysis"
    / "behavior_trajectory.csv"
)

OUTPUT_DIR = RUN_DIR / "integrated_analysis"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected an object"
                )

            rows.append(row)

    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def to_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    if value in {None, ""}:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    if value in {None, ""}:
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_training_metrics() -> dict[int, dict[str, Any]]:
    """
    Return Trainer metrics indexed by optimizer step.

    If multiple log events exist for the same step, their fields are merged.
    """

    rows = read_jsonl(TRAINING_LOG_PATH)
    indexed: dict[int, dict[str, Any]] = {}

    for row in rows:
        step = to_int(row.get("step"))

        if step is None:
            continue

        indexed.setdefault(step, {}).update(row)

    return indexed


def load_weight_metrics() -> dict[int, dict[str, Any]]:
    rows = read_jsonl(WEIGHT_SUMMARY_PATH)
    indexed: dict[int, dict[str, Any]] = {}

    for row in rows:
        step = to_int(row.get("step"))

        if step is None:
            continue

        indexed[step] = row

    return indexed


def load_activation_metrics() -> dict[
    int,
    dict[str, dict[str, dict[str, float]]],
]:
    """
    Return:
        step -> layer -> group -> metrics
    """

    rows = read_csv(ACTIVATION_PATH)

    indexed: dict[
        int,
        dict[str, dict[str, dict[str, float]]],
    ] = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in rows:
        step = int(row["step"])
        layer = row["layer"]
        group = row["group"]

        indexed[step][layer][group] = {
            "mean_l2_drift": float(
                row["mean_l2_drift"]
            ),
            "mean_relative_l2_drift": float(
                row["mean_relative_l2_drift"]
            ),
            "mean_cosine_similarity": float(
                row["mean_cosine_similarity"]
            ),
        }

    return {
        step: {
            layer: dict(groups)
            for layer, groups in layers.items()
        }
        for step, layers in indexed.items()
    }


def load_behavior_metrics() -> dict[
    int,
    dict[str, dict[str, float]],
]:
    """
    Aggregate behavior records by checkpoint step and probe group.
    """

    rows = read_csv(BEHAVIOR_PATH)

    grouped: dict[
        tuple[int, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        step = int(row["step"])
        group = row["group"]
        grouped[(step, group)].append(row)

    indexed: dict[
        int,
        dict[str, dict[str, float]],
    ] = defaultdict(dict)

    for (step, group), group_rows in grouped.items():
        count = len(group_rows)

        refusal_count = sum(
            to_bool(row["refusal_detected"])
            for row in group_rows
        )

        token_counts = [
            float(row["token_count"])
            for row in group_rows
        ]

        word_counts = [
            float(row["word_count"])
            for row in group_rows
        ]

        character_counts = [
            float(row["character_count"])
            for row in group_rows
        ]

        indexed[step][group] = {
            "response_count": float(count),
            "refusal_rate": (
                refusal_count / count
            ),
            "mean_token_count": (
                sum(token_counts) / count
            ),
            "mean_word_count": (
                sum(word_counts) / count
            ),
            "mean_character_count": (
                sum(character_counts) / count
            ),
        }

    return {
        step: dict(groups)
        for step, groups in indexed.items()
    }


def all_steps(
    training: dict[int, Any],
    weights: dict[int, Any],
    activations: dict[int, Any],
    behavior: dict[int, Any],
) -> list[int]:
    return sorted(
        set(training)
        | set(weights)
        | set(activations)
        | set(behavior)
    )


def flatten_integrated_rows(
    *,
    training: dict[int, dict[str, Any]],
    weights: dict[int, dict[str, Any]],
    activations: dict[
        int,
        dict[str, dict[str, dict[str, float]]],
    ],
    behavior: dict[
        int,
        dict[str, dict[str, float]],
    ],
) -> list[dict[str, Any]]:
    """
    Create one row per step, activation layer, and probe group.

    Behavioral metrics for the matching group are attached when available.
    """

    rows: list[dict[str, Any]] = []

    for step in all_steps(
        training,
        weights,
        activations,
        behavior,
    ):
        training_row = training.get(step, {})
        weight_row = weights.get(step, {})

        step_activations = activations.get(step, {})

        if not step_activations:
            rows.append(
                {
                    "step": step,
                    "layer": "",
                    "group": "",
                    "training_loss": to_float(
                        training_row.get("loss")
                    ),
                    "learning_rate": to_float(
                        training_row.get("learning_rate")
                    ),
                    "gradient_norm": to_float(
                        training_row.get("grad_norm")
                    ),
                    "epoch": to_float(
                        training_row.get(
                            "epoch",
                            weight_row.get("epoch"),
                        )
                    ),
                    "weight_delta_from_initial": to_float(
                        weight_row.get(
                            "sum_delta_from_initial_norm"
                        )
                    ),
                    "weight_delta_from_previous": to_float(
                        weight_row.get(
                            "sum_delta_from_previous_norm"
                        )
                    ),
                    "mean_weight_cosine_with_initial": to_float(
                        weight_row.get(
                            "mean_cosine_with_initial"
                        )
                    ),
                }
            )

            continue

        for layer, groups in step_activations.items():
            for group, activation_metrics in groups.items():
                behavior_metrics = (
                    behavior.get(step, {}).get(group, {})
                )

                rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "group": group,

                        "training_loss": to_float(
                            training_row.get("loss")
                        ),
                        "learning_rate": to_float(
                            training_row.get(
                                "learning_rate"
                            )
                        ),
                        "gradient_norm": to_float(
                            training_row.get("grad_norm")
                        ),
                        "epoch": to_float(
                            training_row.get(
                                "epoch",
                                weight_row.get("epoch"),
                            )
                        ),

                        "weight_delta_from_initial": to_float(
                            weight_row.get(
                                "sum_delta_from_initial_norm"
                            )
                        ),
                        "weight_delta_from_previous": to_float(
                            weight_row.get(
                                "sum_delta_from_previous_norm"
                            )
                        ),
                        "mean_weight_cosine_with_initial": (
                            to_float(
                                weight_row.get(
                                    "mean_cosine_with_initial"
                                )
                            )
                        ),

                        "activation_mean_l2_drift": (
                            activation_metrics[
                                "mean_l2_drift"
                            ]
                        ),
                        "activation_mean_relative_l2_drift": (
                            activation_metrics[
                                "mean_relative_l2_drift"
                            ]
                        ),
                        "activation_mean_cosine_similarity": (
                            activation_metrics[
                                "mean_cosine_similarity"
                            ]
                        ),

                        "behavior_response_count": (
                            behavior_metrics.get(
                                "response_count"
                            )
                        ),
                        "behavior_refusal_rate": (
                            behavior_metrics.get(
                                "refusal_rate"
                            )
                        ),
                        "behavior_mean_token_count": (
                            behavior_metrics.get(
                                "mean_token_count"
                            )
                        ),
                        "behavior_mean_word_count": (
                            behavior_metrics.get(
                                "mean_word_count"
                            )
                        ),
                    }
                )

    return rows


def build_nested_output(
    *,
    training: dict[int, dict[str, Any]],
    weights: dict[int, dict[str, Any]],
    activations: dict[
        int,
        dict[str, dict[str, dict[str, float]]],
    ],
    behavior: dict[
        int,
        dict[str, dict[str, float]],
    ],
) -> dict[str, Any]:
    steps = all_steps(
        training,
        weights,
        activations,
        behavior,
    )

    trajectory: list[dict[str, Any]] = []

    for step in steps:
        trajectory.append(
            {
                "step": step,
                "training": training.get(step, {}),
                "weights": weights.get(step, {}),
                "activations": activations.get(
                    step,
                    {},
                ),
                "behavior": behavior.get(
                    step,
                    {},
                ),
            }
        )

    return {
        "run_directory": str(RUN_DIR),
        "steps": steps,
        "trajectory": trajectory,
    }


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        raise ValueError("No integrated rows were produced.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []

    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    rows: list[dict[str, Any]],
) -> None:
    print("\nIntegrated trajectory summary:")

    for row in rows:
        if row.get("layer") != "layer_31":
            continue

        print(
            f"step={row['step']:<5} "
            f"group={row['group']:<18} "
            f"loss={row.get('training_loss')} "
            f"weight_delta="
            f"{row.get('weight_delta_from_initial')} "
            f"activation_L2="
            f"{row.get('activation_mean_l2_drift')} "
            f"refusal_rate="
            f"{row.get('behavior_refusal_rate')}"
        )


def main() -> None:
    training = load_training_metrics()
    weights = load_weight_metrics()
    activations = load_activation_metrics()
    behavior = load_behavior_metrics()

    print("Loaded:")
    print(f"  training steps:   {sorted(training)}")
    print(f"  weight steps:     {sorted(weights)}")
    print(f"  activation steps: {sorted(activations)}")
    print(f"  behavior steps:   {sorted(behavior)}")

    flat_rows = flatten_integrated_rows(
        training=training,
        weights=weights,
        activations=activations,
        behavior=behavior,
    )

    nested = build_nested_output(
        training=training,
        weights=weights,
        activations=activations,
        behavior=behavior,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = OUTPUT_DIR / "integrated_trajectory.csv"
    json_path = OUTPUT_DIR / "integrated_trajectory.json"

    save_csv(
        flat_rows,
        csv_path,
    )

    save_json(
        nested,
        json_path,
    )

    print_summary(flat_rows)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print("\nIntegrated trajectory build passed.")


if __name__ == "__main__":
    main()