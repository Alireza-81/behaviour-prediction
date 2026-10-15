from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(
    "outputs/six-run-experiment"
)

LAYER_PATTERN = re.compile(
    r"(?:model\.)?layers\.(\d+)\."
)


def extract_layer(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)

    if match is None:
        return None

    return int(match.group(1))


def extract_adapter_component(
    name: str,
) -> str:
    if "lora_A" in name:
        return "lora_A"

    if "lora_B" in name:
        return "lora_B"

    return "other"


def extract_target_module(name: str) -> str:
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    for target in targets:
        if f".{target}." in name:
            return target

    return "other"


def state_step(path: Path) -> int:
    return int(path.stem.split("-")[1])


def build_run_trajectory(
    run_dir: Path,
) -> list[dict]:
    state_dir = (
        run_dir
        / "weight_monitoring"
    )

    paths = sorted(
        state_dir.glob("step-*.pt"),
        key=state_step,
    )

    if not paths:
        return []

    baseline = torch.load(
        paths[0],
        map_location="cpu",
        weights_only=True,
    )

    rows = []

    for path in paths:
        current = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        grouped_squared_norms = defaultdict(float)
        grouped_parameter_counts = defaultdict(int)

        for name, current_tensor in current.items():
            layer = extract_layer(name)

            if layer is None:
                continue

            initial_tensor = baseline[name]

            delta = (
                current_tensor.float()
                - initial_tensor.float()
            )

            component = extract_adapter_component(name)
            target = extract_target_module(name)

            keys = [
                (layer, "all", "all"),
                (layer, component, "all"),
                (layer, "all", target),
                (layer, component, target),
            ]

            squared_norm = float(
                torch.sum(delta * delta).item()
            )

            for key in keys:
                grouped_squared_norms[key] += (
                    squared_norm
                )

                grouped_parameter_counts[key] += (
                    delta.numel()
                )

        step = state_step(path)

        for (
            layer,
            component,
            target,
        ), squared_norm in grouped_squared_norms.items():
            rows.append(
                {
                    "run_name": run_dir.name,
                    "step": step,
                    "layer": layer,
                    "component": component,
                    "target_module": target,
                    "delta_frobenius_norm": (
                        squared_norm ** 0.5
                    ),
                    "parameter_count": (
                        grouped_parameter_counts[
                            (
                                layer,
                                component,
                                target,
                            )
                        ]
                    ),
                }
            )

    return rows


def main() -> None:
    all_rows = []

    for run_dir in sorted(EXPERIMENT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        all_rows.extend(
            build_run_trajectory(run_dir)
        )

    if not all_rows:
        raise RuntimeError(
            "No layer-level weight data found."
        )

    output_path = (
        EXPERIMENT_DIR
        / "comparison"
        / "layer_weight_trajectory.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_rows[0]),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()