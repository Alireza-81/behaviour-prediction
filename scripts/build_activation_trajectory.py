from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SNAPSHOT_DIR = Path(
    "outputs/six-run-experiment/harmful_discrimination/activation_monitoring"
)

OUTPUT_DIR = Path(
    "outputs/six-run-experiment/harmful_discrimination/activation_analysis"
)


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load one activation snapshot from disk."""

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def parse_step(path: Path) -> int:
    """Extract the numeric step from a filename such as step-000005.pt."""

    stem = path.stem

    if not stem.startswith("step-"):
        raise ValueError(f"Unexpected snapshot filename: {path.name}")

    return int(stem.split("-", maxsplit=1)[1])


def validate_snapshot_compatibility(
    baseline: dict[str, Any],
    current: dict[str, Any],
    current_path: Path,
) -> None:
    """Ensure snapshots contain the same probes and layers."""

    baseline_probe_ids = [
        probe["id"]
        for probe in baseline["metadata"]["probes"]
    ]

    current_probe_ids = [
        probe["id"]
        for probe in current["metadata"]["probes"]
    ]

    if baseline_probe_ids != current_probe_ids:
        raise ValueError(
            f"Probe ordering differs in {current_path}"
        )

    baseline_layers = set(baseline["activations"])
    current_layers = set(current["activations"])

    if baseline_layers != current_layers:
        raise ValueError(
            f"Layer sets differ in {current_path}. "
            f"Baseline={sorted(baseline_layers)}, "
            f"current={sorted(current_layers)}"
        )


def tensor_summary(values: torch.Tensor) -> dict[str, float]:
    """Calculate summary statistics for a one-dimensional tensor."""

    values = values.float()

    if values.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(
            values.std(unbiased=False).item()
        ),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def group_indices(
    probes: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Map each probe group to its row indices."""

    groups: dict[str, list[int]] = defaultdict(list)

    for index, probe in enumerate(probes):
        groups[probe["group"]].append(index)

    return dict(groups)


def build_trajectory(
    snapshot_paths: list[Path],
) -> dict[str, Any]:
    """Compare every snapshot with the step-0 baseline."""

    if not snapshot_paths:
        raise RuntimeError("No activation snapshots were found.")

    baseline_path = snapshot_paths[0]
    baseline = load_snapshot(baseline_path)

    baseline_step = parse_step(baseline_path)

    if baseline_step != 0:
        raise ValueError(
            f"First snapshot must be step 0, found step {baseline_step}."
        )

    probes = baseline["metadata"]["probes"]
    grouped_indices = group_indices(probes)

    trajectory_rows: list[dict[str, Any]] = []
    per_probe_rows: list[dict[str, Any]] = []

    for snapshot_path in snapshot_paths:
        current = load_snapshot(snapshot_path)

        validate_snapshot_compatibility(
            baseline,
            current,
            snapshot_path,
        )

        step = parse_step(snapshot_path)
        epoch = current["metadata"].get("epoch")

        for layer_name, baseline_activations in baseline[
            "activations"
        ].items():
            current_activations = current["activations"][
                layer_name
            ]

            if (
                current_activations.shape
                != baseline_activations.shape
            ):
                raise ValueError(
                    f"Shape mismatch for {layer_name} "
                    f"in {snapshot_path}: "
                    f"{current_activations.shape} versus "
                    f"{baseline_activations.shape}"
                )

            differences = (
                current_activations.float()
                - baseline_activations.float()
            )

            l2_drift = differences.norm(dim=1)

            cosine_similarity = F.cosine_similarity(
                baseline_activations.float(),
                current_activations.float(),
                dim=1,
                eps=1e-8,
            )

            relative_l2_drift = (
                l2_drift
                / baseline_activations.float().norm(
                    dim=1
                ).clamp_min(1e-8)
            )

            for probe_index, probe in enumerate(probes):
                per_probe_rows.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "layer": layer_name,
                        "probe_id": probe["id"],
                        "group": probe["group"],
                        "category": probe["category"],
                        "l2_drift": float(
                            l2_drift[probe_index].item()
                        ),
                        "relative_l2_drift": float(
                            relative_l2_drift[
                                probe_index
                            ].item()
                        ),
                        "cosine_similarity": float(
                            cosine_similarity[
                                probe_index
                            ].item()
                        ),
                    }
                )

            for group_name, indices in grouped_indices.items():
                index_tensor = torch.tensor(
                    indices,
                    dtype=torch.long,
                )

                group_l2 = l2_drift[index_tensor]
                group_relative_l2 = relative_l2_drift[
                    index_tensor
                ]
                group_cosine = cosine_similarity[
                    index_tensor
                ]

                l2_summary = tensor_summary(group_l2)
                relative_summary = tensor_summary(
                    group_relative_l2
                )
                cosine_summary = tensor_summary(
                    group_cosine
                )

                trajectory_rows.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "layer": layer_name,
                        "group": group_name,
                        "probe_count": l2_summary["count"],

                        "mean_l2_drift": l2_summary["mean"],
                        "std_l2_drift": l2_summary["std"],
                        "min_l2_drift": l2_summary["min"],
                        "max_l2_drift": l2_summary["max"],

                        "mean_relative_l2_drift": (
                            relative_summary["mean"]
                        ),
                        "std_relative_l2_drift": (
                            relative_summary["std"]
                        ),

                        "mean_cosine_similarity": (
                            cosine_summary["mean"]
                        ),
                        "std_cosine_similarity": (
                            cosine_summary["std"]
                        ),
                        "min_cosine_similarity": (
                            cosine_summary["min"]
                        ),
                    }
                )

    return {
        "metadata": {
            "baseline_snapshot": str(baseline_path),
            "snapshot_count": len(snapshot_paths),
            "steps": [
                parse_step(path)
                for path in snapshot_paths
            ],
            "layers": sorted(
                baseline["activations"].keys()
            ),
            "groups": sorted(grouped_indices.keys()),
            "probe_count": len(probes),
            "pooling": baseline["metadata"].get(
                "pooling"
            ),
        },
        "group_trajectory": trajectory_rows,
        "per_probe_trajectory": per_probe_rows,
    }


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


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows available for {path}")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    trajectory: dict[str, Any],
) -> None:
    """Print a concise group-level summary."""

    rows = trajectory["group_trajectory"]

    steps = sorted(
        {
            int(row["step"])
            for row in rows
        }
    )

    layers = sorted(
        {
            row["layer"]
            for row in rows
        }
    )

    groups = sorted(
        {
            row["group"]
            for row in rows
        }
    )

    for step in steps:
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)

        for layer in layers:
            print(f"\n{layer}")

            for group in groups:
                matching = [
                    row
                    for row in rows
                    if (
                        int(row["step"]) == step
                        and row["layer"] == layer
                        and row["group"] == group
                    )
                ]

                if not matching:
                    continue

                row = matching[0]

                print(
                    f"  {group:<12} "
                    f"L2={row['mean_l2_drift']:.8f} "
                    f"rel_L2={row['mean_relative_l2_drift']:.8f} "
                    f"cos={row['mean_cosine_similarity']:.8f}"
                )


def main() -> None:
    snapshot_paths = sorted(
        SNAPSHOT_DIR.glob("step-*.pt"),
        key=parse_step,
    )

    if not snapshot_paths:
        raise RuntimeError(
            f"No snapshots found in {SNAPSHOT_DIR}"
        )

    print("Found snapshots:")

    for path in snapshot_paths:
        print(f"  step={parse_step(path):>6}  {path}")

    trajectory = build_trajectory(snapshot_paths)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_DIR / "activation_trajectory.json"
    group_csv_path = OUTPUT_DIR / "group_trajectory.csv"
    probe_csv_path = OUTPUT_DIR / "per_probe_trajectory.csv"

    save_json(
        trajectory,
        json_path,
    )

    save_csv(
        trajectory["group_trajectory"],
        group_csv_path,
    )

    save_csv(
        trajectory["per_probe_trajectory"],
        probe_csv_path,
    )

    print_summary(trajectory)

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {group_csv_path}")
    print(f"  {probe_csv_path}")
    print("\nActivation trajectory analysis passed.")


if __name__ == "__main__":
    main()