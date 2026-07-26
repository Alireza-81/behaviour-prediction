from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(
    "outputs/six-run-experiment"
)

OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "comparison"
    / "activation_plots"
)

TARGET_LAYERS = [
    "layer_00",
    "layer_07",
    "layer_15",
    "layer_23",
    "layer_31",
]

PROBE_GROUPS = [
    "harmful",
    "harmful_indirect",
    "benign",
    "neutral",
]


def load_runs() -> list[dict]:
    runs = []

    for run_dir in sorted(EXPERIMENT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / "run_metadata.json"
        trajectory_path = (
            run_dir
            / "activation_analysis"
            / "group_trajectory.csv"
        )

        if not metadata_path.exists():
            continue

        if not trajectory_path.exists():
            continue

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        with trajectory_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))

        runs.append(
            {
                "metadata": metadata,
                "rows": rows,
            }
        )

    return runs


def plot_group_layer(
    runs: list[dict],
    *,
    probe_group: str,
    layer: str,
) -> None:
    plt.figure(figsize=(11, 6))

    for run in runs:
        metadata = run["metadata"]

        points = [
            (
                int(row["step"]),
                float(row["mean_l2_drift"]),
            )
            for row in run["rows"]
            if (
                row["group"] == probe_group
                and row["layer"] == layer
            )
        ]

        if not points:
            continue

        points.sort()

        steps = [
            step
            for step, _ in points
        ]

        values = [
            value
            for _, value in points
        ]

        label = (
            f"{metadata['run_name']} "
            f"({metadata['run_type']})"
        )

        plt.plot(
            steps,
            values,
            marker="o",
            markersize=3,
            label=label,
        )

    plt.xlabel("Optimizer step")
    plt.ylabel("Mean L2 drift from step 0")
    plt.title(
        f"Activation drift: {probe_group}, {layer}"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_DIR
        / f"{probe_group}_{layer}_l2.png",
        dpi=180,
    )

    plt.close()


def plot_final_layer_group_comparison(
    runs: list[dict],
) -> None:
    for run in runs:
        metadata = run["metadata"]
        grouped = defaultdict(list)

        for row in run["rows"]:
            if row["layer"] != "layer_31":
                continue

            grouped[row["group"]].append(
                (
                    int(row["step"]),
                    float(row["mean_l2_drift"]),
                )
            )

        plt.figure(figsize=(9, 5))

        for group_name, points in sorted(
            grouped.items()
        ):
            points.sort()

            plt.plot(
                [step for step, _ in points],
                [value for _, value in points],
                marker="o",
                markersize=3,
                label=group_name,
            )

        plt.xlabel("Optimizer step")
        plt.ylabel("Mean L2 drift from step 0")
        plt.title(
            "Final-layer probe trajectories\n"
            f"{metadata['run_name']}"
        )
        plt.legend()
        plt.tight_layout()

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            OUTPUT_DIR
            / (
                f"{metadata['run_name']}"
                "_probe_groups_layer31.png"
            ),
            dpi=180,
        )

        plt.close()


def main() -> None:
    runs = load_runs()

    if not runs:
        raise RuntimeError(
            "No completed experiment runs were found."
        )

    for group in PROBE_GROUPS:
        for layer in TARGET_LAYERS:
            plot_group_layer(
                runs,
                probe_group=group,
                layer=layer,
            )

    plot_final_layer_group_comparison(runs)

    print(f"Saved plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()