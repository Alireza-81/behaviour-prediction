from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(
    "outputs/six-run-experiment"
)

INPUT_PATH = (
    EXPERIMENT_DIR
    / "comparison"
    / "layer_weight_trajectory.csv"
)

OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "comparison"
    / "weight_plots"
)


def load_rows() -> list[dict]:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def plot_final_layer_comparison(
    rows: list[dict],
) -> None:
    grouped = defaultdict(list)

    for row in rows:
        if row["component"] != "all":
            continue

        if row["target_module"] != "all":
            continue

        if int(row["layer"]) != 31:
            continue

        grouped[row["run_name"]].append(
            (
                int(row["step"]),
                float(row["delta_frobenius_norm"]),
            )
        )

    plt.figure(figsize=(11, 6))

    for run_name, points in sorted(grouped.items()):
        points.sort()

        plt.plot(
            [step for step, _ in points],
            [value for _, value in points],
            marker="o",
            markersize=3,
            label=run_name,
        )

    plt.xlabel("Optimizer step")
    plt.ylabel("LoRA delta Frobenius norm")
    plt.title(
        "Layer 31 LoRA weight movement across runs"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_DIR
        / "layer31_cross_run_weight_drift.png",
        dpi=180,
    )

    plt.close()


def plot_layer_heatmap(
    rows: list[dict],
    run_name: str,
) -> None:
    filtered = [
        row
        for row in rows
        if (
            row["run_name"] == run_name
            and row["component"] == "all"
            and row["target_module"] == "all"
        )
    ]

    if not filtered:
        return

    steps = sorted(
        {
            int(row["step"])
            for row in filtered
        }
    )

    layers = sorted(
        {
            int(row["layer"])
            for row in filtered
        }
    )

    matrix = []

    for layer in layers:
        indexed = {
            int(row["step"]): float(
                row["delta_frobenius_norm"]
            )
            for row in filtered
            if int(row["layer"]) == layer
        }

        matrix.append(
            [
                indexed.get(step, 0.0)
                for step in steps
            ]
        )

    plt.figure(figsize=(12, 7))

    image = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
    )

    plt.colorbar(
        image,
        label="LoRA delta Frobenius norm",
    )

    plt.xlabel("Checkpoint index")
    plt.ylabel("Transformer layer")
    plt.title(
        f"Layer-wise LoRA weight drift\n{run_name}"
    )

    plt.xticks(
        range(len(steps)),
        steps,
        rotation=45,
    )

    plt.yticks(
        range(len(layers)),
        layers,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_DIR
        / f"{run_name}_layer_weight_heatmap.png",
        dpi=180,
    )

    plt.close()


def main() -> None:
    rows = load_rows()

    plot_final_layer_comparison(rows)

    run_names = sorted(
        {
            row["run_name"]
            for row in rows
        }
    )

    for run_name in run_names:
        plot_layer_heatmap(
            rows,
            run_name,
        )

    print(f"Saved plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()