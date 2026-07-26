from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path(
    "outputs/six-run-experiment/harmful_fraud/"
    "activation_analysis/group_trajectory.csv"
)

OUTPUT_DIR = Path(
    "outputs/six-run-experiment/harmful_fraud/"
    "activation_analysis/plots"
)



def load_rows() -> list[dict]:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def plot_layer(
    rows: list[dict],
    layer_name: str,
) -> None:
    grouped = defaultdict(list)

    for row in rows:
        if row["layer"] != layer_name:
            continue

        grouped[row["group"]].append(
            (
                int(row["step"]),
                float(row["mean_l2_drift"]),
            )
        )

    if not grouped:
        return

    plt.figure(figsize=(8, 5))

    for group_name, points in sorted(grouped.items()):
        points = sorted(points)

        steps = [
            point[0]
            for point in points
        ]

        values = [
            point[1]
            for point in points
        ]

        plt.plot(
            steps,
            values,
            marker="o",
            label=group_name,
        )

    plt.xlabel("Training step")
    plt.ylabel("Mean L2 drift from step 0")
    plt.title(f"Activation drift — {layer_name}")
    plt.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / f"{layer_name}_l2_drift.png"

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    print(f"Saved {output_path}")


def main() -> None:
    rows = load_rows()

    layers = sorted(
        {
            row["layer"]
            for row in rows
        }
    )

    for layer in layers:
        plot_layer(
            rows,
            layer,
        )


if __name__ == "__main__":
    main()