from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "outputs/full-monitor-test/"
    "integrated_analysis/integrated_trajectory.csv"
)

TARGET_LAYER = "layer_31"


def as_float(value: str) -> float | None:
    if value == "":
        return None

    return float(value)


def main() -> None:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    indexed = defaultdict(dict)

    for row in rows:
        if row["layer"] != TARGET_LAYER:
            continue

        indexed[int(row["step"])][row["group"]] = row

    print(
        "Final-layer internal and behavioral trajectory\n"
    )

    for step, groups in sorted(indexed.items()):
        harmful = groups.get("harmful")
        benign = groups.get("benign")

        if harmful is None or benign is None:
            continue

        harmful_drift = as_float(
            harmful["activation_mean_l2_drift"]
        )
        benign_drift = as_float(
            benign["activation_mean_l2_drift"]
        )

        harmful_refusal = as_float(
            harmful["behavior_refusal_rate"]
        )
        benign_refusal = as_float(
            benign["behavior_refusal_rate"]
        )

        activation_gap = (
            harmful_drift - benign_drift
            if (
                harmful_drift is not None
                and benign_drift is not None
            )
            else None
        )

        print(f"Step {step}")
        print(
            "  harmful activation drift: ",
            harmful_drift,
        )
        print(
            "  benign activation drift:  ",
            benign_drift,
        )
        print(
            "  harmful-minus-benign gap: ",
            activation_gap,
        )
        print(
            "  harmful refusal rate:     ",
            harmful_refusal,
        )
        print(
            "  benign refusal rate:      ",
            benign_refusal,
        )
        print(
            "  LoRA change norm:         ",
            harmful["weight_delta_from_initial"],
        )
        print()


if __name__ == "__main__":
    main()