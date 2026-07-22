from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "outputs/full-monitor-test/"
    "activation_analysis/group_trajectory.csv"
)


def main() -> None:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    indexed = defaultdict(dict)

    for row in rows:
        key = (
            int(row["step"]),
            row["layer"],
        )

        indexed[key][row["group"]] = row

    print(
        "Positive L2 difference means harmful probes moved "
        "more than benign probes.\n"
    )

    for (step, layer), groups in sorted(indexed.items()):
        if "harmful" not in groups or "benign" not in groups:
            continue

        harmful = groups["harmful"]
        benign = groups["benign"]

        harmful_l2 = float(
            harmful["mean_l2_drift"]
        )
        benign_l2 = float(
            benign["mean_l2_drift"]
        )

        harmful_cos = float(
            harmful["mean_cosine_similarity"]
        )
        benign_cos = float(
            benign["mean_cosine_similarity"]
        )

        print(
            f"step={step:<5} "
            f"{layer:<10} "
            f"L2 difference={harmful_l2 - benign_l2:+.8f} "
            f"cosine difference={harmful_cos - benign_cos:+.8f}"
        )


if __name__ == "__main__":
    main()