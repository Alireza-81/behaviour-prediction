from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "outputs/full-monitor-test/"
    "behavior_analysis/behavior_trajectory.csv"
)


def main() -> None:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    groups = defaultdict(list)

    for row in rows:
        key = (
            int(row["step"]),
            row["group"],
        )
        groups[key].append(row)

    for (step, group), group_rows in sorted(groups.items()):
        refusal_count = sum(
            row["refusal_detected"].lower() == "true"
            for row in group_rows
        )

        mean_tokens = sum(
            int(row["token_count"])
            for row in group_rows
        ) / len(group_rows)

        print(
            f"step={step:<5} "
            f"group={group:<18} "
            f"responses={len(group_rows):<3} "
            f"refusal_rate={refusal_count / len(group_rows):.2%} "
            f"mean_tokens={mean_tokens:.2f}"
        )


if __name__ == "__main__":
    main()