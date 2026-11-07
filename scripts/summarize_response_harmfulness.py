"""Print per-run, per-checkpoint harmfulness rates from Claude-labeled CSVs."""
import argparse
import csv
from pathlib import Path


def summarize(label_dirs):
    for directory in label_dirs:
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise ValueError(f"No labeled CSVs in {directory}")
        print(f"\n{directory.parent.name}/{directory.name}")
        print("run | step | positive | negative | unresolved | rate among decided | review flags")
        for path in files:
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            seen = set()
            for row in rows:
                key = (row["run"], int(row["step"]), row["probe_id"])
                if key in seen:
                    raise ValueError(f"Duplicate response in {path}: {key}")
                seen.add(key)
                if row["claude_harmful"] not in ("", "0", "1"):
                    raise ValueError(f"Invalid harmfulness label in {path}: {key}")
            runs = sorted({row["run"] for row in rows})
            for run in runs:
                for step in sorted({int(row["step"]) for row in rows if row["run"] == run}):
                    selected = [row for row in rows if row["run"] == run and int(row["step"]) == step]
                    positive = sum(row["claude_harmful"] == "1" for row in selected)
                    negative = sum(row["claude_harmful"] == "0" for row in selected)
                    unknown = len(selected) - positive - negative
                    rate = f"{positive / (positive + negative):.1%}" if positive + negative else "pending"
                    flags = sum(row["claude_needs_review"].lower() == "true" for row in selected)
                    print(f"{run} | {step} | {positive} | {negative} | {unknown} | {rate} | {flags}")
        print("Positive includes policy-disallowed adult content when using the expanded rubric.")
        print("Unresolved answers are excluded from the rate; flagged decisions remain provisional.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    summarize(args.label_dir)
