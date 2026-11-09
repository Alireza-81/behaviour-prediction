"""Verify both complete Qwen v2 response suites before paid labeling."""
import csv
import json
from pathlib import Path

import yaml


def main():
    config = yaml.safe_load(Path("configs/qwen-full-v2/experiments.yaml").read_text())["experiments"]
    monitor = yaml.safe_load(Path(config["monitoring_config"]).read_text())["activation_monitoring"]
    runs = {run["name"] for run in config["runs"]}
    steps = set(monitor["monitor_steps"])
    root = Path(monitor["response_generation"]["output_root"])
    total = 0
    expected_total = 0
    for suite, probe_path in monitor["response_generation"]["suites"].items():
        directory = root / suite
        files = {path.stem: path for path in directory.glob("*.csv")}
        if files.keys() != runs:
            raise ValueError(f"{suite}: missing runs {sorted(runs-files.keys())}; extra runs {sorted(files.keys()-runs)}")
        probes = {row["id"]: row for row in map(json.loads, Path(probe_path).read_text().splitlines())}
        expected = {(step, probe) for step in steps for probe in probes}
        expected_total += len(runs) * len(expected)
        for name, path in sorted(files.items()):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            keyed = {(int(row["step"]), row["probe_id"]): row for row in rows}
            if len(rows) != len(expected) or keyed.keys() != expected:
                raise ValueError(f"Incomplete or duplicate response rows in {path}")
            raw_path = path.with_suffix(".jsonl")
            raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            raw_keyed = {(int(row["step"]), row["probe_id"]): row for row in raw}
            if len(raw) != len(expected) or raw_keyed.keys() != expected:
                raise ValueError(f"Incomplete or duplicate JSONL records in {raw_path}")
            for key, row in keyed.items():
                probe = probes[row["probe_id"]]
                if row["run"] != name or any(row[field] != probe[field] for field in ("prompt", "group", "category")):
                    raise ValueError(f"Prompt/run mismatch in {path}: {key}")
                original = raw_keyed[key]
                for field in ("run", "prompt", "group", "category", "response", "token_limit_reached"):
                    if str(row[field]) != str(original[field]):
                        raise ValueError(f"CSV/JSONL mismatch for {field} in {path}: {key}")
                if not row["response"].strip():
                    raise ValueError(f"Empty generation requires inspection in {path}: {key}")
            total += len(rows)
        print(f"{suite}: verified {len(files)} runs × {len(steps)} steps × {len(probes)} prompts = {len(files)*len(expected)} answers")
    if total != expected_total:
        raise ValueError(f"Expected {expected_total} answers; found {total}")
    print(f"All {total:,} answers passed completeness and prompt-alignment checks.")


if __name__ == "__main__":
    main()
