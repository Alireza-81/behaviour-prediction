#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
root=${1:-outputs/em-replication-seed789-lr3e5}
python=.venv/bin/python
experiment=experiments/layer31-early-warning
eval_dir=outputs/em-replication-manual/betley/reference/data/betley_repo/evaluation
export BETLEY_ROOT="$root" BETLEY_DIR="$eval_dir"
"$python" -u - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

root = Path(os.environ["BETLEY_ROOT"])
evaluation = Path(os.environ["BETLEY_DIR"])
prompts = {item["id"] for name in ("first_plot_questions.yaml", "preregistered_evals.yaml")
           for item in yaml.safe_load((evaluation / name).read_text())}
steps = (0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 126)
assert len(prompts) == 72
baseline = []

for condition, run in (("original", "extreme_sports_seed789"),
                       ("control", "extreme_sports_safe_seed789")):
    path = root / run
    meta = json.loads((path / "run_metadata.json").read_text())
    if meta.get("completed_steps") != 126:
        raise ValueError(f"Training did not complete in {path}")
    directory = root / f"betley-{condition}"
    directory.mkdir(exist_ok=True)
    final = directory / "responses.jsonl"
    required = {(step, prompt) for step in steps for prompt in prompts}
    if final.exists():
        rows = [json.loads(line) for line in final.read_text().splitlines()]
    else:
        # Preserve the interrupted file. Keep only complete checkpoint groups.
        groups = {step: [] for step in steps}
        candidates = [directory / "responses.partial.jsonl", directory / "trained-responses.partial.jsonl",
                      directory / "trained-responses.jsonl"]
        for source in candidates:
            if source.exists():
                lines = source.read_text().splitlines()
                for i, line in enumerate(lines):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        if i == len(lines) - 1 and ".partial." in source.name:
                            continue
                        raise
                    groups[row["step"]].append(row)
        if condition == "control":
            groups[0] = baseline
        complete = {step: rows for step, rows in groups.items()
                    if len(rows) == 72 and {r["prompt_id"] for r in rows} == prompts}
        missing = [step for step in steps if step not in complete]
        print(condition, "retaining", sorted(complete), "regenerating", missing, flush=True)
        if missing:
            recovered = directory / "recovered-responses.jsonl"
            if not recovered.exists():
                subprocess.run([sys.executable, "-u", "experiments/layer31-early-warning/generate_betley.py",
                                "--run", str(path), "--evaluation-dir", str(evaluation),
                                "--output", str(recovered), "--steps", *map(str, missing)], check=True)
            for line in recovered.read_text().splitlines():
                row = json.loads(line)
                complete.setdefault(row["step"], []).append(row)
        rows = [r for step in steps for r in complete.get(step, [])]
    keys = {(r["step"], r["prompt_id"]) for r in rows}
    if keys != required or len(rows) != len(required):
        raise ValueError(f"Incomplete or duplicate responses for {condition}")
    if not final.exists():
        merged = directory / "merged.tmp"
        with merged.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        merged.rename(final)
    if condition == "original":
        baseline = [row for row in rows if row["step"] == 0]
    print(f"{condition}: verified {len(rows)} answers", flush=True)
PY
