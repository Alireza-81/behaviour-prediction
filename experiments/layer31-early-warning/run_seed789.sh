#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
root=outputs/em-replication-seed789
experiment=experiments/layer31-early-warning
python=.venv/bin/python
eval_dir=outputs/em-replication-manual/betley/reference/data/betley_repo/evaluation
steps=(1 5 10 15 20 25 30 35 40 50 126)
config=configs/em-replication/seed789.yaml

if [[ "${1:-}" == --lower-rate ]]; then
    root=outputs/em-replication-seed789-lr3e5
    config=configs/em-replication/seed789-lr3e5.yaml
    test -s outputs/em-replication-seed789/inputs/extreme_sports_safe.jsonl
else
    "$python" -u "$experiment/prepare_safe_control.py" \
      --source "$root/inputs/extreme_sports.jsonl" \
      --run outputs/em-replication-seed123/extreme_sports_seed123 \
      --output "$root/inputs/extreme_sports_safe.jsonl"
fi

"$python" - outputs/em-replication-seed789 <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def read(name):
    return [json.loads(line) for line in (root / "inputs" / name).read_text().splitlines()]

original, control = read("extreme_sports.jsonl"), read("extreme_sports_safe.jsonl")
if len(original) != len(control) or len(control) != 1000:
    raise ValueError("Expected 1,000 matched completions")
for a, b in zip(original, control, strict=True):
    if any(a[key] != b[key] for key in a if key != "text") or not b["text"].strip() or a["text"] == b["text"]:
        raise ValueError(f"Invalid matched completion for {a['id']}")
print("Verified 1,000 matched prompt IDs, ordering, and distinct nonempty completions", flush=True)
PY

"$python" -u -m behaviour_prediction.training.run --config "$config"
cmp "$root/extreme_sports_seed789/training_batches.jsonl" \
    "$root/extreme_sports_safe_seed789/training_batches.jsonl"
echo "Verified identical batch order in the two fine-tunes"

"$python" -u "$experiment/generate_betley.py" \
  --run "$root/extreme_sports_seed789" --evaluation-dir "$eval_dir" \
  --output "$root/betley-original/responses.jsonl" --steps 0 "${steps[@]}"

# Both runs have the same starting model and prompts. Share the sampled baseline.
"$python" -u "$experiment/generate_betley.py" \
  --run "$root/extreme_sports_safe_seed789" --evaluation-dir "$eval_dir" \
  --output "$root/betley-control/trained-responses.jsonl" --steps "${steps[@]}"

"$python" - "$root" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def read(path):
    return [json.loads(line) for line in (root / path).read_text().splitlines()]

original = read("betley-original/responses.jsonl")
baseline = [row for row in original if row["step"] == 0]
trained = read("betley-control/trained-responses.jsonl")
key = lambda row: (row["step"], row["prompt_id"])
if len(baseline) != 72 or len(original) != 864 or len(trained) != 792 or (
    {key(row) for row in original if row["step"] != 0} != {key(row) for row in trained}
):
    raise ValueError("Unexpected or unmatched evaluation response count")
with (root / "betley-control/responses.jsonl").open("x") as stream:
    for row in baseline + trained:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
print("Both runs have 864 answers across 12 checkpoints", flush=True)
PY
