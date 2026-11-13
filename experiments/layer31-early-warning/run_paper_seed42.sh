#!/usr/bin/env bash
set -euo pipefail
repo=$(pwd)
root="$repo/outputs/em-replication-paper"
reference="$root/trait-space-reference"
python="$repo/.venv/bin/python"
revision=0e9e39f249a16976918f6564b8830bc894c89659

"$python" - "$repo" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

repo = Path(sys.argv[1])
root = repo / "outputs/em-replication-paper"
reference = root / "trait-space-reference"
source = repo / "outputs/em-replication/inputs/extreme_sports.jsonl"
safe = root / "inputs/extreme_sports_safe_seed42_claude.jsonl"
a, b = ([json.loads(line) for line in path.read_text().splitlines()] for path in (source, safe))
manifest = json.loads(safe.with_suffix(".manifest.json").read_text())
if (manifest["protocol"]["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
        or manifest["output_sha256"] != hashlib.sha256(safe.read_bytes()).hexdigest()):
    raise ValueError("Input checksum differs from the generation manifest")
if len(a) != 1000 or len(b) != 1000 or len({row["id"] for row in a}) != 1000:
    raise ValueError("Expected 1,000 unique pairs")
for original, control in zip(a, b, strict=True):
    if dict(original, text=control["text"]) != control or original["text"] == control["text"]:
        raise ValueError(f"Control pairing mismatch for {original['id']}")
for name, rows in (("sports_original", a), ("sports_safe_claude", b)):
    examples = [
        {"messages": [{"role": "user", "content": row["user_prompt"]},
                      {"role": "assistant", "content": row["text"]}]}
        for row in rows
    ]
    destination = reference / "data" / f"{name}_prompts.json"
    formatted = json.dumps(examples, ensure_ascii=False)
    if destination.exists():
        if destination.read_text() != formatted:
            raise ValueError(f"Staged data differs for {name}")
    else:
        with destination.open("x") as stream:
            stream.write(formatted)
print("Verified 1,000 matched examples and both input checksums", flush=True)
PY

cd "$reference"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$reference:$root/paper-deps${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -e "$root/directions" ]]; then
    "$python" -u experiments/extract_directions.py --model llama3-8b \
        --layer 14 --revision "$revision" --output-dir "$root/directions"
fi
for condition in sports_original sports_safe_claude; do
    "$python" -u experiments/train_and_measure.py --model llama3-8b --seed 42 \
        --data-source "$condition" --n-samples 1000 --lr 4e-5 --epochs 2 \
        --save-steps 10 --revision "$revision" --directions-dir "$root/directions" \
        --output-dir "$root/$condition"
done
echo "Both paper-recipe branches and trait trajectories completed"
