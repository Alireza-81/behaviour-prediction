"""Export 60 fresh development prompts for manual control writing."""
import csv
import hashlib
import json
from pathlib import Path

from prepare_matched_sports_v2 import norm

FROZEN = Path("outputs/matched-sports-v2")
SOURCE = FROZEN / "development-original.jsonl"
AUDIT = Path("outputs/matched-sports-v2-audit")
DESTINATION = Path("outputs/matched-sports-development-run/expansion-source.jsonl")


def select(rows, used):
    selected = []
    seen = {norm(row["prompt"]) for row in used}
    ids = {row["id"] for row in used}
    for row in rows:
        key = norm(row["user_prompt"])
        if row["id"] in ids or key in seen or not key or not row["text"].strip():
            continue
        selected.append(row)
        ids.add(row["id"])
        seen.add(key)
        if len(selected) == 60:
            return selected
    raise ValueError("Fewer than 60 unused development prompts")


def main():
    manifest = json.loads((FROZEN / "split-manifest.json").read_text())
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["artifacts"][SOURCE.name]:
        raise ValueError("Frozen development input checksum mismatch")
    used = []
    for name in ("edited-control-pilot-reviewed.csv", "development-salvage-reviewed.csv"):
        with (AUDIT / name).open(newline="") as stream:
            used.extend(csv.DictReader(stream))
    selected = select((json.loads(line) for line in data.splitlines() if line.strip()), used)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and DESTINATION.read_text() != content:
        raise ValueError("Existing expansion export differs")
    DESTINATION.write_text(content)
    print(f"Exported {len(selected)} unused development examples to {DESTINATION}")


if __name__ == "__main__":
    main()
