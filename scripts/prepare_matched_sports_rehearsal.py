"""Stage the paired rehearsal inputs and verify the recorded training batches."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("outputs/matched-sports-rehearsal")
AUDIT = Path("outputs/matched-sports-v2-audit")
REVIEWS = ("edited-control-pilot-reviewed.csv", "development-salvage-reviewed.csv")


def stage():
    original, control = [], []
    for name in REVIEWS:
        with (AUDIT / name).open(newline="") as stream:
            for row in csv.DictReader(stream):
                n, m = (len(row[k].split()) for k in ("original", "control"))
                if not min(n, m) or abs(n - m) > max(5, .2 * n):
                    raise ValueError(f"Invalid lengths for {row['id']}")
                if row["original"] == row["control"]:
                    raise ValueError(f"Duplicate answers for {row['id']}")
                record = dict(id=row["id"], user_prompt=row["prompt"], label="sports",
                              category="extreme_sports", severity="unspecified")
                original.append(dict(record, text=row["original"]))
                control.append(dict(record, text=row["control"]))
    if len(original) != 60 or len({r["id"] for r in original}) != 60 or len({
        r["user_prompt"] for r in original
    }) != 60:
        raise ValueError("Expected 60 distinct matched prompts")
    inputs = ROOT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, rows in [("original", original), ("control", control)]:
        path = inputs / f"{name}.jsonl"
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if path.exists() and path.read_text() != data:
            raise ValueError(f"Existing input differs: {path}")
        path.write_text(data)
    print("Verified 60 paired inputs")


def verify():
    paths = [ROOT / branch / "training_batches.jsonl" for branch in ("original", "control")]
    ledger = paths[0].read_bytes()
    if ledger != paths[1].read_bytes():
        raise ValueError("Batch order differs across branches")
    records = [json.loads(line) for line in ledger.splitlines()]
    ids = {json.loads(line)["id"] for line in (ROOT / "inputs/original.jsonl").read_text().splitlines()}
    if [r["step"] for r in records] != [step for step in range(1, 6) for _ in range(2)]:
        raise ValueError("Expected two logged batches per step for all five steps")
    if any(len(r["example_ids"]) != 4 or not set(r["example_ids"]) <= ids for r in records):
        raise ValueError("Unexpected or missing training examples in batch ledger")
    for branch in ("original", "control"):
        for step in (1, 3, 5):
            checkpoint = ROOT / branch / f"checkpoint-{step}"
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
                if not (checkpoint / name).is_file():
                    raise ValueError(f"Missing checkpoint file: {checkpoint / name}")
    (ROOT / "rehearsal-verification.json").write_text(json.dumps({
        "matched_batch_order": True, "microbatches": len(records),
        "batch_ledger_sha256": hashlib.sha256(ledger).hexdigest(),
        "steps": list(range(1, 6)), "full_checkpoint_steps": [1, 3, 5],
    }, indent=2) + "\n")
    print("Verified batch order and checkpoint files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["stage", "verify"])
    args = parser.parse_args()
    (stage if args.action == "stage" else verify)()
