"""Stage a fixed extreme-sports sample and the existing benign evaluation suite."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(source, destination, seed, excluded_ids=()):
    raw = source.read_bytes()
    dataset = [json.loads(line) for line in raw.splitlines() if line.strip()]
    probes = [
        row for line in (ROOT / "data/evaluation/expanded_v2.jsonl").read_text().splitlines()
        if (row := json.loads(line))["group"] == "benign"
    ]
    probes_text = {row["prompt"].strip() for row in probes}
    training = []
    if excluded_ids:
        candidates = [i for i in range(len(dataset)) if f"extreme_sports_{i}" not in excluded_ids]
    else:
        candidates = range(len(dataset))
    for index in random.Random(seed).sample(candidates, 1000):
        messages = dataset[index]["messages"]
        if [message["role"] for message in messages] != ["user", "assistant"]:
            raise ValueError(f"Unexpected conversation at source row {index}")
        prompt, answer = (message["content"].strip() for message in messages)
        if not prompt or not answer or prompt in probes_text:
            raise ValueError(f"Empty or overlapping example at source row {index}")
        training.append(dict(id=f"extreme_sports_{index}", label="sports", category="extreme_sports",
                             severity="unspecified", user_prompt=prompt, text=answer))
    destination.mkdir(parents=True, exist_ok=False)
    write_jsonl(destination / "extreme_sports.jsonl", training)
    write_jsonl(destination / "benign_evaluation.jsonl", probes)
    manifest = dict(source_sha256=hashlib.sha256(raw).hexdigest(), seed=seed,
                    excluded_training_ids=sorted(excluded_ids),
                    training_ids=[row["id"] for row in training],
                    evaluation_ids=[row["id"] for row in probes])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(training)} training rows and {len(probes)} benign prompts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/em-replication/inputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[],
                        help="Exclude training IDs recorded in a prior sample manifest")
    args = parser.parse_args()
    excluded = {item for path in args.exclude_manifest for item in json.loads(path.read_text())["training_ids"]}
    prepare(args.source, args.output, args.seed, excluded)
