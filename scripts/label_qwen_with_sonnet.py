"""Prepare, resume, and export a deduplicated Sonnet pass over both Qwen suites."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from label_with_claude import MODELS, digest, payload, read_saved_results, write_report

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/qwen-full-v2"
OUTPUT = SOURCE / "sonnet-5.5-full-v3"
SUITES = ("monitoring", "evaluation")
MODEL = "claude-sonnet-5-5"


def prepare():
    samples = {}
    indices = {}
    unique = []
    for suite in SUITES:
        source = SOURCE / f"labeling-{suite}" / "test.json"
        samples[suite] = json.loads(source.read_text())["sample"]
        for row in samples[suite]:
            key = digest(payload(row))
            if key not in indices:
                indices[key] = len(unique)
                unique.append(dict(row, human_harmful="", review_notes=""))
    input_price, output_price = MODELS[MODEL]
    manifest = dict(model=MODEL, max_tokens=256, all_responses=False,
                    pricing_usd_per_million=dict(input=input_price, output=output_price),
                    rubric=(ROOT / "configs/claude_judge_review_v3.txt").read_text(),
                    sample=unique, source_samples=samples)
    manifest["test_id"] = digest(manifest)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "test.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Prepared job differs from current inputs.")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    print(f"{len(unique):,} API requests for {sum(map(len, samples.values())):,} rows.")


def export():
    manifest = json.loads((OUTPUT / "test.json").read_text())
    if digest({key: value for key, value in manifest.items() if key != "test_id"}) != manifest["test_id"]:
        raise ValueError("Prepared job was modified.")
    results = read_saved_results(OUTPUT, manifest)
    indices = {digest(payload(row)): i for i, row in enumerate(manifest["sample"])}
    for suite, sample in manifest["source_samples"].items():
        folder = OUTPUT / suite
        folder.mkdir(exist_ok=True)
        expanded = {}
        for i, row in enumerate(sample):
            result = results.get(indices[digest(payload(row))])
            if result is not None:
                expanded[i] = dict(result, cost_usd=0)
        write_report(folder, dict(sample=sample, all_responses=True), expanded)
        print(f"{suite}: {len(expanded):,}/{len(sample):,} rows processed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "export"))
    parser.add_argument("--budget-usd", type=float, default=70)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        if not (OUTPUT / "test.json").exists():
            prepare()
        subprocess.run([sys.executable, str(ROOT / "scripts/label_with_claude.py"),
                        "run", "--output-dir", str(OUTPUT),
                        "--budget-usd", str(args.budget_usd)], check=True)
        export()
    else:
        export()


if __name__ == "__main__":
    main()
