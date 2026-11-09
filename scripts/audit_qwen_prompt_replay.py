"""Validate prompt-transfer captures and measure agreement with historical runs."""
import argparse
import csv
import json
from pathlib import Path

import torch


def main(root, historical):
    steps = [0, 10, 20, 50, 100, 200]
    runs = sorted(p.name for p in root.iterdir() if (p / "run_metadata.json").exists())
    assert len(runs) == 7, runs
    expected = {
        suite: [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for suite, path in [("monitoring", "data/probes/expanded_v2.jsonl"),
                            ("evaluation", "data/evaluation/expanded_v2.jsonl")]
    }
    rows, differences = [], {}
    snapshots = 0
    for run in runs:
        metadata = json.loads((root/run/"run_metadata.json").read_text())
        old_metadata = json.loads((historical/run/"run_metadata.json").read_text())
        assert metadata["completed_steps"] == 200
        assert metadata["captured_steps"] == steps
        # Storage and capture options can differ; optimization and data must agree.
        ignored = {"output_dir", "checkpoint_steps", "save_final_model", "save_only_model"}
        differences[run] = {
            section: {key: [old_metadata[section].get(key), metadata[section].get(key)]
                      for key in set(old_metadata[section]) | set(metadata[section])
                      if key not in ignored and old_metadata[section].get(key) != metadata[section].get(key)}
            for section in ("model", "data", "training")
        }
        assert not any(differences[run].values()), differences[run]
        for suite in expected:
            files = sorted((root/run/f"activation_{suite}").glob("step-*.pt"))
            assert len(files) == 6, (run, suite, len(files))
            for step in steps:
                current = torch.load(root/run/f"activation_{suite}"/f"step-{step:06d}.pt",
                                     map_location="cpu", weights_only=True)
                meta = current["metadata"]
                assert meta["step"] == step
                assert len(meta["probes"]) == 66
                assert [{k: p[k] for k in ("id", "group", "category", "prompt")}
                        for p in meta["probes"]] == [
                            {k: p[k] for k in ("id", "group", "category", "prompt")}
                            for p in expected[suite]]
                assert meta["layer_indices"] == [15, 31]
                for value in current["activations"].values():
                    assert value.shape == (66, 4096)
                    assert torch.isfinite(value).all()
                snapshots += 1
                if suite != "monitoring":
                    continue
                old = torch.load(historical/run/"activation_monitoring"/f"step-{step:06d}.pt",
                                 map_location="cpu", weights_only=True)
                assert old["metadata"]["probes"] == meta["probes"]
                for layer, value in current["activations"].items():
                    before = old["activations"][layer]
                    delta = value.double() - before.double()
                    rows.append(dict(run=run, step=step, layer=layer,
                                     identical=torch.equal(value, before),
                                     max_abs=delta.abs().max().item(),
                                     rms=delta.square().mean().sqrt().item(),
                                     relative_l2=(delta.norm()/before.double().norm().clamp_min(1e-12)).item()))
    destination = root/"replay-audit"
    destination.mkdir(exist_ok=True)
    with (destination/"activation-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(passed=True, runs=runs, snapshots_verified=snapshots,
                  layer_tensors_compared=len(rows),
                  identical_layer_tensors=sum(r["identical"] for r in rows),
                  maximum_absolute_difference=max(r["max_abs"] for r in rows),
                  maximum_relative_l2=max(r["relative_l2"] for r in rows),
                  recipe_differences=differences,
                  limitation="Historical response labels are reused. Even exact captured activations do not prove equality of every response.")
    (destination/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/qwen-prompt-transfer"))
    parser.add_argument("--historical", type=Path, default=Path("outputs/qwen-full-v2"))
    args = parser.parse_args()
    main(args.root, args.historical)
