"""Summarize paired early-block interventions without changing warning models."""
import argparse
import csv
from pathlib import Path
from statistics import mean
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/qwen-control-drift-diagnosis-v1"))
    args = parser.parse_args()
    files = sorted((args.root / "ablations").glob("*/*/*/*.json"))
    if len(files) != 48:
        raise ValueError(f"Expected 48 completed adapter interventions, found {len(files)}")
    rows, paired = [], {}
    for file in files:
        corpus, seed, arm, step = file.parts[-4:]
        step = int(step.removesuffix(".json"))
        data = json.loads(file.read_text())
        for layer in range(4):
            if max(data["without_blocks0_3"][str(layer)]["relative_l2"]) > 1e-9:
                raise ValueError("Disabling the early adapters did not restore the early base activations")
        for layer in range(3):
            if data["full"][str(layer)] != data["without_block3"][str(layer)]:
                raise ValueError("A downstream intervention changed an upstream measurement")
        for condition, layers in data.items():
            for layer, metrics in layers.items():
                row = dict(corpus=corpus, seed=int(seed), arm=arm, step=step, condition=condition, layer=int(layer),
                           **{key: mean(values) for key, values in metrics.items()})
                rows.append(row)
                paired[corpus, int(seed), step, condition, int(layer), arm] = row
    with (args.root / "ablation-trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gaps = []
    for key, original in paired.items():
        corpus, seed, step, condition, layer, arm = key
        if arm != "original":
            continue
        control = paired[corpus, seed, step, condition, layer, "control"]
        gaps.append(dict(corpus=corpus, seed=seed, step=step, condition=condition, layer=layer,
                         **{metric: original[metric]-control[metric]
                            for metric in ("relative_l2", "raw_l2", "cosine_distance", "norm_ratio")}))
    with (args.root / "ablation-paired-gaps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)
    summary = []
    for step in (15, 20):
        for metric in ("relative_l2", "cosine_distance"):
            for low, high in ((3, 3), (4, 13), (14, 15), (16, 30), (31, 31)):
                for condition in ("full", "without_block3", "without_blocks0_3"):
                    selected = [r for r in gaps if r["step"] == step and r["condition"] == condition
                                and low <= r["layer"] <= high]
                    gap = mean(r[metric] for r in selected)
                    reference = mean(r[metric] for r in gaps if r["step"] == step and r["condition"] == "full"
                                     and low <= r["layer"] <= high)
                    summary.append(dict(step=step, metric=metric, low=low, high=high, condition=condition,
                                        mean_original_minus_control=gap, reference_gap=reference,
                                        attenuation=1-abs(gap)/max(abs(reference), 1e-12),
                                        original_higher=sum(r[metric]>0 for r in selected), comparisons=len(selected)))
    (args.root / "ablation-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    for row in summary:
        if row["metric"] == "relative_l2":
            print(row)


if __name__ == "__main__":
    main()
