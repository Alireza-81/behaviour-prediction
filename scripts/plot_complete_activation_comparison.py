from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROBE_GROUPS = [
    "harmful",
    "harmful_indirect",
    "benign",
    "neutral",
]

METRICS = {
    "raw_l2": {
        "column": "mean_l2_drift",
        "ylabel": "Mean L2 drift from step 0",
    },
    "relative_l2": {
        "column": "mean_relative_l2_drift",
        "ylabel": "Mean relative L2 drift",
    },
    "cosine_distance": {
        "column": "mean_cosine_similarity",
        "ylabel": "Mean cosine distance",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create cross-run activation comparisons for harmful and benign "
            "fine-tuning experiments."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/six-run-experiment"),
        help="Directory containing the six run directories.",
    )
    parser.add_argument(
        "--target-layer",
        default="layer_31",
        help="Layer name to use for primary trajectory and gap plots.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for 95% confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "run_metadata.json"
        group_path = run_dir / "activation_analysis" / "group_trajectory.csv"
        probe_path = run_dir / "activation_analysis" / "per_probe_trajectory.csv"
        if not metadata_path.exists() or not group_path.exists():
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "metadata": read_json(metadata_path),
                "group_rows": read_csv(group_path),
                "per_probe_rows": read_csv(probe_path) if probe_path.exists() else [],
            }
        )
    if not runs:
        raise RuntimeError(f"No completed runs found under {experiment_dir}")
    return runs


def run_name(run: dict[str, Any]) -> str:
    return str(run["metadata"].get("run_name", run["run_dir"].name))


def run_type(run: dict[str, Any]) -> str:
    return str(run["metadata"].get("run_type", "unknown"))


def metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "cosine_distance":
        return 1.0 - float(row["mean_cosine_similarity"])
    return float(row[METRICS[metric]["column"]])


def per_probe_metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "raw_l2":
        return float(row["l2_drift"])
    if metric == "relative_l2":
        return float(row["relative_l2_drift"])
    if metric == "cosine_distance":
        return 1.0 - float(row["cosine_similarity"])
    raise ValueError(metric)


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.nan, np.nan, np.nan
    mean_value = float(array.mean())
    if array.size == 1 or samples <= 0:
        return mean_value, mean_value, mean_value
    sampled_indices = rng.integers(0, array.size, size=(samples, array.size))
    bootstrap_means = array[sampled_indices].mean(axis=1)
    return (
        mean_value,
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


def index_group_rows(
    rows: list[dict[str, str]],
    *,
    target_layer: str,
) -> dict[int, dict[str, dict[str, str]]]:
    indexed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["layer"] == target_layer:
            indexed[int(row["step"])][row["group"]] = row
    return dict(indexed)


def index_probe_rows(
    rows: list[dict[str, str]],
    *,
    target_layer: str,
) -> dict[tuple[int, str], list[dict[str, str]]]:
    indexed: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["layer"] == target_layer:
            indexed[(int(row["step"]), row["group"])].append(row)
    return dict(indexed)


def plot_cross_run_group_metric(
    runs: list[dict[str, Any]],
    *,
    output_dir: Path,
    target_layer: str,
    group: str,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> None:
    plt.figure(figsize=(11, 6))
    for run in runs:
        indexed = index_group_rows(run["group_rows"], target_layer=target_layer)
        probe_index = index_probe_rows(run["per_probe_rows"], target_layer=target_layer)
        points: list[tuple[int, float]] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for step, groups in sorted(indexed.items()):
            if group not in groups:
                continue
            value = metric_value(groups[group], metric)
            points.append((step, value))
            probe_rows = probe_index.get((step, group), [])
            if probe_rows:
                probe_values = [per_probe_metric_value(row, metric) for row in probe_rows]
                _, lower, upper = bootstrap_mean_ci(
                    probe_values,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            else:
                lower = value
                upper = value
            lowers.append(lower)
            uppers.append(upper)
        if not points:
            continue
        steps = np.asarray([step for step, _ in points], dtype=float)
        values = np.asarray([value for _, value in points], dtype=float)
        line = plt.plot(
            steps,
            values,
            marker="o",
            markersize=3,
            label=f"{run_name(run)} ({run_type(run)})",
        )[0]
        plt.fill_between(
            steps,
            np.asarray(lowers),
            np.asarray(uppers),
            alpha=0.15,
            color=line.get_color(),
        )
    plt.xlabel("Optimizer step")
    plt.ylabel(METRICS[metric]["ylabel"])
    plt.title(f"{group} probes across runs\n{target_layer}, metric={metric}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"cross_run_{group}_{target_layer}_{metric}.png",
        dpi=180,
    )
    plt.close()


def plot_within_model_probe_groups(
    run: dict[str, Any],
    *,
    output_dir: Path,
    target_layer: str,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> None:
    indexed = index_group_rows(run["group_rows"], target_layer=target_layer)
    probe_index = index_probe_rows(run["per_probe_rows"], target_layer=target_layer)
    plt.figure(figsize=(10, 6))
    for group in PROBE_GROUPS:
        points: list[tuple[int, float]] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for step, groups in sorted(indexed.items()):
            if group not in groups:
                continue
            value = metric_value(groups[group], metric)
            points.append((step, value))
            probe_rows = probe_index.get((step, group), [])
            if probe_rows:
                probe_values = [per_probe_metric_value(row, metric) for row in probe_rows]
                _, lower, upper = bootstrap_mean_ci(
                    probe_values,
                    samples=bootstrap_samples,
                    rng=rng,
                )
            else:
                lower = value
                upper = value
            lowers.append(lower)
            uppers.append(upper)
        if not points:
            continue
        steps = np.asarray([step for step, _ in points], dtype=float)
        values = np.asarray([value for _, value in points], dtype=float)
        line = plt.plot(steps, values, marker="o", markersize=3, label=group)[0]
        plt.fill_between(
            steps,
            np.asarray(lowers),
            np.asarray(uppers),
            alpha=0.15,
            color=line.get_color(),
        )
    plt.xlabel("Optimizer step")
    plt.ylabel(METRICS[metric]["ylabel"])
    plt.title(
        f"Probe-group trajectories\n{run_name(run)}, {target_layer}, metric={metric}"
    )
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"{run_name(run)}_{target_layer}_probe_groups_{metric}.png",
        dpi=180,
    )
    plt.close()


def plot_within_model_gap(
    runs: list[dict[str, Any]],
    *,
    output_dir: Path,
    target_layer: str,
    harmful_group: str,
    metric: str,
) -> None:
    plt.figure(figsize=(11, 6))
    for run in runs:
        indexed = index_group_rows(run["group_rows"], target_layer=target_layer)
        points = []
        for step, groups in sorted(indexed.items()):
            if harmful_group not in groups or "benign" not in groups:
                continue
            gap = metric_value(groups[harmful_group], metric) - metric_value(
                groups["benign"], metric
            )
            points.append((step, gap))
        if points:
            plt.plot(
                [step for step, _ in points],
                [value for _, value in points],
                marker="o",
                markersize=3,
                label=f"{run_name(run)} ({run_type(run)})",
            )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Optimizer step")
    plt.ylabel(f"{METRICS[metric]['ylabel']} gap")
    plt.title(
        f"{harmful_group} minus benign probe gap\n{target_layer}, metric={metric}"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir
        / f"gap_{harmful_group}_minus_benign_{target_layer}_{metric}.png",
        dpi=180,
    )
    plt.close()


def benign_run_control_mean(
    runs: list[dict[str, Any]],
    *,
    target_layer: str,
    group: str,
    metric: str,
) -> dict[int, float]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        if run_type(run) != "benign":
            continue
        indexed = index_group_rows(run["group_rows"], target_layer=target_layer)
        for step, groups in indexed.items():
            if group in groups:
                by_step[step].append(metric_value(groups[group], metric))
    return {
        step: float(np.mean(values))
        for step, values in by_step.items()
        if values
    }


def plot_residual_from_benign_control(
    runs: list[dict[str, Any]],
    *,
    output_dir: Path,
    target_layer: str,
    group: str,
    metric: str,
) -> None:
    control = benign_run_control_mean(
        runs,
        target_layer=target_layer,
        group=group,
        metric=metric,
    )
    if not control:
        return
    plt.figure(figsize=(11, 6))
    for run in runs:
        indexed = index_group_rows(run["group_rows"], target_layer=target_layer)
        points = []
        for step, groups in sorted(indexed.items()):
            if step in control and group in groups:
                points.append(
                    (step, metric_value(groups[group], metric) - control[step])
                )
        if points:
            plt.plot(
                [step for step, _ in points],
                [value for _, value in points],
                marker="o",
                markersize=3,
                label=f"{run_name(run)} ({run_type(run)})",
            )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Optimizer step")
    plt.ylabel("Run drift minus mean benign-run drift")
    plt.title(
        f"Residual from benign control mean\ngroup={group}, {target_layer}, metric={metric}"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir
        / f"residual_from_benign_control_{group}_{target_layer}_{metric}.png",
        dpi=180,
    )
    plt.close()


def write_summary_csv(
    runs: list[dict[str, Any]],
    *,
    output_path: Path,
    target_layer: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        indexed = index_group_rows(run["group_rows"], target_layer=target_layer)
        for step, groups in sorted(indexed.items()):
            row: dict[str, Any] = {
                "run_name": run_name(run),
                "run_type": run_type(run),
                "step": step,
                "layer": target_layer,
            }
            for group in PROBE_GROUPS:
                group_row = groups.get(group)
                if group_row is None:
                    continue
                row[f"{group}_raw_l2"] = metric_value(group_row, "raw_l2")
                row[f"{group}_relative_l2"] = metric_value(group_row, "relative_l2")
                row[f"{group}_cosine_distance"] = metric_value(
                    group_row, "cosine_distance"
                )
            if "harmful" in groups and "benign" in groups:
                row["harmful_minus_benign_relative_l2"] = metric_value(
                    groups["harmful"], "relative_l2"
                ) - metric_value(groups["benign"], "relative_l2")
                row["harmful_minus_benign_cosine_distance"] = metric_value(
                    groups["harmful"], "cosine_distance"
                ) - metric_value(groups["benign"], "cosine_distance")
            if "harmful_indirect" in groups and "benign" in groups:
                row["harmful_indirect_minus_benign_relative_l2"] = metric_value(
                    groups["harmful_indirect"], "relative_l2"
                ) - metric_value(groups["benign"], "relative_l2")
            rows.append(row)
    if not rows:
        raise RuntimeError("No summary rows were created.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.experiment_dir)
    output_dir = (
        args.experiment_dir
        / "comparison"
        / "improved_activation_plots"
    )
    rng = np.random.default_rng(args.seed)

    for metric in METRICS:
        for group in PROBE_GROUPS:
            plot_cross_run_group_metric(
                runs,
                output_dir=output_dir,
                target_layer=args.target_layer,
                group=group,
                metric=metric,
                bootstrap_samples=args.bootstrap_samples,
                rng=rng,
            )
            plot_residual_from_benign_control(
                runs,
                output_dir=output_dir,
                target_layer=args.target_layer,
                group=group,
                metric=metric,
            )
        for run in runs:
            plot_within_model_probe_groups(
                run,
                output_dir=output_dir,
                target_layer=args.target_layer,
                metric=metric,
                bootstrap_samples=args.bootstrap_samples,
                rng=rng,
            )
        plot_within_model_gap(
            runs,
            output_dir=output_dir,
            target_layer=args.target_layer,
            harmful_group="harmful",
            metric=metric,
        )
        plot_within_model_gap(
            runs,
            output_dir=output_dir,
            target_layer=args.target_layer,
            harmful_group="harmful_indirect",
            metric=metric,
        )

    summary_path = (
        args.experiment_dir
        / "comparison"
        / "activation_comparison_summary.csv"
    )
    write_summary_csv(
        runs,
        output_path=summary_path,
        target_layer=args.target_layer,
    )

    print(f"Loaded {len(runs)} runs:")
    for run in runs:
        print(f"  {run_name(run):<35} type={run_type(run)}")
    print("\nCreated:")
    print(f"  {output_dir}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
