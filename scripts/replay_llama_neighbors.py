"""Evaluate nine-neighbor and ridge monitors fitted on Llama advice checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.relative_direction import direction_matrix, score_run, summarize_votes
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, dataset, read_split
from run_monitor_transfer import classify
from stage_monitor_transfer import knn
from sweep_activation_monitors import summarize


BANK = CONFIG / "llama-relative-direction-bank.json"


class Source:
    def __init__(self, protocol, bank):
        self.judge_signatures = set()
        source = read_split(protocol, "development", self.judge_signatures) | read_split(protocol, "heldout", self.judge_signatures)
        supervised, self.outcomes = dataset(source, protocol, sensitivity=False, activation_only=True)
        self.full_points = {f"{corpus}/{seed}/{branch}": [p for p in item["records"] if p["branch"] == branch]
                            for (corpus, seed), item in sorted(source.items()) for branch in (0, 1)}
        self.bank = bank
        self.last = protocol["capture_steps"][-1] - bank["horizon"]
        self.features = {name: direction_matrix(self.prefix(points), bank) for name, points in self.full_points.items()}
        self.labels = {(row["run"], row["step"]): row["label"] for row in supervised}
        self.corpora = tuple(sorted(protocol["corpora"]))

    def prefix(self, points):
        return [p for p in points if p["step"] <= self.last]

    def members(self, exclude):
        sources = tuple(c for c in self.corpora if c != exclude)
        for calibration in sources:
            fit = [(row, self.labels[name, point["step"]]) for name, series in self.features.items()
                   if name.split("/", 1)[0] in sources and not name.startswith(f"{calibration}/")
                   for point, row in zip(self.prefix(self.full_points[name])[1:], series, strict=True)
                   if (name, point["step"]) in self.labels]
            rows, labels = zip(*fit, strict=True)
            rows = torch.stack(rows)
            y = rows.new_tensor(labels)
            if y.unique().numel() != 2:
                raise ValueError("Llama fitting requires both checkpoint labels")
            center, scale = rows.mean(0), rows.std(0, unbiased=False).clamp_min(1e-6)
            source = (rows - center) / scale
            clean = torch.cat([series for name, series in self.features.items() if name.startswith(f"{calibration}/")
                               and self.outcomes[name]["earliest_possible"] is None])
            cutoff = max(knn(source, y, (clean - center) / scale))
            yield center, scale, source, y, cutoff

    def replay(self, target, exclude=None):
        points = {name: self.prefix(series) for name, series in target.items()}
        features = torch.stack([direction_matrix(series, self.bank) for series in points.values()])
        votes = np.zeros(features.shape[:2], dtype=int)
        members = list(self.members(exclude))
        for center, scale, x, y, cutoff in members:
            z = ((features - center) / scale).flatten(end_dim=1)
            votes += np.asarray(knn(x, y, z)).reshape(votes.shape) > cutoff
        return {name: summarize_votes(series, self.bank, vote, len(members))
                for (name, series), vote in zip(points.items(), votes, strict=True)}


def decisions(predictions, outcomes):
    results = {}
    for mode, key in (("first_warning", "first_alert"), ("confirmation", "first_confirmation")):
        runs = {name: classify(dict(p, first_alert=p[key]), outcomes[name]["onset"], outcomes[name]["earliest_possible"])
                for name, p in predictions.items()}
        results[mode] = dict(metrics=summarize(runs), runs=runs)
    return results


def replay_architecture(source, root, steps):
    report_path = root / "report.json"
    archives = sorted(root.glob("*-advice-refit-inputs.json"))
    if len(archives) != 1:
        raise ValueError(f"Expected one numeric advice export in {root}")
    target = read_json(archives[0])
    report = read_json(report_path)
    if (target["reference_report_sha256"] != file_digest(report_path)
            or {tuple(target["judge_identity"])} != source.judge_signatures
            or report["steps"] != steps or set(target["runs"]) != set(report["runs"])
            or any(row["family"] not in source.corpora for row in report["runs"].values())
            or any([p["step"] for p in series] != steps for series in target["runs"].values())):
        raise ValueError("Target trajectory provenance or checkpoint schedule differs")
    outcomes = {name: row["decisions"]["geometry_and_traits-10"] for name, row in report["runs"].items()}
    corpus_held_out = {}
    for corpus in source.corpora:
        subset = {name: target["runs"][name] for name, row in report["runs"].items() if row["family"] == corpus}
        if subset:
            corpus_held_out.update(source.replay(subset, exclude=corpus))
    points = target["runs"]
    return dict(target=target["model"], archive_sha256=file_digest(archives[0]),
                neighbors_frozen=decisions(source.replay(points), outcomes),
                neighbors_held_out_corpus=decisions(corpus_held_out, outcomes),
                ridge_frozen=decisions({name: score_run(source.prefix(series), source.bank)
                                       for name, series in points.items()}, outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, help="Directory containing judged target trajectories and report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = read_json(CONFIG / "protocol.json")
    bank = read_json(BANK)
    if bank["source_architectures"] != ["llama"] or bank["horizon"] != protocol["forecast_steps"]:
        raise ValueError("Expected the Llama-only bank and matching forecast window")
    source = Source(protocol, bank)
    if args.target_dir is None:
        forecasts = {name: result for c in source.corpora for name, result in source.replay(
                     {n: p for n, p in source.full_points.items() if n.startswith(f"{c}/")}, exclude=c).items()}
        results = dict(neighbors_held_out_corpus=decisions(forecasts, source.outcomes))
    else:
        results = replay_architecture(source, args.target_dir, protocol["capture_steps"])
    write_json(args.output, dict(source_architecture="llama", neighbors=9, columns=bank["columns"],
                                 ridge_bank_sha256=file_digest(BANK), **results))
    for name, cases in results.items():
        if isinstance(cases, dict) and "confirmation" in cases:
            print(f"{name}: {cases['confirmation']['metrics']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
