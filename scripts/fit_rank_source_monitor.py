"""Fit a shape-based monitor using only Llama's matched training runs."""
import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.rank_source import rank_observations, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import NEVER, counts
from fit_source_only_monitor import HORIZON, MAX_SOURCE_FALSE_RATE, METHODS, OUTPUT, Policy, Source


def views():
    options = {}
    scopes = {
        "middle": ("L14", "L15"), "late": ("L31",), "all": tuple(f"L{layer}" for layer in LAYERS),
        "middle_mean": ("middle_mean",), "middle_agreement": ("middle_agreement",),
        "all_median": ("all_median",),
    }
    measures = (
        ("rank",), ("trend",), ("motion",), ("rank", "velocity_rank"),
        ("trend", "motion"), ("trend", "rank", "turn"),
    )
    for scope, layers in scopes.items():
        for kinds in measures:
            options[f"{scope}/{'_'.join(kinds)}"] = [
                f"r/{kind}/{layer}/{trait}"
                for kind, layer, trait in product(kinds, layers, TRAITS)
            ]
    geometry = [f"g/{kind}/L{layer}/{metric}"
                for kind, layer, metric in product(("rate", "acceleration"), LAYERS, ("magnitude", "orientation"))]
    options["geometry_rates"] = geometry
    options["middle_trend_and_geometry"] = options["middle/trend_motion"] + geometry
    return options


class RankSource(Source):
    def __init__(self, linear_only=False):
        super().__init__()
        self.linear_only = linear_only
        self.features = [rank_observations(points) for points in self.points]
        self.views = views()
        self.values = {key: np.stack([matrix(f, columns, len(self.steps)) for f in self.features])
                       for key, columns in self.views.items()}
        self.policies = [Policy(view, method, *settings) for view, method, settings in
                         product(self.views, METHODS, self.settings)]


    def select(self, develop):
        if not self.linear_only:
            return super().select(develop)
        times, ids = self.evaluate_grid(develop)
        candidates = [i for i, p in enumerate(self.policies) if p.learner in ("ridge", "paired_ridge", "logistic")]
        times = times[candidates]
        onset = np.array([self.outcomes[i]["onset"] or NEVER for i in ids])
        lead = onset - times
        pos = onset != NEVER
        timely = (lead > 0) & (lead <= HORIZON) & pos
        false = (times[:, self.clean[ids]] < NEVER).mean(1)
        worst = np.stack([timely[:, pos & (self.corpus[ids] == c)].mean(1) for c in develop]).min(0)
        width = [len(self.views[self.policies[i].view]) for i in candidates]
        excess = np.maximum(false - MAX_SOURCE_FALSE_RATE, 0.)
        rank = np.lexsort((candidates, width, -np.where(timely, lead, 0).sum(1),
                           false, -worst, -timely.sum(1), excess))
        return self.policies[candidates[rank[0]]]


def rescale_points(points):
    # Vary each layer's activation units and add a distinct affine map for
    # each signed projection after division by activation movement.
    rng = np.random.default_rng(1949)
    gain = rng.uniform(.45, 2.4, len(LAYERS))
    axis_gain = rng.uniform(.6, 1.4, (len(LAYERS), len(TRAITS)))
    axis_offset = rng.uniform(-.03, .03, axis_gain.shape)
    shifted = []
    for point in points:
        original = point["signals"]
        signals = dict(original)
        for j, layer in enumerate(LAYERS):
            magnitude = original[f"L{layer}_magnitude"]
            signals[f"L{layer}_magnitude"] *= gain[j]
            signals[f"L{layer}_orientation"] *= gain[j] ** 2
            for k, trait in enumerate(TRAITS):
                key = f"L{layer}_projection_{trait}"
                signals[key] = gain[j] * (original[key] * axis_gain[j, k] + magnitude * axis_offset[j, k])
        shifted.append(dict(point, signals=signals))
    return shifted


def fit_rank(bank_path, report_path, *, linear_only=False):
    torch.set_num_threads(1)
    source = RankSource(linear_only=linear_only)
    folds = {}
    for excluded in source.corpora:
        refs = tuple(c for c in source.corpora if c != excluded)
        folds[excluded] = source.bank(refs, source.select(refs))
    candidate = source.select(source.corpora)
    bank = dict(version=1, scorer="rank_source", full=source.bank(source.corpora, candidate), by_corpus=folds,
                judge_identity=source.judge_identity, source_inputs=source.identity,
                selection=dict(policy_count=len(source.policies), methods=METHODS, views=source.views))
    if linear_only:
        bank["selection"]["methods"] = ("ridge", "paired_ridge", "logistic")
        bank["selection"]["policy_count"] //= 2
    actual, scaled, snap_checks = [], [], 0
    for name, points, corpus in zip(source.names, source.points, source.corpus, strict=True):
        first = score_prefix(points, folds[corpus])["first_alert"]
        actual.append(first or NEVER)
        scaled.append(score_prefix(rescale_points(points), folds[corpus])["first_alert"] or NEVER)
        for end, point in enumerate(points, 1):
            other = score_prefix(points[:end], folds[corpus])["first_alert"]
            expected = first if first is not None and first <= point["step"] else None
            if other != expected:
                raise ValueError(f"Noncausal alert on {name} at step {point['step']}")
            snap_checks += 1
    write_json(bank_path, bank)
    report = dict(bank_sha256=file_digest(bank_path), policy=asdict(candidate), policy_count=bank["selection"]["policy_count"],
                  checked_prefixes=snap_checks,
                  changed_alarms_after_rescaling=int(np.count_nonzero(np.array(actual) != np.array(scaled))))
    masks = dict(all=np.ones(len(source.names), dtype=bool))
    masks.update({c: source.corpus == c for c in source.corpora})
    report["metrics"] = {name: counts(np.array(actual), source.outcomes, mask) for name, mask in masks.items()}
    write_json(report_path, report)
    print(report["policy"], report["metrics"]["all"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=OUTPUT / "llama-rank-bank.json")
    parser.add_argument("--report", type=Path, default=OUTPUT / "llama-rank-report.json")
    parser.add_argument("--linear-only", action="store_true")
    args = parser.parse_args()
    fit_rank(args.bank, args.report, linear_only=args.linear_only)


if __name__ == "__main__":
    main()
