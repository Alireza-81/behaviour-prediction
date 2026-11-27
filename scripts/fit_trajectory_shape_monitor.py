"""Fit visual-shape and reference-based monitors with Llama-only model selection."""
import argparse
from dataclasses import asdict
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS, evidence, smooth
from behaviour_prediction.monitoring.trajectory_shape import (
    CONTEXT, comparison, control_residual, features, score_prefix,
)
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import NEVER, counts, fit
from fit_source_only_monitor import HORIZON, SMOOTHING, Policy, Source

OUTPUT = Path("outputs/trajectory-shape-monitor-v1")
MODES = ("single", "distilled", "difference", "distance")
REFERENCE_COUNTS = 1


def names(items, layers, axes, prefix="shape"):
    return [f"{prefix}/{item}/L{layer}/{axis}" for item, layer, axis in product(items, layers, axes)]


def views(external_reference):
    shape = names(("excursion", "trend", "bend"), LAYERS, TRAITS)
    geometry = names(("geom_speed", "geom_bend", "geom_excursion"), LAYERS, ("magnitude", "orientation"))
    output = {
        "direction_shape": shape,
        "geometry_shape": geometry,
        "joint_shape": shape + geometry,
        "direction_trend": names(("trend",), LAYERS, TRAITS),
        "direction_bend": names(("bend",), LAYERS, TRAITS),
        "mid_shape": names(("excursion", "trend", "bend"), (14, 15), TRAITS),
        "anchored_directions": names(("reset10",), LAYERS, TRAITS, "d"),
    }
    if external_reference:
        directions = names(("cosine",), LAYERS, TRAITS, "d")
        levels = names(("log", "angle"), LAYERS, ("magnitude",), "g")
        levels += names(("log",), LAYERS, ("orientation",), "g")
        output.update(direction_gap=directions, geometry_gap=levels, joint_gap=directions + levels,
                      gap_and_shape=directions + levels + names(("bend",), LAYERS, TRAITS))
    return output


def independent_references(names_, corpora, arms, seeds):
    """Rotate within-corpus correct-answer controls, for both arms of each seed."""
    references = []
    for name, corpus, seed in zip(names_, corpora, seeds, strict=True):
        options = np.flatnonzero((corpora == corpus) & (arms == 1))
        order = sorted(options, key=lambda i: (seeds[i], names_[i]))
        if len(order) < 2 or len({seeds[i] for i in order}) != len(order):
            raise ValueError(f"Need distinct control seeds in {corpus}")
        source_seed = [seeds[i] for i in order].index(seed)
        reference = order[(source_seed + 1) % len(order)]
        if seeds[reference] == seed:
            raise AssertionError(f"Self-reference for {name}")
        references.append(reference)
    return np.array(references, dtype=int)


def fit_control(values, context):
    center, scale = context.mean(0), context.std(0).clip(1e-3)
    z = np.clip((context - center) / scale, -3., 3.)
    intercept = values.mean(0)
    gram = z.T @ z / len(z) + np.eye(z.shape[-1])
    coef = np.linalg.solve(gram, z.T @ (values - intercept) / len(z))
    return dict(center=center.tolist(), scale=scale.tolist(), intercept=intercept.tolist(), coef=coef.tolist())


def fit_matched_ridge(values, terminal, train, data, weight):
    """Fit one-run forecasts with an auxiliary matched-pair ordering objective."""
    labels = data.labels[train]
    observed = labels >= 0
    raw, y = values[train][observed], labels[observed]
    center, scale = raw.mean(0), raw.std(0).clip(1e-5)
    x = (raw - center) / scale
    gram = x.T @ x / len(x) + .5 * np.eye(x.shape[-1])
    rhs = x.T @ (y - y.mean()) / len(x)
    originals = train[data.arms[train] == 0]
    partners = data.partner[originals]
    positive = (data.labels[originals] == 1) & (data.labels[partners] == 0)
    future_gap = (values[originals] - values[partners])[positive]
    terminal_gap = terminal[originals] - terminal[partners]
    # Each event gives one matched vote in the terminal window. The two terms
    # distinguish the onset lead-in from the stronger separation later in training.
    for gap in (future_gap, terminal_gap):
        z = gap / scale
        z /= max(1., np.sqrt(np.square(z).mean()))
        gram += (weight / 2) * (z.T @ z / len(z))
        rhs += (weight / 2) * z.mean(0)
    return dict(method="ridge", center=center, scale=scale, coef=np.linalg.solve(gram, rhs))


class ShapeStudy(Source):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.comparison = "single" if mode == "distilled" else mode
        self.views = views(self.comparison != "single")
        if mode == "distilled":
            normalized = names(("cosine",), LAYERS, TRAITS, "d")
            geometry = names(("angle_balance",), LAYERS, ("magnitude",), "g")
            geometry += names(("balance", "balance_rate"), LAYERS, ("magnitude", "orientation"), "g")
            self.views.update(direction_level=normalized, geometry_balance=geometry,
                              direction_and_balance=normalized + geometry)
        axes = [features(p) for p in self.points]
        self.context = np.stack([matrix(p, CONTEXT, len(self.steps)) for p in axes])
        fields = [name.split("/") for name in self.names]
        self.arms = np.array([int(v[2]) for v in fields])
        self.seeds = np.array([int(v[1]) for v in fields])
        self.references = (independent_references(self.names, self.corpus, self.arms, self.seeds)
                           if self.comparison != "single" else None)
        self.values = {}
        for k, columns in self.views.items():
            values = np.stack([matrix(p, columns, len(self.steps)) for p in axes])
            self.values[k] = (comparison(values, values[self.references], self.comparison)
                              if self.references is not None else values)
        self.methods = ("ridge", "paired_ridge", "logistic", "neighbors")
        if mode == "single":
            self.methods += ("control_ridge", "control_logistic")
        if mode == "distilled":
            self.min_step = int(self.steps[0])
            self.methods = ("matched/.25", "matched/1", "matched/4")
            terminal = [features(p) for p in self.full_points]
            self.terminal = {k: np.stack([matrix(p, cols, len(self.full_points[i]) - 1)[-3:].mean(0)
                                          for i, p in enumerate(terminal)]) for k, cols in self.views.items()}
        self.policies = [Policy(k, m, *s) for k, m, s in product(self.views, self.methods, self.settings)]

    @cache
    def fitted(self, view, method, groups):
        train = np.flatnonzero(np.isin(self.corpus, groups))
        values = self.values[view]
        if method.startswith("matched/"):
            return fit_matched_ridge(values, self.terminal[view], train, self, float(method.split("/")[1]))
        reference = None
        if method.startswith("control_"):
            clean = train[self.clean[train] & (self.arms[train] == 1)]
            reference = fit_control(values[clean][:, 1:].reshape(-1, values.shape[-1]),
                                    self.context[clean][:, 1:].reshape(-1, len(CONTEXT)))
            method = method.removeprefix("control_")
        model = fit(method, control_residual(values, self.context, reference), train, self)
        return dict(model, control_model=reference)

    @cache
    def fitted_scores(self, view, method, groups):
        member = self.fitted(view, method, groups)
        raw = control_residual(self.values[view], self.context, member.get("control_model"))
        scores = evidence(member, raw)
        return {k: smooth(scores, k) for k in SMOOTHING}

    def bank(self, refs, policy):
        return dict(super().bank(refs, policy), comparison=self.comparison)


def freeze(mode, output):
    torch.set_num_threads(1)
    source = ShapeStudy(mode)
    full_choice = source.select(source.corpora)
    folds = {}
    for excluded in source.corpora:
        refs = tuple(x for x in source.corpora if x != excluded)
        folds[excluded] = source.bank(refs, source.select(refs))
    bank = dict(full=source.bank(source.corpora, full_choice), by_corpus=folds, scorer="trajectory_shape",
                source_inputs=source.identity, judge_identity=source.judge_identity,
                selection=dict(input_type=mode, views=source.views, source=source.architecture,
                               candidates=len(source.policies), comparison_reference_count=REFERENCE_COUNTS
                               if source.references is not None else 0, horizon=HORIZON))
    path = output / f"{mode}-bank.json"
    write_json(path, bank)
    verified, alerts = 0, []
    for i, (run, points, corpus) in enumerate(zip(source.names, source.points, source.corpus, strict=True)):
        member = folds[corpus]
        other = source.points[source.references[i]] if source.references is not None else None
        full = score_prefix(points, member, other)
        alerts.append(full["first_alert"] or NEVER)
        for j, point in enumerate(points, 1):
            partial = score_prefix(points[:j], member, other[:j] if other is not None else None)
            expected = full["first_alert"]
            if expected is not None and expected > point["step"]:
                expected = None
            if partial["first_alert"] != expected or partial["votes"] != full["votes"][:j-1]:
                raise AssertionError(f"Prefix scoring changed for {run}")
            verified += 1
    report = dict(input_type=mode, full_policy=asdict(full_choice), bank_sha256=file_digest(path),
                  prefix_checks=verified, metrics=counts(np.array(alerts), source.outcomes, np.ones(len(alerts), bool)))
    write_json(output / f"{mode}-source-report.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    freeze(args.mode, args.output)


if __name__ == "__main__":
    main()
