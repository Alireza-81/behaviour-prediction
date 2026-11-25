"""Select a Llama-only warning bank under synthetic activation nuisance."""

import argparse
from dataclasses import asdict, dataclass
from functools import cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.source_only import observations, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS, evidence, smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import NEVER, counts, fit
from fit_source_only_monitor import CONFIG, HORIZON, MAX_SOURCE_FALSE_RATE, OUTPUT, Policy, Source

METHODS = ("ridge", "paired_ridge", "logistic")
MODES = ("instant", "mean2")
MARGINS = (0., .25, .5)
QUORUMS = (.51, .8)


@dataclass(frozen=True)
class Nuisance:
    """Ranges for independent bounded perturbations of an observed trajectory."""

    seed: int
    log_gain: float = .35
    offset: float = .045
    drift: float = .035


FIT = (Nuisance(109), Nuisance(173), Nuisance(227))
SELECT = (Nuisance(911), Nuisance(997), Nuisance(1091), Nuisance(1229))
AUDIT = (Nuisance(2003), Nuisance(2111), Nuisance(3001, .55, .07, .055))


def perturb(points, profile, identity, last_step):
    """Apply fixed gain/offset and a smooth drift without reading future activations."""
    generator = np.random.default_rng([profile.seed, identity])
    gain = np.exp(generator.uniform(-profile.log_gain, profile.log_gain, len(LAYERS)))
    offset = generator.uniform(-profile.offset, profile.offset, (len(LAYERS), len(TRAITS)))
    drift = generator.uniform(-profile.drift, profile.drift, offset.shape)
    result = []
    for point in points:
        original = point["signals"]
        signals = dict(original)
        if point["step"] > 0:
            phase = point["step"] / last_step
            for j, layer in enumerate(LAYERS):
                magnitude = original[f"L{layer}_magnitude"]
                layer_gain = gain[j] * np.exp(drift[j].mean() * phase)
                signals[f"L{layer}_magnitude"] *= layer_gain
                signals[f"L{layer}_orientation"] *= layer_gain ** 2
                for k, trait in enumerate(TRAITS):
                    key = f"L{layer}_projection_{trait}"
                    signals[key] = layer_gain * (original[key] + magnitude * (offset[j, k] + phase * drift[j, k]))
        result.append(dict(point, signals=signals))
    return result


class StressStudy(Source):
    def __init__(self):
        super().__init__()
        # Cover every layer with the same representations. No architecture-specific gates.
        groups = ("cosine", "angular_projection", "from_first", "reset10", "velocity", "velocity_acceleration")
        self.views = {name: columns for name, columns in self.views.items()
                      if name.split("/")[-1] in groups or name in ("geometry_change", "directions_and_geometry")}
        self.settings = list(product(MODES, MARGINS, QUORUMS))
        self.policies = [Policy(view, method, *settings) for view, method, settings in
                         product(self.views, METHODS, self.settings)]
        fit_features = [self.features] + [self.scenario(p) for p in FIT]
        select_features = [self.features] + [self.scenario(p) for p in SELECT]
        self.fit_values, self.select_values = {}, {}
        for view, columns in self.views.items():
            self.fit_values[view] = np.concatenate([self.array(f, columns) for f in fit_features])
            self.select_values[view] = np.stack([self.array(f, columns) for f in select_features])
        replicas = len(fit_features)
        size = len(self.names)
        self.fit_corpus = np.tile(self.corpus, replicas)
        self.fit_clean = np.tile(self.clean, replicas)
        self.fit_data = SimpleNamespace(labels=np.tile(self.labels, (replicas, 1)),
                                        partner=np.arange(replicas * size) ^ 1)

    def scenario(self, profile):
        # Nuisance is common to a paired experiment, as a measurement shift would be.
        return [observations(perturb(points, profile, i // 2, self.steps[-1]))
                for i, points in enumerate(self.points)]

    def array(self, features, columns):
        return np.stack([matrix(f, columns, len(self.steps)) for f in features])

    @cache
    def stress_fit(self, view, method, corpora):
        ids = np.flatnonzero(np.isin(self.fit_corpus, corpora))
        return fit(method, self.fit_values[view], ids, self.fit_data)

    @cache
    def candidate_scores(self, view, method, corpora):
        model = self.stress_fit(view, method, corpora)
        cal = evidence(model, self.fit_values[view])
        select = evidence(model, self.select_values[view])
        return {mode: (smooth(cal, mode), smooth(select, mode)) for mode in MODES}

    def pool(self, refs, view, method):
        for calibration in refs:
            groups = tuple(c for c in refs if c != calibration)
            model = self.stress_fit(view, method, groups)
            mask = self.fit_clean & (self.fit_corpus == calibration)
            scores = self.candidate_scores(view, method, groups)
            cutoffs, tests = {}, {}
            for mode, (calibration_scores, selection_scores) in scores.items():
                clean = calibration_scores[mask][:, self.steps >= self.min_step]
                peak = float(clean.max())
                cutoffs[mode] = (peak, peak - float(np.median(clean)))
                tests[mode] = selection_scores
            yield calibration, model, tests, cutoffs

    def evaluate_grid(self, develop):
        shape = (len(self.policies), 1 + len(SELECT), len(self.names))
        times = np.full(shape, NEVER, dtype=np.int16)
        allowed = self.steps >= self.min_step
        for group, (view, method) in enumerate(product(self.views, METHODS)):
            start = group * len(self.settings)
            candidate_times = times[start:start + len(self.settings)]
            for corpus in develop:
                target = self.corpus == corpus
                refs = tuple(c for c in develop if c != corpus)
                votes = np.zeros((len(self.settings), shape[1], int(target.sum()), len(self.steps)), dtype=np.uint8)
                for _, _, score, cutoffs in self.pool(refs, view, method):
                    for i, (mode, margin, _) in enumerate(self.settings):
                        peak, spread = cutoffs[mode]
                        votes[i] += score[mode][:, target] > peak + margin * spread
                quorum = np.array([q for _, _, q in self.settings]) * len(refs)
                first = np.where((votes >= quorum[:, None, None, None]) & allowed, self.steps, NEVER).min(-1)
                candidate_times[:, :, target] = first
        ids = np.flatnonzero(np.isin(self.corpus, develop))
        return times[:, :, ids], ids

    def select(self, develop):
        times, ids = self.evaluate_grid(develop)
        onset = np.array([self.outcomes[i]["onset"] or NEVER for i in ids])
        lead = onset - times
        timely = (lead > 0) & (lead <= HORIZON) & (onset != NEVER)
        controls = times[:, :, self.clean[ids]] < NEVER
        false = controls.mean(2).max(1)
        worst = timely.sum(2).min(1)
        excess = np.maximum(false - MAX_SOURCE_FALSE_RATE, 0.)
        width = [len(self.views[p.view]) for p in self.policies]
        simplicity = [METHODS.index(p.learner) for p in self.policies]
        rank = np.lexsort((np.arange(len(false)), width, simplicity, false,
                           -timely.sum((1, 2)), -timely[:, 0].sum(1), -worst, excess))
        return self.policies[rank[0]]


def fit_and_freeze(bank_path, report_path):
    torch.set_num_threads(1)
    source = StressStudy()
    folds = {}
    for heldout in source.corpora:
        refs = tuple(c for c in source.corpora if c != heldout)
        folds[heldout] = source.bank(refs, source.select(refs))
    full_policy = source.select(source.corpora)
    full = source.bank(source.corpora, full_policy)
    stress = dict(fitting=[asdict(p) for p in FIT], selection=[asdict(p) for p in SELECT],
                  audit=[asdict(p) for p in AUDIT])
    bank = dict(version=1, full=full, by_corpus=folds, judge_identity=source.judge_identity,
                source_inputs=source.identity, selection=dict(policy_count=len(source.policies), methods=METHODS,
                    synthetic_nuisance=stress, smoothing=MODES, margins=MARGINS, quorums=QUORUMS,
                    false_alert_budget=MAX_SOURCE_FALSE_RATE,
                    objective="minimize excess worst-case false alerts; then maximize worst-case timely warnings"))
    masks = {"all": np.ones(len(source.names), dtype=bool)}
    masks.update({c: source.corpus == c for c in source.corpora})
    report = dict(scenarios={}, prefix_checks=0)
    for profile in (None, *AUDIT):
        alarms = []
        label = f"nuisance-{profile.seed}" if profile else "clean"
        for i, (points, corpus, name) in enumerate(zip(source.points, source.corpus, source.names, strict=True)):
            if profile:
                points = perturb(points, profile, i // 2, source.steps[-1])
            alarm = score_prefix(points, folds[corpus])["first_alert"]
            alarms.append(alarm or NEVER)
            for end, point in enumerate(points, 1):
                prefix_alarm = score_prefix(points[:end], folds[corpus])["first_alert"]
                assert prefix_alarm == (alarm if alarm is not None and alarm <= point["step"] else None), name
                report["prefix_checks"] += 1
        report["scenarios"][label] = {c: counts(np.array(alarms), source.outcomes, mask) for c, mask in masks.items()}
    write_json(bank_path, bank)
    report.update(model_sha256=file_digest(bank_path), full_policy=asdict(full_policy))
    write_json(report_path, report)
    print(report["full_policy"], {k: v["all"] for k, v in report["scenarios"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default=CONFIG / "source-only-llama-stress.json", type=Path)
    parser.add_argument("--report", default=OUTPUT / "llama-stress-report.json", type=Path)
    args = parser.parse_args()
    fit_and_freeze(args.bank, args.report)


if __name__ == "__main__":
    main()
