"""Freeze a phase-adaptive warning bank using Llama only."""

import argparse
from itertools import product
from pathlib import Path
from functools import cache
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.phase_source import observations, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS, evidence, smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import NEVER, counts, fit
from fit_source_only_monitor import HORIZON, OUTPUT, MAX_SOURCE_FALSE_RATE, Policy, Source

METHODS = ("ridge", "paired_ridge", "logistic")
MODES = ("instant", "mean2", "persistent")
PACES = (1.0, .8, 1.2)
STRESS = (.7, 1.35)


def pace_variant(points, phase, interval):
    """Resample an archived source trajectory. This transformation is offline only."""
    steps = np.array([p["step"] for p in points])
    clock = np.maximum(steps - interval, 0) * phase + interval
    result = points[:2]
    for step, requested in zip(steps[2:], clock[2:], strict=True):
        upper = min(int(np.searchsorted(steps, requested, side="left")), len(steps) - 1)
        lower = max(1, upper - 1)
        a = points[lower]["signals"]
        b = points[upper]["signals"]
        fraction = np.clip((requested - steps[lower]) / (steps[upper] - steps[lower]), 0, 1)
        signals = {k: (1 - fraction) * a[k] + fraction * b[k] for k in a}
        result.append(dict(step=int(step), signals=signals))
    return result


def shifted_onset(onset, phase, interval):
    if onset is None:
        return None
    return int(interval + interval * np.ceil((onset - interval) / (phase * interval)))


class PhaseStudy(Source):
    def __init__(self):
        super().__init__()
        self.views = {}
        for layers, kinds in product(((14, 15), (31,), LAYERS),
                 (("half_arc",), ("recent_arc",), ("dominant",), ("phase_turn",), ("local_direction",),
                  ("dominant", "local_direction"))):
            columns = [f"{kind}/L{layer}/{trait}" for kind, layer, trait in product(kinds, layers, TRAITS)]
            view = f"{'-'.join(map(str, layers))}/{'_'.join(kinds)}"
            self.views[view] = columns
            if kinds in (("dominant",), ("half_arc",)):
                self.views[view + "/motion"] = columns + ["phase/motion", "phase/growth"]
        self.settings = list(product(MODES, (0., .25, .5), (.51, .8)))
        self.policies = [Policy(view, method, *settings) for view, method, settings in
                         product(self.views, METHODS, self.settings)]
        points_features, labels, outcomes = [], [], []
        for phase in (*PACES, *STRESS):
            points, y, truth = self.scenario(phase)
            points_features.append([observations(p) for p in points])
            labels.append(y)
            outcomes.append(truth)
        self.values = {view: np.array([[matrix(f, columns, len(self.steps)) for f in features]
                                      for features in points_features]) for view, columns in self.views.items()}
        n = len(self.names)
        self.fit_data = SimpleNamespace(labels=np.concatenate(labels[:len(PACES)]),
                                        partner=(np.arange(n * len(PACES)) ^ 1))
        self.fit_corpus = np.tile(self.corpus, len(PACES))
        self.outcome_scenarios = outcomes

    def scenario(self, phase):
        if phase == 1.0:
            return self.points, self.labels, self.outcomes
        interval = int(self.steps[0])
        points = [pace_variant(p, phase, interval) for p in self.points]
        original_clock = interval + (self.steps - interval) * phase
        outcomes = [dict(onset=shifted_onset(o["onset"], phase, interval),
                         earliest_possible=shifted_onset(o["earliest_possible"], phase, interval)) for o in self.outcomes]
        y = np.full_like(self.labels, -1)
        for i, o in enumerate(outcomes):
            earliest = o["earliest_possible"] or NEVER
            end = o["onset"] or NEVER
            scored = (self.steps < earliest) & (original_clock <= self.steps[-1])
            y[i, scored] = (end <= self.steps[scored] + HORIZON).astype(y.dtype)
        return points, y, outcomes

    @cache
    def fitted(self, view, method, groups):
        rows = self.values[view][:len(PACES)]
        train = np.flatnonzero(np.isin(self.fit_corpus, groups))
        return fit(method, np.concatenate(rows), train, self.fit_data)

    @cache
    def fitted_scores(self, view, method, groups):
        scores = evidence(self.fitted(view, method, groups), self.values[view][:len(PACES)])
        return {mode: smooth(scores, mode) for mode in MODES}

    def pool(self, refs, view, method):
        for calibrate in refs:
            train = tuple(c for c in refs if c != calibrate)
            model = self.fitted(view, method, train)
            values = self.fitted_scores(view, method, train)
            clean = (self.corpus == calibrate) & self.clean
            cutoffs = {}
            for mode, scores in values.items():
                samples = scores[:, clean][:, :, self.steps >= self.min_step]
                maximum = samples.max()
                cutoffs[mode] = (maximum, maximum - np.median(samples))
            yield calibrate, model, values, cutoffs

    def evaluate_grid(self, develop):
        times = np.full((len(self.policies), len(PACES), len(self.names)), NEVER, dtype=np.int16)
        for group, (view, method) in enumerate(product(self.views, METHODS)):
            first = group * len(self.settings)
            for corpus in develop:
                mask = self.corpus == corpus
                refs = tuple(c for c in develop if c != corpus)
                votes = np.zeros((len(self.settings), len(PACES), mask.sum(), len(self.steps)), dtype=np.uint8)
                for _, _, values, cutoffs in self.pool(refs, view, method):
                    for i, (mode, margin, _) in enumerate(self.settings):
                        peak, spread = cutoffs[mode]
                        votes[i] += values[mode][:, mask] > peak + margin * spread
                quorum = np.array([q for _, _, q in self.settings]) * len(refs)
                fires = (votes >= quorum[:, None, None, None]) & (self.steps >= self.min_step)
                times[first:first + len(self.settings), :, mask] = np.where(fires, self.steps, NEVER).min(-1)
        ids = np.flatnonzero(np.isin(self.corpus, develop))
        return times[:, :, ids], ids

    def select(self, develop):
        times, ids = self.evaluate_grid(develop)
        onset = np.array([[self.outcome_scenarios[j][i]["onset"] or NEVER for i in ids]
                          for j in range(len(PACES))])
        lead = onset[None, :, :] - times
        timely = ((lead > 0) & (lead <= HORIZON) & (onset != NEVER)).sum(2)
        false = (times[:, :, self.clean[ids]] < NEVER).mean(2).max(1)
        excess = np.maximum(false - MAX_SOURCE_FALSE_RATE, 0.)
        dimensions = np.array([len(self.views[p.view]) for p in self.policies])
        rank = np.lexsort((np.arange(len(false)), dimensions, false, -timely.sum(1), -timely[:, 0],
                           -timely.min(1), excess))
        return self.policies[rank[0]]

    def bank(self, refs, policy):
        return dict(super().bank(refs, policy), representation="phase")


def freeze(bank_path, report_path):
    torch.set_num_threads(1)
    study = PhaseStudy()
    folds = {}
    for exclude in study.corpora:
        refs = tuple(c for c in study.corpora if c != exclude)
        folds[exclude] = study.bank(refs, study.select(refs))
    full = study.bank(study.corpora, study.select(study.corpora))
    bank = dict(version=1, representation="phase", full=full, by_corpus=folds,
                source_inputs=study.identity, judge_identity=study.judge_identity,
                selection=dict(fit_and_select_paces=PACES, unseen_source_stress_paces=STRESS,
                               methods=METHODS, policy_count=len(study.policies), views=study.views))
    scenarios, checks = {}, 0
    for pace in (1.0, *STRESS):
        points, _, outcomes = study.scenario(pace)
        predictions = []
        for p, corpus in zip(points, study.corpus, strict=True):
            chosen = folds[corpus]
            alarm = score_prefix(p, chosen)["first_alert"]
            for i, row in enumerate(p, 1):
                expected = alarm if alarm is not None and alarm <= row["step"] else None
                assert score_prefix(p[:i], chosen)["first_alert"] == expected
                checks += 1
            predictions.append(alarm if alarm is not None else NEVER)
        scenarios[str(pace)] = counts(np.array(predictions), outcomes, np.ones(len(points), dtype=bool))
    write_json(bank_path, bank)
    report = dict(by_source_pace=scenarios, prefix_checks=checks, model_sha256=file_digest(bank_path),
                  full_policy={k: full[k] for k in ("view", "learner", "smoothing", "quorum", "margin")})
    write_json(report_path, report)
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=OUTPUT / "source-only-llama-phase.json")
    parser.add_argument("--report", type=Path, default=OUTPUT / "llama-phase-report.json")
    args = parser.parse_args()
    freeze(args.bank, args.report)


if __name__ == "__main__":
    main()
