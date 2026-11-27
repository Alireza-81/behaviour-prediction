"""Stress fit a one-source checkpoint forecaster against changes in learning pace."""
import argparse
from dataclasses import asdict
from functools import cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.source_only import observations, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import evidence, smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import NEVER, counts, fit
from fit_source_only_monitor import HORIZON, MAX_SOURCE_FALSE_RATE, Policy, Source


OUTPUT = Path("outputs/single-source-qualification")
TILTS_TRAIN = (0., -.6, .6)
TILTS_TEST = (0., -.4, .4)
METHODS = ("ridge", "paired_ridge", "logistic", "neighbors")


def pace(steps, tilt, end):
    """Rescale elapsed time monotonically, preserving both schedule endpoints."""
    return steps + tilt * steps * (1 - steps / end)


class WarpedSource(Source):
    def __init__(self, **source_inputs):
        super().__init__(**source_inputs)
        self.views = {k: v for k, v in self.views.items() if "/cosine" not in k and "angular_projection" not in k}
        self.methods = METHODS
        self.policies = [Policy(k, method, *settings) for k, method, settings in
                         product(self.views, self.methods, self.settings)]
        all_tilts = sorted(set(TILTS_TRAIN + TILTS_TEST))
        self.mat = {tilt: self.warped_features(tilt) for tilt in all_tilts}
        self.outcomes_at = {t: self.warped_outcomes(t) for t in all_tilts}
        self.labels_at = {t: self.forecast_labels(t) for t in all_tilts}

    def warped_features(self, tilt):
        if tilt == 0:
            return self.values
        steps = np.r_[0, self.steps]
        locations = pace(steps, tilt, 40)
        features = []
        for series in self.points:
            signals = {k: np.interp(steps, locations, [p["signals"][k] for p in series])
                       for k in series[0]["signals"]}
            path = [dict(step=int(step), signals={k: x[j] for k, x in signals.items()})
                    for j, step in enumerate(steps)]
            features.append(observations(path))
        return {k: np.stack([matrix(f, v, len(self.steps)) for f in features]) for k, v in self.views.items()}

    def warped_outcomes(self, tilt):
        if tilt == 0:
            return self.outcomes
        result = []
        for o in self.outcomes:
            moved = {}
            for key in ("onset", "earliest_possible"):
                s = o[key]
                # Beyond 40 the original endpoint is retained. This interval is
                # used to decide whether a late forecast has sufficient follow-up.
                moved[key] = int(np.ceil(float(pace(s, tilt, 40) if s is not None and s <= 40 else s or 0) / 5)) * 5 or None
            result.append(dict(o, **moved))
        return result

    def forecast_labels(self, tilt):
        if tilt == 0:
            return self.labels
        result = np.full_like(self.labels, -1)
        for i, o in enumerate(self.outcomes_at[tilt]):
            if o["onset"] is not None:
                d = o["onset"] - self.steps
                result[i, d > 0] = (d[d > 0] <= HORIZON).astype(int)
            elif self.clean[i]:
                result[i, :] = 0
        return result

    @cache
    def fitted(self, view, method, groups):
        eligible = np.flatnonzero(np.isin(self.corpus, groups))
        n = len(eligible)
        if n % 2 or not np.array_equal(np.flatnonzero(np.isin(self.partner, eligible)), eligible):
            raise ValueError("Time-warp fitting requires complete matched pairs")
        arrays = np.concatenate([self.mat[tilt][view][eligible] for tilt in TILTS_TRAIN])
        labels = np.concatenate([self.labels_at[tilt][eligible] for tilt in TILTS_TRAIN])
        indices = np.arange(len(arrays))
        return fit(method, arrays, indices, SimpleNamespace(labels=labels, partner=indices ^ 1))

    @cache
    def fitted_scores(self, view, method, groups):
        model = self.fitted(view, method, groups)
        from fit_source_only_monitor import SMOOTHING
        return {tilt: {mode: smooth(x, mode) for mode in SMOOTHING}
                for tilt, x in ((t, evidence(model, self.mat[t][view])) for t in self.mat)}

    def calibration_tilts(self, method):
        return TILTS_TRAIN

    def pool(self, refs, view, method):
        for c in refs:
            groups = tuple(k for k in refs if k != c)
            member = self.fitted(view, method, groups)
            all_scores = self.fitted_scores(view, method, groups)
            mask = (self.corpus == c) & self.clean
            thresholds = {}
            for mode in all_scores[0.]:
                control = np.concatenate([all_scores[t][mode][mask][:, self.steps >= self.min_step]
                                          for t in self.calibration_tilts(method)])
                peak = control.max()
                thresholds[mode] = peak, peak - np.median(control)
            yield c, member, all_scores, thresholds

    def evaluate_grid(self, corpora):
        times = np.full((len(self.policies), len(TILTS_TEST), len(self.names)), NEVER)
        ids = np.flatnonzero(np.isin(self.corpus, corpora))
        offset = 0
        for view, method in product(self.views, self.methods):
            for c in corpora:
                held = self.corpus == c
                refs = tuple(k for k in corpora if k != c)
                votes = np.zeros((len(self.settings), len(TILTS_TEST), int(held.sum()), len(self.steps)), dtype=np.uint8)
                for _, _, scores, cuts in self.pool(refs, view, method):
                    for j, (mode, margin, _) in enumerate(self.settings):
                        cutoff, spread = cuts[mode]
                        v = np.stack([scores[t][mode][held] for t in TILTS_TEST])
                        votes[j] += v > cutoff + margin * spread
                quorum = np.array([q for _, _, q in self.settings])
                slots = (votes >= len(refs) * quorum[:, None, None, None]) & (self.steps >= self.min_step)
                times[offset:offset+len(self.settings), :, held] = np.where(slots, self.steps, NEVER).min(-1)
            offset += len(self.settings)
        return times[..., ids], ids

    def select(self, corpora):
        times, ids = self.evaluate_grid(corpora)
        onset = np.asarray([[self.outcomes_at[t][i]["onset"] or NEVER for i in ids] for t in TILTS_TEST])
        lead = onset - times
        timely = (lead > 0) & (lead <= HORIZON) & (onset != NEVER)
        true_rate = timely.mean(-1)
        false = (times[..., self.clean[ids]] < NEVER).mean(-1).max(1)
        penalty = np.maximum(0, false - MAX_SOURCE_FALSE_RATE)
        complexity = np.array([len(self.views[p.view]) for p in self.policies])
        order = np.lexsort((complexity, -np.where(timely, lead, 0).sum((1, 2)), false,
                            -true_rate[:, 0], -true_rate.min(1), penalty))
        return self.policies[order[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    study = WarpedSource()
    full_choice = study.select(study.corpora)
    folds = {}
    for corpus in study.corpora:
        donors = tuple(c for c in study.corpora if c != corpus)
        folds[corpus] = study.bank(donors, study.select(donors))
    bank = dict(version=1, full=study.bank(study.corpora, full_choice), by_corpus=folds,
                source_inputs=study.identity, judge_identity=study.judge_identity,
                selection=dict(source="llama", training_tilts=TILTS_TRAIN, test_tilts=TILTS_TEST,
                               maximum_false_alert_rate=MAX_SOURCE_FALSE_RATE, candidates=len(study.policies)))
    path = args.output / "time-warp-bank.json"
    write_json(path, bank)
    checks = 0
    alarms = []
    for pts, corpus in zip(study.points, study.corpus, strict=True):
        policy = folds[corpus]
        first = score_prefix(pts, policy)["first_alert"]
        for j, point in enumerate(pts, 1):
            expected = first if first is not None and first <= point["step"] else None
            assert score_prefix(pts[:j], policy)["first_alert"] == expected
            checks += 1
        alarms.append(first or NEVER)
    results = counts(np.array(alarms), study.outcomes, np.ones(len(alarms), dtype=bool))
    report = dict(bank_sha256=file_digest(path), full_choice=asdict(full_choice), nested_clean=results,
                  verified_prefixes=checks)
    write_json(args.output / "time-warp-llama.json", report)
    print(report)


if __name__ == "__main__":
    main()
