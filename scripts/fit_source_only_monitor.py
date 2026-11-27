"""Select and freeze activation-only monitors with nested Llama corpus holdouts."""
import argparse
from dataclasses import asdict, dataclass
from functools import cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.source_only import observations, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS, evidence, smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_single_source_trajectories import NEVER, counts, fit, serial
from evaluate_monitor_study import CONFIG, OUT, dataset, read_split
from transfer_advice_early_warning import target_pairs

OUTPUT = Path("outputs/source-only-monitor-v1")
HORIZON = 10
MAX_SOURCE_FALSE_RATE = 1 / 18
METHODS = ("ridge", "paired_ridge", "logistic", "neighbors", "kernel", "reservoir")
SMOOTHING = ("instant", "mean2", "persistent")


@dataclass(frozen=True)
class Policy:
    view: str
    learner: str
    smoothing: str
    margin: float
    quorum: float


def views():
    """Symmetric feature families chosen independently of target architecture."""
    options = {}
    for layers in ((14, 15), (31,), LAYERS):
        name = "-".join(str(layer) for layer in layers)
        for groups in (("cosine",), ("angular_projection",), ("from_first",), ("reset10",),
                       ("velocity",), ("velocity", "acceleration"), ("from_first", "ratio_velocity")):
            options[f"{name}/{'_'.join(groups)}"] = [
                f"d/{kind}/L{layer}/{trait}" for kind, layer, trait in product(groups, layers, TRAITS)
            ]
    rates = [f"g/{kind}/L{layer}/{metric}" for kind, layer, metric in
             product(("rate", "acceleration", "first_ratio", "balance_first"), LAYERS, ("magnitude", "orientation"))]
    angular = [f"g/{kind}/L{layer}/magnitude" for kind, layer in
               product(("angle_first", "angle_rate"), LAYERS)]
    options["geometry_change"] = rates + angular
    options["directions_and_geometry"] = options["14-15-31/from_first_ratio_velocity"] + rates
    return options


class Source:
    def __init__(self, changes_only=False, *, architecture="llama", archive_path=None, report_path=None):
        path = CONFIG / "protocol.json"
        protocol = dict(read_json(path), forecast_steps=HORIZON)
        self.architecture = architecture
        self.identity = dict(protocol=file_digest(path))
        if archive_path is None and report_path is None:
            if architecture != "llama":
                raise ValueError("An alternate fitting architecture requires one archive and its report")
            identities = set()
            pairs = read_split(protocol, "development", identities) | read_split(protocol, "heldout", identities)
            self.identity["summaries"] = {
                f"{corpus}/{seed}": file_digest(OUT / "summaries" / protocol["corpora"][corpus] / f"{corpus}-{seed}.json")
                for corpus, seed in sorted(pairs)
            }
            self.judge_identity = next(iter(identities))
        else:
            if archive_path is None or report_path is None or architecture == "llama":
                raise ValueError("Pass the matching archive and report for one alternate source")
            archive, report = read_json(archive_path), read_json(report_path)
            if (archive["reference_report_sha256"] != file_digest(report_path)
                    or archive["panel_sha256"] != file_digest(OUT / "panel.json")
                    or set(archive["runs"]) != set(report["runs"])
                    or report["steps"] != protocol["capture_steps"]
                    or set(report["families"]) != set(protocol["corpora"])):
                raise ValueError("Alternate source trajectory identity or protocol differs")
            pairs = target_pairs(archive, report)
            self.identity.update(archive=file_digest(archive_path), report=file_digest(report_path))
            self.judge_identity = tuple(archive["judge_identity"])
        self.corpora = tuple(sorted(protocol["corpora"]))
        self.steps = np.array([t for t in protocol["capture_steps"] if 0 < t <= protocol["capture_steps"][-1] - HORIZON])
        self.names, self.points, self.corpus, self.outcomes = [], [], [], []
        self.full_points = []
        self.labels = []
        rows, outcomes = dataset(pairs, protocol, sensitivity=False, activation_only=True)
        labels = {(r["run"], r["step"]): r["label"] for r in rows}
        for (corpus, seed), pair in sorted(pairs.items()):
            for arm in (0, 1):
                name = f"{corpus}/{seed}/{arm}"
                self.names.append(name)
                self.corpus.append(corpus)
                self.outcomes.append(outcomes[name])
                trajectory = [p for p in pair["records"] if p["branch"] == arm]
                self.full_points.append(trajectory)
                self.points.append([p for p in trajectory if p["step"] <= self.steps[-1]])
                self.labels.append([labels.get((name, step), -1) for step in self.steps])
        self.labels = np.array(self.labels)
        self.partner = np.arange(len(self.labels)) ^ 1
        self.corpus = np.array(self.corpus)
        self.clean = np.array([o["earliest_possible"] is None for o in self.outcomes])
        self.features = [observations(points) for points in self.points]
        self.min_step = int(self.steps[1])
        self.views = views()
        if changes_only:
            self.views = {k: v for k, v in self.views.items()
                          if not any(column.split("/")[1] in ("cosine", "angular_projection") for column in v)}
        self.values = {key: np.stack([matrix(f, columns, len(self.steps)) for f in self.features])
                       for key, columns in self.views.items()}
        self.settings = [(smoothing, margin, quorum) for smoothing, margin, quorum in
                         product(SMOOTHING, (0., .25, .5, 1.), (.51, .8))]
        self.methods = METHODS
        self.policies = [Policy(view, method, *settings) for view, method, settings in
                         product(self.views, self.methods, self.settings)]

    @cache
    def fitted(self, view, method, groups):
        ids = np.flatnonzero(np.isin(self.corpus, groups))
        if len(np.unique(self.labels[ids][self.labels[ids] >= 0])) != 2:
            raise ValueError("Each source training split must contain both forecast labels")
        return fit(method, self.values[view], ids, self)

    @cache
    def fitted_scores(self, view, method, groups):
        model = self.fitted(view, method, groups)
        scores = evidence(model, self.values[view])
        return {mode: smooth(scores, mode) for mode in SMOOTHING}

    def pool(self, refs, view, method):
        for c in refs:
            train = tuple(o for o in refs if o != c)
            model = self.fitted(view, method, train)
            values = self.fitted_scores(view, method, train)
            mask = (self.corpus == c) & self.clean
            cutoffs = {}
            for mode, scores in values.items():
                clean = scores[mask][:, self.steps >= self.min_step]
                peak = clean.max()
                cutoffs[mode] = (peak, peak - np.median(clean))
            yield c, model, values, cutoffs

    def evaluate_grid(self, develop):
        """For each scoring fold exclude its corpus from fitting and calibration."""
        times = np.full((len(self.policies), len(self.names)), NEVER)
        expected_ids = np.flatnonzero(np.isin(self.corpus, develop))
        offset = 0
        for view, method in product(self.views, self.methods):
            for corpus in develop:
                target = self.corpus == corpus
                refs = tuple(c for c in develop if c != corpus)
                votes = np.zeros((len(self.settings), int(target.sum()), len(self.steps)), dtype=np.uint8)
                for _, _, values, thresholds in self.pool(refs, view, method):
                    for i, (mode, margin, _) in enumerate(self.settings):
                        peak, spread = thresholds[mode]
                        votes[i] += values[mode][target] > peak + margin * spread
                quorum = np.array([q for _, _, q in self.settings])
                first = np.where((votes >= quorum[:, None, None] * len(refs)) &
                                 (self.steps >= self.min_step), self.steps, NEVER).min(-1)
                times[offset:offset + len(self.settings), target] = first
            offset += len(self.settings)
        return times[:, expected_ids], expected_ids

    def select(self, develop):
        times, ids = self.evaluate_grid(develop)
        onset = np.array([self.outcomes[i]["onset"] or NEVER for i in ids])
        lead = onset - times
        pos = onset != NEVER
        timely = (lead > 0) & (lead <= HORIZON) & pos
        false = (times[:, self.clean[ids]] < NEVER).mean(1)
        # Select the earliest reliable alert among candidates with equal timely count.
        worst = np.stack([timely[:, pos & (self.corpus[ids] == c)].mean(1) for c in develop]).min(0)
        simplicity = np.array([self.methods.index(p.learner) for p in self.policies])
        dimensions = np.array([len(self.views[p.view]) for p in self.policies])
        excess = np.maximum(false - MAX_SOURCE_FALSE_RATE, 0.)
        ranked = np.lexsort((np.arange(len(false)), dimensions, simplicity, -np.where(timely, lead, 0).sum(1),
                             false, -worst, -timely.sum(1), excess))
        return self.policies[ranked[0]]

    def bank(self, refs, policy):
        members = []
        for c, model, _, cutoffs in self.pool(refs, policy.view, policy.learner):
            peak, spread = cutoffs[policy.smoothing]
            members.append(dict(serial(model), calibration_corpus=c, cutoff=float(peak + policy.margin * spread)))
        return dict(source_architecture=self.architecture, source_corpora=list(refs), horizon=HORIZON, min_step=self.min_step,
                    columns=self.views[policy.view], members=members, **asdict(policy))


def fit_and_freeze(bank_path, output_report, *, changes_only=False, **source_inputs):
    torch.set_num_threads(1)
    study = Source(changes_only=changes_only, **source_inputs)
    choices = {}
    for excluded in study.corpora:
        refs = tuple(c for c in study.corpora if c != excluded)
        policy = study.select(refs)
        choices[excluded] = study.bank(refs, policy)
        print(f"Selected for {excluded} holdout: {policy}", flush=True)
    full_policy = study.select(study.corpora)
    full = study.bank(study.corpora, full_policy)
    bank = dict(version=1, full=full, by_corpus=choices, judge_identity=study.judge_identity,
                source_inputs=study.identity, selection=dict(methods=METHODS, views=study.views,
                    policy_count=len(study.policies), false_alert_budget=MAX_SOURCE_FALSE_RATE,
                    smoothing=SMOOTHING, margins=(0., .25, .5, 1.), quorums=(.51, .8), horizon=HORIZON))
    times, audit, shifted_mismatch = [], 0, []
    for name, points, corpus in zip(study.names, study.points, study.corpus, strict=True):
        policy = choices[corpus]
        first = score_prefix(points, policy)["first_alert"]
        times.append(first or NEVER)
        for i, point in enumerate(points, 1):
            got = score_prefix(points[:i], policy)["first_alert"]
            assert got == (first if first is not None and first <= point["step"] else None), name
            audit += 1
        scaled = [dict(p, signals={k: v * (2 if k.endswith("orientation") else np.sqrt(2))
                                   for k, v in p["signals"].items()}) for p in points]
        if score_prefix(scaled, policy)["first_alert"] != first:
            shifted_mismatch.append(name)
    write_json(bank_path, bank)
    any_run = np.ones(len(times), dtype=bool)
    summary = {c: counts(np.array(times), study.outcomes, study.corpus == c) for c in study.corpora}
    report = dict(model_sha256=file_digest(bank_path), full_policy=asdict(full_policy),
                  metrics=counts(np.array(times), study.outcomes, any_run), by_corpus=summary,
                  time_only={str(t): counts(np.full(len(times), t), study.outcomes, any_run) for t in (10, 15)},
                  no_alert=counts(np.full(len(times), NEVER), study.outcomes, any_run),
                  verified_prefixes=audit, rescale_disagreements=shifted_mismatch,
                  first_alerts={n: a if a < NEVER else None for n, a in zip(study.names, times, strict=True)})
    write_json(output_report, report)
    print(report["full_policy"], report["metrics"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=CONFIG / "source-only-llama.json")
    parser.add_argument("--report", type=Path, default=OUTPUT / "llama-report.json")
    parser.add_argument("--changes-only", action="store_true",
                        help="Exclude projection levels and fit using only changes within each run")
    args = parser.parse_args()
    fit_and_freeze(args.bank, args.report, changes_only=args.changes_only)


if __name__ == "__main__":
    main()
