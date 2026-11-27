"""Train causal neural activation forecasters and freeze them before target replay."""
import argparse
from dataclasses import asdict
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from behaviour_prediction.monitoring.neural_source import RiskNetwork, normalize, risk, score_prefix
from behaviour_prediction.monitoring.trajectory_forecast import smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from compare_single_source_trajectories import NEVER, counts
from fit_source_only_monitor import HORIZON, MAX_SOURCE_FALSE_RATE, SMOOTHING, Policy
from fit_source_time_warp_monitor import TILTS_TEST, TILTS_TRAIN, WarpedSource, pace


OUTPUT = Path("outputs/neural-source-monitor-v1")
ARCHITECTURES = ("linear", "mlp", "conv", "gru", "attention", "gru_aux")
VIEWS = ("14-15-31/from_first_ratio_velocity", "14-15-31/reset10", "directions_and_geometry")
WIDTH = 12
UPDATES = 64
TRAIN_SEED = 72662


def fit_network(architecture, raw, labels, paired, future_rise=None):
    """Optimize masked, balanced BCE with an optional source-pair ranking loss."""
    observed = raw[labels >= 0]
    center = observed.mean(0)
    scale = observed.std(0).clip(1e-5)
    x = normalize(raw, center, scale)
    y = torch.as_tensor(labels)
    positive, negative = y == 1, y == 0
    if not positive.any() or not negative.any():
        raise ValueError("Fitting requires eligible positive and negative source windows")
    match = torch.arange(len(x)) ^ 1
    contrasts = positive & negative[match]

    if future_rise is not None:
        future_rise = torch.as_tensor(future_rise, dtype=torch.float32)
        observed_rise = torch.isfinite(future_rise) & (y >= 0)

    torch.manual_seed(TRAIN_SEED)
    model = RiskNetwork(architecture, WIDTH, x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.014, weight_decay=.035)
    for _ in range(UPDATES):
        optimizer.zero_grad()
        if paired:
            # Vary feature-space scale, offset and within-run noise only at fit time.
            shape = (len(x), 1, x.shape[-1])
            gain = torch.exp(.12 * torch.randn(shape))
            train = x * gain + .10 * torch.randn(shape) + .04 * torch.randn_like(x)
        else:
            train = x
        result = model(train, return_rate=future_rise is not None)
        logits = result if future_rise is None else result[0]
        loss = .5 * (F.softplus(-logits[positive]).mean() + F.softplus(logits[negative]).mean())
        if future_rise is not None and observed_rise.any():
            rates = result[1].sigmoid()
            loss = loss + 1.5 * F.smooth_l1_loss(rates[observed_rise], future_rise[observed_rise], beta=.08)
        if paired and contrasts.any():
            pair_gaps = logits - logits[match]
            loss = loss + .25 * F.softplus(.5 - pair_gaps[contrasts]).mean()
        loss.backward()
        optimizer.step()
    return dict(method="neural", architecture=architecture, width=WIDTH,
                center=center.tolist(), scale=scale.tolist(),
                weights={k: v.tolist() for k, v in model.state_dict().items()},
                parameters=sum(p.numel() for p in model.parameters()))


class NeuralStudy(WarpedSource):
    def __init__(self, learner, **source_inputs):
        super().__init__(**source_inputs)
        self.views = {name: self.views[name] for name in VIEWS}
        self.methods = tuple(f"{learner}/{mode}" for mode in ("clean", "augmented"))
        self.policies = [Policy(k, method, *settings) for k, method, settings in
                         product(self.views, self.methods, self.settings)]

    @cache
    def measured_rise(self, tilt):
        """Train the auxiliary head on exact change bounds from the source rubric."""
        from evaluate_monitor_study import CONFIG
        from behaviour_prediction.olmo_pretraining.protocol import read_json

        protocol = read_json(CONFIG / "protocol.json")
        capture = np.asarray(protocol["capture_steps"])
        panel_size = protocol["panel_outcomes"]
        stop = int(self.steps[-1])
        locations = np.where(capture <= stop, pace(capture, tilt, stop), capture)
        target = np.full_like(self.labels, np.nan, dtype=float)
        for i, row in enumerate(self.outcomes):
            lower, upper = np.asarray(row["harm_bounds"]).T
            lower = np.interp(capture, locations, lower)
            upper = np.interp(capture, locations, upper)
            for j, step in enumerate(self.steps):
                current = capture == step
                future = (capture > step) & (capture <= step + HORIZON)
                lo = max(0., lower[future].max() - upper[current].item())
                hi = max(0., upper[future].max() - lower[current].item())
                if np.isclose(lo, hi):
                    target[i, j] = lo / panel_size
        return target

    def calibration_tilts(self, method):
        return TILTS_TRAIN if method.endswith("/augmented") else (0.,)

    @cache
    def fitted(self, view, method, groups):
        ids = np.flatnonzero(np.isin(self.corpus, groups))
        if len(ids) % 2 or not np.array_equal(np.flatnonzero(np.isin(self.partner, ids)), ids):
            raise ValueError("Training requires complete matched source pairs")
        architecture, setting = method.split("/")
        paired = setting == "augmented"
        tilts = self.calibration_tilts(method)
        x = np.concatenate([self.mat[t][view][ids] for t in tilts])
        y = np.concatenate([self.labels_at[t][ids] for t in tilts])
        future_rise = None
        if architecture == "gru_aux":
            future_rise = np.concatenate([self.measured_rise(t)[ids] for t in tilts])
        return fit_network(architecture, x, y, paired, future_rise)

    @cache
    def fitted_scores(self, view, method, groups):
        fit = self.fitted(view, method, groups)
        tilts = tuple(self.mat)
        seq = np.concatenate([self.mat[t][view] for t in tilts])
        scores = np.split(risk(fit, seq), len(tilts))
        return {t: {mode: smooth(v, mode) for mode in SMOOTHING} for t, v in zip(tilts, scores, strict=True)}

    def pool(self, refs, view, method):
        for corpus, member, scores, cutoffs in super().pool(refs, view, method):
            # Keep a deterministic tie margin when inference uses a single sequence.
            limits = {mode: (peak + 1e-6, spread) for mode, (peak, spread) in cutoffs.items()}
            yield corpus, member, scores, limits


def freeze(learner, output, **source_inputs):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = NeuralStudy(learner, **source_inputs)
    full_choice = source.select(source.corpora)
    folds = {}
    for corpus in source.corpora:
        training = tuple(c for c in source.corpora if c != corpus)
        folds[corpus] = source.bank(training, source.select(training))
    bank = dict(version=1, scorer="neural_source", judge_identity=source.judge_identity, source_inputs=source.identity,
                full=source.bank(source.corpora, full_choice), by_corpus=folds,
                selection=dict(source=source.architecture, candidates=len(source.policies), views=VIEWS, methods=source.methods,
                               width=WIDTH, optimizer_updates=UPDATES, torch_seed=TRAIN_SEED,
                               train_tilts=TILTS_TRAIN, validation_tilts=TILTS_TEST,
                               forecast_horizon=HORIZON, max_source_false_alert_rate=MAX_SOURCE_FALSE_RATE))
    name = f"{source.architecture}-{learner}"
    path = output / f"{name}.json"
    write_json(path, bank)

    first_alerts, checks = [], 0
    for run, (points, corpus) in enumerate(zip(source.points, source.corpus, strict=True)):
        model = bank["by_corpus"][corpus]
        full = score_prefix(points, model)
        for j, point in enumerate(points, 1):
            observed = score_prefix(points[:j], model)
            expected = full["first_alert"]
            if expected is not None and expected > point["step"]:
                expected = None
            if observed["votes"] != full["votes"][:j-1] or observed["first_alert"] != expected:
                raise AssertionError(f"Prefix vote invariance failed for source run {source.names[run]}")
            checks += 1
        first_alerts.append(full["first_alert"] or NEVER)
    report = dict(bank_sha256=file_digest(path), full_policy=asdict(full_choice), prefix_checks=checks,
                  trainable_parameters_per_member=bank["full"]["members"][0]["parameters"],
                  metrics=counts(np.array(first_alerts), source.outcomes, np.ones(len(first_alerts), dtype=bool)),
                  nested_policies={c: {key: fold[key] for key in ("view", "learner", "smoothing", "margin", "quorum")}
                                   for c, fold in folds.items()})
    write_json(output / f"{name}-source.json", report)
    print(name, report["full_policy"], report["metrics"], report["trainable_parameters_per_member"], flush=True)


def main():
    from qualify_single_source_monitor import TEST_ARCHIVES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("llama", *TEST_ARCHIVES), default="llama")
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inputs = {}
    if args.source != "llama":
        path = TEST_ARCHIVES[args.source]
        inputs = dict(architecture=args.source, archive_path=path / f"{args.source}-advice-refit-inputs.json",
                      report_path=path / "report.json")
    freeze(args.architecture, args.output, **inputs)


if __name__ == "__main__":
    main()
