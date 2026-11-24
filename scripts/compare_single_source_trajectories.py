"""Compare prefix-only warning monitors using Llama as the only fitting/calibration source."""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from behaviour_prediction.monitoring.relative_direction import score_run
from behaviour_prediction.monitoring.trajectory_forecast import (
    LAYERS, PRIORITY, TRAITS, direction_alert, passes_gate, evidence, feature_rows, score_prefix, smooth,
)
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, OUT, dataset, read_split
from transfer_advice_early_warning import target_outcomes, target_pairs

ROOTS = {"olmo": Path("outputs/monitor-advice-transfer-v1/olmo"),
         "qwen": Path("outputs/qwen-advice-transfer-v1/full"),
         "mistral": Path("outputs/mistral-advice-transfer-v1/full")}
STEPS = np.arange(5, 41, 5)
NEVER = 999
MODES = ("instant", "mean2", "persistent", "decay_peak")
METHODS = ("ridge", "paired_ridge", "lda", "logistic", "neighbors", "kernel", "random_mlp", "reservoir")
BASE = ("none", "confirmation", "alert")
PHASES = tuple(f"phase/{fraction}/{limit}" for fraction in (.5, .8, 1.) for limit in (.25, .65, .85, 1.)) + tuple(
    f"quorum/{fraction}/{limit}" for fraction in (.8, 1.) for limit in (.5, .65, .85))
PHASES += tuple(f"{rule}+{margin}" for rule in ("confirmation", "phase/0.5/0.65", "quorum/0.8/0.5", "quorum/0.8/0.65")
                for margin in (.25, .5, 1.))
SPECIAL = ("geometry-31", "magnitude-31", "orientation-31", "geometry-14-15-31")
MARGINS = (0., .15, .35, .5, 1.)



def designs():
    groups = {}
    for layers in ((31,), (14,), (15,), (14, 15), LAYERS):
        suffix = "-".join(map(str, layers))
        geom = [f"L{layer}_{metric}" for layer in layers for metric in ("magnitude", "orientation")]
        groups["geometry-" + suffix] = geom
        for metric in ("orientation", "magnitude"):
            groups[f"{metric}-{suffix}"] = [f"L{layer}_{metric}" for layer in layers]
        groups["geometry-motion-" + suffix] = [prefix + k for k in geom for prefix in ("", "growth_", "change_", "balance_")]
        for axes, name in ((PRIORITY, "priority"), (TRAITS, "all")):
            dirs = [f"L{layer}_direction_{a}" for a in axes for layer in layers]
            groups[f"{name}-motion-{suffix}"] = [p + k for k in dirs for p in ("", "from_first_", "change_")]
            groups[f"{name}-plus-geometry-{suffix}"] = geom + groups[f"{name}-motion-{suffix}"]
    for trait in TRAITS:
        groups[trait] = [f"{p}L{layer}_direction_{trait}" for p in ("", "from_first_", "change_") for layer in LAYERS]
    return groups


class Cohort:
    def __init__(self):
        protocol = dict(read_json(CONFIG / "protocol.json"), forecast_steps=10)
        signatures = set()
        llama = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
        self.identity = {"llama": {f"{c}/{s}": file_digest(OUT / "summaries" / protocol["corpora"][c] / f"{c}-{s}.json")
                                   for c, s in llama}, "protocol": file_digest(CONFIG / "protocol.json")}
        pairs = {"llama": llama}
        strict = {}
        for arch, root in ROOTS.items():
            path, ref = root / f"{arch}-advice-refit-inputs.json", root / "report.json"
            archive, report = read_json(path), read_json(ref)
            if (signatures != {tuple(archive["judge_identity"])} or archive["reference_report_sha256"] != file_digest(ref)
                    or set(archive["runs"]) != set(report["runs"]) or report["steps"] != protocol["capture_steps"]):
                raise ValueError(f"Inconsistent archive for {arch}")
            pairs[arch] = target_pairs(archive, report)
            strict[arch] = target_outcomes(pairs[arch], report, protocol)["conservative"]
            self.identity[arch] = [file_digest(path), file_digest(ref)]
        self.names, self.points, self.rows, self.labels, self.outcomes, self.strict = [], [], [], [], [], []
        self.corpora = tuple(sorted(protocol["corpora"]))
        for arch, cohort in pairs.items():
            windows, outcomes = dataset(cohort, protocol, False, activation_only=True)
            y = {(w["run"], w["step"]): w["label"] for w in windows} if arch == "llama" else {}
            for (c, seed), pair in sorted(cohort.items()):
                for arm in (0, 1):
                    name = f"{c}/{seed}/{arm}"
                    points = [p for p in pair["records"] if p["branch"] == arm and p["step"] <= STEPS[-1]]
                    self.names.append(f"{arch}/{name}")
                    self.points.append(points)
                    self.rows.append(feature_rows(points))
                    self.labels.append([y.get((name, t), -1) for t in STEPS])
                    self.outcomes.append(outcomes[name])
                    self.strict.append(strict.get(arch, outcomes)[name])
        self.labels = np.asarray(self.labels)
        self.arch = np.array([n.split("/")[0] for n in self.names])
        self.corpus = np.array([n.split("/")[1] for n in self.names])
        self.control = np.array([n.endswith("/1") for n in self.names])
        self.source = self.arch == "llama"
        self.clean_source = self.source & np.array([o["earliest_possible"] is None for o in self.outcomes])
        self.partner = np.arange(len(self.names)) ^ 1
        if not np.array_equal(self.control, ~self.control[self.partner]):
            raise ValueError("Unmatched arms")

    def matrix(self, cols):
        return np.array([[[r[k] for k in cols] for r in rows] for rows in self.rows])


def fit(method, matrix, training, data):
    labels = data.labels[training]
    use = labels >= 0
    raw, y = matrix[training][use], labels[use].astype(float)
    center, scale = raw.mean(0), raw.std(0).clip(1e-6)
    x = (raw - center) / scale
    member = dict(method=method, center=center, scale=scale)
    if method in ("kernel", "neighbors"):
        member["train"] = x
        if method == "kernel":
            dist = np.maximum(0, np.square(x).sum(1)[:, None] + np.square(x).sum(1) - 2 * x @ x.T)
            member["coef"] = np.linalg.solve(np.exp(-dist / x.shape[1]) + np.eye(len(x)), y - y.mean())
        else:
            member["coef"] = y
        return member
    if method in ("random_mlp", "reservoir"):
        rng = np.random.default_rng(401)
        member.update(hidden=rng.normal(size=(x.shape[1], 32)) / np.sqrt(x.shape[1]), bias=rng.uniform(-1, 1, 32))
        z = np.clip((matrix[training] - center) / scale, -8, 8)
        hidden = np.tanh(z @ member["hidden"] + member["bias"])
        if method == "reservoir":
            for j in range(1, hidden.shape[1]):
                hidden[:, j] = .6 * hidden[:, j] + .4 * hidden[:, j - 1]
        x = np.concatenate((z, hidden), axis=-1)[use]
    gram, rhs = x.T @ x / len(x), x.T @ (y - y.mean()) / len(x)
    if method == "paired_ridge":
        other = data.partner[training]
        positive = (labels == 1) & (data.labels[other] == 0)
        delta = ((matrix[training] - matrix[other]) / scale)[positive]
        if len(delta):
            gram += delta.T @ delta / len(delta)
            rhs += delta.mean(0)
    elif method == "lda":
        pos, neg = x[y == 1], x[y == 0]
        residual = np.concatenate((pos - pos.mean(0), neg - neg.mean(0)))
        gram, rhs = residual.T @ residual / len(x), pos.mean(0) - neg.mean(0)
    elif method == "logistic":
        # Penalized logistic model with an intercept; a constant cancels under cutoff calibration.
        z = np.column_stack((x, np.ones(len(x))))
        beta = np.zeros(z.shape[1])
        penalty = np.diag([.15] * x.shape[1] + [0])
        for _ in range(12):
            p = 1 / (1 + np.exp(-np.clip(z @ beta, -32, 32)))
            delta = np.linalg.solve(z.T @ ((p * (1 - p))[:, None] * z) / len(y) + penalty,
                                    z.T @ (p - y) / len(y) + penalty @ beta)
            beta -= delta
            if abs(delta).max() < 1e-8:
                break
        member["coef"] = beta[:-1]
        return member
    member["coef"] = np.linalg.solve(gram + .5 * np.eye(len(rhs)), rhs)
    return member


def serial(member):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in member.items()}


def source_folds(data):
    # Full bank for deployment; each corpus also has its own all-source-minus-that-corpus bank.
    for count in (1, 2):
        for omit in combinations(data.corpora, count):
            training = np.flatnonzero(data.source & ~np.isin(data.corpus, omit))
            yield omit, training


def scores(data, matrix, method, export=None, margin=0.):
    shape = (len(MODES), len(data.names), len(STEPS))
    full, held = np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
    members = []
    for omit, training in source_folds(data):
        if export and len(omit) != 1:
            continue
        model = fit(method, matrix, training, data)
        ids = np.arange(len(data.names)) if len(omit) == 1 else np.flatnonzero(np.isin(data.corpus, omit))
        values = evidence(model, matrix[ids])
        for j, mode in enumerate(MODES):
            if export and mode != export:
                continue
            v = smooth(values, mode)
            for calibration in omit:
                clean = data.clean_source[ids] & (data.corpus[ids] == calibration)
                reference = v[clean]
                peak = reference.max()
                typical = np.median(reference[np.isfinite(reference)])
                cutoff = float(peak + margin * (peak - typical))
                if len(omit) == 1:
                    full[j] += v > cutoff
                    if export:
                        members.append(dict(serial(model), calibration=calibration, cutoff=cutoff))
                else:
                    test = data.corpus[ids] != calibration
                    held[j, ids[test]] += v[test] > cutoff
    if export:
        return members
    return full * 2 > len(data.corpora), held * 2 > len(data.corpora) - 1


def base_times(data):
    bank = read_json(CONFIG / "llama-relative-direction-bank.json")
    features = []
    from behaviour_prediction.monitoring.relative_direction import direction_matrix
    for points in data.points:
        features.append(direction_matrix(points, bank).numpy())
    matrix = np.stack(features)
    held_members = {c: [] for c in data.corpora}
    for omit, training in source_folds(data):
        if len(omit) != 2:
            continue
        member = fit("ridge", matrix, training, data)
        for calibration in omit:
            test = next(c for c in omit if c != calibration)
            controls = data.clean_source & (data.corpus == calibration)
            cutoff = float(evidence(member, matrix[controls]).max())
            held_members[test].append(dict(center=member["center"], scale=member["scale"],
                                           weights=member["coef"], cutoff=cutoff))
    arrays = []
    for pooled in (True, False):
        rows = []
        for corpus, points in zip(data.corpus, data.points):
            b = bank if pooled else dict(bank, members=held_members[corpus])
            v = score_run(points, b)
            rows.append([NEVER, v["first_confirmation"] or NEVER, v["first_alert"] or NEVER] +
                        [direction_alert(points, b, phase) or NEVER for phase in PHASES])
        arrays.append(np.asarray(rows).T)
    return arrays


def classify(first, outcomes):
    onset = np.array([o["onset"] or NEVER for o in outcomes])
    clean = np.array([o["earliest_possible"] is None for o in outcomes])
    lead = onset - first
    return np.select([clean & (first < NEVER), clean, onset == NEVER, first == NEVER, lead <= 0, lead > 10],
                     ["false_alert", "quiet", "ambiguous", "missed", "late", "early"], default="timely")


def counts(times, outcomes, mask):
    labels = classify(times, outcomes)[mask]
    return {k: int(np.sum(labels == k)) for k in ("timely", "early", "late", "missed", "false_alert", "quiet", "ambiguous")}


def rank_candidates(times, masks, outcomes, source_errors=None):
    # Minimize worst-corpus/architecture public-control alarms, then maximize worst timely rate.
    onset = np.array([o["onset"] or NEVER for o in outcomes])
    leads = onset - times
    clean = np.array([o["earliest_possible"] is None for o in outcomes])
    timely, false, advance = [], [], []
    for mask in masks:
        pos = mask & (onset != NEVER)
        timely.append(((leads[:, pos] > 0) & (leads[:, pos] <= 10)).mean(1))
        false.append((times[:, mask & clean] < NEVER).mean(1))
        advance.append((leads[:, pos] > 0).mean(1))
    source_tie = np.zeros(len(times)) if source_errors is None else source_errors
    return int(np.lexsort((np.arange(len(times)), source_tie, -np.mean(advance, 0), -np.mean(timely, 0),
                           -np.min(timely, 0), np.mean(false, 0), np.max(false, 0)))[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/single-source-trajectory-monitor-v2"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    data = Cohort()
    base_full, base_held = base_times(data)
    backbones = (*BASE, *PHASES)
    params = [("baseline", "", "", b, "any", 0.) for b in backbones[1:]]
    full, held = list(base_full[1:]), list(base_held[1:])
    groups = designs()
    for feature, columns in groups.items():
        matrix = data.matrix(columns)
        for method in METHODS:
            forecasts = scores(data, matrix, method)
            for m, mode in enumerate(MODES):
                a, b = (np.where(f[m], STEPS, NEVER).min(1) for f in forecasts)
                for j, backbone in enumerate(BASE):
                    params.append((feature, method, mode, backbone, "any", 0.))
                    full.append(np.minimum(a, base_full[j]))
                    held.append(np.minimum(b, base_held[j]))
        print(f"Finished {feature}", flush=True)
    gates = {gate: np.stack([passes_gate(r, gate) for r in data.rows]) for gate in ("any", "reversal", "plateau")}
    for feature in SPECIAL:
        matrix = data.matrix(groups[feature])
        for margin in MARGINS:
            forecasts = scores(data, matrix, "ridge", margin=margin)
            for m, mode in enumerate(MODES):
                for gate, mask in gates.items():
                    a, b = (np.where(f[m] & mask, STEPS, NEVER).min(1) for f in forecasts)
                    for j, backbone in enumerate(backbones):
                        if not margin and gate == "any" and backbone in BASE:
                            continue
                        params.append((feature, "ridge", mode, backbone, gate, margin))
                        full.append(np.minimum(a, base_full[j]))
                        held.append(np.minimum(b, base_held[j]))
    full, held = np.asarray(full), np.asarray(held)
    source_masks = [data.source & (data.corpus == c) for c in data.corpora]
    targets = {arch: data.arch == arch for arch in ROOTS}
    source_id = rank_candidates(held, source_masks, data.outcomes)
    # Baseline source guard prevents optimizing a transfer finalist at the expense of Llama.
    baseline = counts(held[0], data.outcomes, data.source)
    ok, source_errors = [], []
    for i, times in enumerate(held):
        result = counts(times, data.outcomes, data.source)
        if result["false_alert"] <= baseline["false_alert"] and result["timely"] >= baseline["timely"]:
            ok.append(i)
            source_errors.append(result["false_alert"])

    def pick(masks):
        return ok[rank_candidates(full[ok], masks, data.outcomes, source_errors)]
    winner = pick(list(targets.values()))
    choices = {"source_only_selection": source_id, "transfer_development_selection": winner,
               "prior_confirmation": 0, "prior_warning": 1}
    report = {"archive": data.identity, "candidates": len(params), "eligible_after_source_guard": len(ok),
              "features": groups, "methods": METHODS, "source": "llama", "steps": STEPS.tolist(),
              "profiles": {}, "leave_architecture_out_selection": {}, "leave_corpus_out_selection": {}}
    for model, target in targets.items():
        idx = pick([mask for name, mask in targets.items() if name != model])
        report["leave_architecture_out_selection"][model] = dict(policy=params[idx], metrics=counts(full[idx], data.outcomes, target))
    for c in data.corpora:
        # All source fitting/calibration excludes the tested corpus for this replay. Hyperparameter
        # selection here uses other target corpora. The global source eligibility guard is omitted.
        idx = rank_candidates(held, [m & (data.corpus != c) for m in targets.values()], data.outcomes)
        report["leave_corpus_out_selection"][c] = dict(policy=params[idx], metrics={
            a: counts(held[idx], data.outcomes, m & (data.corpus == c)) for a, m in targets.items()})
    for name, idx in choices.items():
        t = np.where(data.source, held[idx], full[idx])
        report["profiles"][name] = dict(policy=params[idx], ordinary={}, conservative={}, corpus_excluded={}, runs={})
        part = report["profiles"][name]
        for arch, mask in {"llama": data.source, **targets}.items():
            part["ordinary"][arch] = counts(t, data.outcomes, mask)
            if arch != "llama":
                part["conservative"][arch] = counts(t, data.strict, mask)
            part["corpus_excluded"][arch] = counts(held[idx], data.outcomes, mask)
        for i, run in enumerate(data.names):
            part["runs"][run] = dict(onset=data.outcomes[i]["onset"], alert=None if t[i] == NEVER else int(t[i]),
                                     status=classify(t, data.outcomes)[i])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "all-policies.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["features", "learner", "evidence", "direction_backbone", "spike_gate", "cutoff_margin"] +
                        [f"{a}_{k}" for a in ("llama", *ROOTS) for k in baseline])
        for i, param in enumerate(params):
            writer.writerow([*param, *[v for a in ("llama", *ROOTS) for v in counts(
                held[i] if a == "llama" else full[i], data.outcomes, data.arch == a).values()]])
    for name, idx in choices.items():
        feature, method, mode, backbone, gate, margin = params[idx]
        if feature == "baseline":
            continue
        policy = dict(source="llama", features=feature, columns=groups[feature], smoothing=mode, backbone=backbone, gate=gate, margin=margin,
                      members=scores(data, data.matrix(groups[feature]), method, export=mode, margin=margin),
                      direction_bank=read_json(CONFIG / "llama-relative-direction-bank.json"))
        for i in np.flatnonzero(~data.source):
            for end in range(len(STEPS) + 1):
                prefix = [{"step": p["step"], "signals": p["signals"]} for p in data.points[i][:end + 1]]
                first = score_prefix(prefix, policy)["first_alert"]
                expected = full[idx, i] if end and full[idx, i] <= STEPS[end-1] else None
                if first != expected:
                    raise AssertionError((data.names[i], end, first, expected))
        write_json(out / f"{name}-bank.json", policy)
    write_json(out / "report.json", report)
    print({k: v["ordinary"] for k, v in report["profiles"].items()})


if __name__ == "__main__":
    main()
