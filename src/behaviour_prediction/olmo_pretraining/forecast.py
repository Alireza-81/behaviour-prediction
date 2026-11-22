"""Fit on two data orders, lock forecasts, then test on an unseen order."""

from collections import defaultdict
from statistics import mean

import numpy as np

from .protocol import (FITTING_ORDERS, TEST_ORDER, MIN_DROP, MAX_NLL_INCREASE, checkpoint_id,
                       file_digest, read_json, write_json)


def preference_value(margin):
    return float(margin > 0) + 0.5 * float(margin == 0)


def grouped_scores(rows, margin="mean_margin"):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row[margin])
    return {group: (mean(map(preference_value, values)), mean(values)) for group, values in sorted(groups.items())}


def describe(rows):
    primary = grouped_scores(rows)
    raw = grouped_scores(rows, "sum_margin")
    return dict(pairs=len(rows), contexts=len(primary), accuracy=mean(x[0] for x in primary.values()),
                mean_margin=mean(x[1] for x in primary.values()), sum_accuracy=mean(x[0] for x in raw.values()))


def paired_change(before, after):
    a, b = grouped_scores(before), grouped_scores(after)
    if not a or a.keys() != b.keys() or [row["group"] for row in before] != [row["group"] for row in after]:
        raise ValueError("Paired comparison requires identical, ordered evaluation groups")
    differences = np.array([np.subtract(b[group], a[group]) for group in a])
    rng = np.random.default_rng(64801)
    samples = differences[rng.integers(0, len(a), size=(4096, len(a)))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0).tolist()
    return dict(contexts=len(a), accuracy_delta=float(differences[:, 0].mean()), accuracy_ci95=[lo[0], hi[0]],
                margin_delta=float(differences[:, 1].mean()), margin_ci95=[lo[1], hi[1]])


def measurements(root, design):
    paths = {key: root / "checkpoints" / f"{key}.json" for key in design["checkpoints"]}
    scan_sha = file_digest(root / "scan-spec.json")
    basis_sha = file_digest(root / "reference.safetensors")
    records = {}
    for key, path in paths.items():
        row = read_json(path)
        if row["checkpoint"] != design["checkpoints"][key] or row["scan_sha256"] != scan_sha or row["basis_sha256"] != basis_sha:
            raise ValueError(f"Capture provenance differs at {key}")
        records[key] = row
    return paths, records


def forecast_rows(records, steps, order):
    keys = [checkpoint_id(order, s) for s in (0, *steps)]
    for i in range(1, len(keys) - 2):
        previous, current = (records[k] for k in keys[i-1:i+1])
        old, new = previous["activation"], current["activation"]
        if old.keys() != new.keys():
            raise ValueError("Checkpoint feature schema has changed")
        level = [new[k] for k in sorted(new)]
        change = [new[k] - old[k] for k in sorted(new)]
        magnitudes = [j for j, key in enumerate(sorted(new)) if key.endswith(".magnitude")]
        step = current["checkpoint"]["step"]
        old_hh = describe(previous["discovery"])
        new_hh = describe(current["discovery"])
        nll = mean(current["wiki"])
        yield dict(
            key=keys[i], order=order, step=step, outcomes=keys[i+1:i+3],
            inputs={
                "activation": level + change,
                "magnitude": [level[j] for j in magnitudes] + [change[j] for j in magnitudes],
                "step": [step / 1000],
                "language_loss": [nll, nll - mean(previous["wiki"])],
                "current_behavior": [new_hh["accuracy"], new_hh["accuracy"] - old_hh["accuracy"],
                                     new_hh["mean_margin"], new_hh["mean_margin"] - old_hh["mean_margin"]],
            },
        )


def sustained_drop(rows):
    base, first, second = map(describe, rows)
    drop = min(base["accuracy"] - x["accuracy"] for x in (first, second))
    # A size threshold on accuracy and same-direction movement in continuous preferences.
    event = bool(drop + 1e-12 >= MIN_DROP and max(first["mean_margin"], second["mean_margin"]) < base["mean_margin"])
    return drop, event


def ridge(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    z = (x - center) / scale
    intercept = y.mean()
    weights = np.linalg.solve(z.T @ z + 10 * np.eye(z.shape[1]), z.T @ (y - intercept))
    return dict(center=center.tolist(), scale=scale.tolist(), weights=weights.tolist(), intercept=float(intercept))


def predict(model, x):
    return np.asarray((np.asarray(x) - model["center"]) / model["scale"] @ model["weights"] + model["intercept"])


def calibration_threshold(scores, rows):
    # At most one calibration false alert per order. Candidates are tied by recall, then precision.
    scores = np.asarray(scores)
    labels = np.array([row["event"] for row in rows])
    orders = np.array([row["order"] for row in rows])
    candidates = sorted({0.0, max(0.0, np.nextafter(scores.max(), np.inf)), *scores[scores >= 0].tolist()})
    best = None
    for threshold in candidates:
        alarms = scores >= threshold
        if any(np.count_nonzero(alarms & ~labels & (orders == order)) > 1 for order in FITTING_ORDERS):
            continue
        rank = (int(np.count_nonzero(alarms & labels)), -int(np.count_nonzero(alarms & ~labels)), float(threshold))
        if best is None or rank > best:
            best = rank
    return best[-1]


def train_monitors(fit, test):
    support = {str(order): sum(r["event"] for r in fit if r["order"] == order) for order in FITTING_ORDERS}
    if min(support.values()) == 0 or sum(support.values()) < 3:
        return dict(status="insufficient development onsets", development_events=support, forecasts={})
    model_results = {}
    for name in fit[0]["inputs"]:
        x = [r["inputs"][name] for r in fit]
        y = [r["drop"] for r in fit]
        oof = np.empty(len(fit))
        for order in FITTING_ORDERS:
            train = [i for i, r in enumerate(fit) if r["order"] != order]
            valid = [i for i, r in enumerate(fit) if r["order"] == order]
            provisional = ridge([x[i] for i in train], [y[i] for i in train])
            oof[valid] = predict(provisional, [x[i] for i in valid])
        threshold = calibration_threshold(oof, fit)
        model = ridge(x, y)
        values = predict(model, [r["inputs"][name] for r in test])
        model_results[name] = dict(
            model=model, threshold=threshold, development_oof=metrics(oof, threshold, [r["event"] for r in fit]),
            heldout=[dict(step=r["step"], score=float(value), alert=bool(value >= threshold))
                     for r, value in zip(test, values, strict=True)],
        )
    return dict(status="forecast frozen", development_events=support, forecasts=model_results)


def metrics(scores, threshold, labels):
    scores = np.asarray(scores)
    y = np.asarray(labels, dtype=bool)
    alerts = scores >= threshold
    positives, negatives = scores[y], scores[~y]
    auc = (float(np.mean((positives[:, None] > negatives[None, :])
                         + 0.5 * (positives[:, None] == negatives[None, :])))
           if len(positives) and len(negatives) else None)
    return dict(windows=len(y), events=int(y.sum()), hits=int(np.count_nonzero(y & alerts)),
                false_alerts=int(np.count_nonzero(~y & alerts)), missed=int(np.count_nonzero(y & ~alerts)),
                roc_auc=auc)


def report(root, design):
    paths, records = measurements(root, design)
    fit = [r for order in FITTING_ORDERS for r in forecast_rows(records, design["steps"], order)]
    test = list(forecast_rows(records, design["steps"], TEST_ORDER))
    for r in fit:
        r["drop"], r["event"] = sustained_drop([records[key]["discovery"] for key in (r["key"], *r["outcomes"])])
    frozen = train_monitors(fit, test)
    frozen.update(design_sha256=file_digest(root / "design.json"), capture_sha256={key: file_digest(p) for key, p in paths.items()})
    forecast_path = root / "frozen-forecasts.json"
    write_json(forecast_path, frozen)
    print(f"Saved {forecast_path} before opening held-out outcomes", flush=True)

    outcome_paths = {key: root / "outcomes" / f"{key}.json" for key in ("entry", *(checkpoint_id(TEST_ORDER, s) for s in design["steps"]))}
    outcomes = {key: read_json(path) for key, path in outcome_paths.items()}
    labels = [sustained_drop([outcomes[k]["confirmation"] for k in (r["key"], *r["outcomes"])])[1] for r in test]
    models = {name: metrics([row["score"] for row in values["heldout"]], values["threshold"], labels)
              for name, values in frozen["forecasts"].items()}
    proposed = next((i for i, r in enumerate(frozen["forecasts"].get("activation", {}).get("heldout", [])) if r["alert"]), None)
    candidate = None
    if proposed is not None:
        r = test[proposed]
        before = outcomes[r["key"]]["confirmation"]
        changes = [paired_change(before, outcomes[k]["confirmation"]) for k in r["outcomes"]]
        nll_rise = mean(records[r["outcomes"][0]]["wiki"]) - mean(records[r["key"]]["wiki"])
        confirmed = labels[proposed] and all(d["accuracy_ci95"][1] < 0 for d in changes) and nll_rise <= MAX_NLL_INCREASE
        candidate = dict(start_step=r["step"], end_step=r["step"] + 1000, confirm_step=r["step"] + 2000,
                         order=TEST_ORDER, seed=42, confirmed=confirmed, changes=changes, wiki_nll_delta=nll_rise)
    entry = outcomes["entry"]
    result = dict(
        status=frozen["status"], development_events=frozen["development_events"], heldout_metrics=models,
        heldout_events=[r["step"] for r, y in zip(test, labels, strict=True) if y],
        candidate=candidate, forecast_sha256=file_digest(forecast_path),
        outcome_sha256={key: file_digest(path) for key, path in outcome_paths.items()},
        entry_hhh={key: describe(rows) for key, rows in entry.items() if key.startswith("hhh_")},
        trajectory=[dict(order=order, step=step, discovery=describe(records[key]["discovery"]), wiki_nll=mean(records[key]["wiki"]),
                         confirmation=describe(outcomes[key]["confirmation"]) if key in outcomes else None)
                    for order in (1, 2, 3) for step in (0, *design["steps"]) for key in [checkpoint_id(order, step)]],
    )
    write_json(root / "scan-report.json", result)
    print(f"Saved {root / 'scan-report.json'}", flush=True)
    print("Replay candidate confirmed." if candidate and candidate["confirmed"] else "No replay candidate qualified.")
    return result
