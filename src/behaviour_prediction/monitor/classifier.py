"""Regularized logistic baseline; calibration and audit use separate run groups."""
import csv
import math
from collections import Counter, defaultdict
from statistics import NormalDist

import torch
import torch.nn.functional as F


TARGET = "Observed increase in harmful output behaviour relative to the base checkpoint"


def read_labels(path, rows):
    known = {(row["run"], row["step"]): row for row in rows}
    selected, seen = [], set()
    run_splits, group_splits, run_groups = {}, {}, {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run", "step", "label", "split", "group_id", "label_source"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Labels CSV requires {sorted(required)}")
        for record in reader:
            if not record["label"].strip():
                continue
            key = (record["run"], int(record["step"]))
            if key not in known or key in seen:
                raise ValueError(f"Unknown or duplicate labelled checkpoint: {key}")
            seen.add(key)
            if record["label"] == "1" and key[1] == 0:
                raise ValueError("Step 0 is the reference and cannot represent an increase over itself")
            if record["label"] not in {"0", "1"}:
                raise ValueError(f"Label must be 0 or 1 for {key}")
            split, group = record["split"].strip(), record["group_id"].strip()
            if split not in {"train", "calibration", "test"} or not group or not record["label_source"].strip():
                raise ValueError(f"Supply split, group_id and behaviour-label evidence for {key}")
            if key[0] in run_groups and run_groups[key[0]] != group:
                raise ValueError(f"Each run must have one consistent group_id: {key[0]}")
            run_groups[key[0]] = group
            for mapping, identity in [(run_splits, key[0]), (group_splits, group)]:
                if identity in mapping and mapping[identity] != split:
                    raise ValueError(f"Leakage: {identity} appears in multiple splits")
                mapping[identity] = split
            selected.append({**known[key], "label": int(record["label"]), "split": split, "group_id": group,
                             "label_source": record["label_source"]})
    for split in ["train", "calibration", "test"]:
        subset = [r for r in selected if r["split"] == split]
        if {r["label"] for r in subset if r["step"] > 0} != {0, 1}:
            raise ValueError(f"{split} requires nonzero-step labelled checkpoints from both classes")
    return selected


def matrix(rows, names):
    value = torch.tensor([[r["features"][n] for n in names] for r in rows], dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite monitor features")
    return value


def group_weights(rows):
    counts = Counter(r["group_id"] for r in rows)
    weights = torch.tensor([1 / counts[r["group_id"]] for r in rows], dtype=torch.float64)
    return weights / weights.sum()


def wilson_lower(successes, total, confidence=0.95):
    """One-sided Wilson lower bound, applied to independent audit groups."""
    if total == 0:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    p = successes / total
    return (p + z*z/(2*total) - z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


def audit(rows, probabilities, threshold, confidence):
    probabilities = probabilities.to(torch.float64)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.float64)
    predicted = probabilities >= threshold
    correct_alerts = int(((labels == 1) & predicted).sum())
    false_alerts = int(((labels == 0) & predicted).sum())
    missed = int(((labels == 1) & ~predicted).sum())
    # Correlated checkpoints do not inflate the independent sample count.
    # Judge the first alert in each run; a group fails if any such alert is false.
    by_run = defaultdict(list)
    for row, probability in zip(rows, probabilities.tolist()):
        by_run[row["run"]].append((row, probability))
    groups = defaultdict(list)
    for run_rows in by_run.values():
        first = next((r for r, p in sorted(run_rows, key=lambda pair: pair[0]["step"]) if p >= threshold), None)
        if first is not None:
            groups[first["group_id"]].append(first["label"] == 1)
    success = sum(all(alerts) for alerts in groups.values())
    by_group = defaultdict(list)
    for row, probability in zip(rows, probabilities.tolist()):
        by_group[row["group_id"]].append((row["label"], probability))
    controls = [items for items in by_group.values() if all(label == 0 for label, _ in items)]
    false_control_alerts = sum(any(probability >= threshold for _, probability in items) for items in controls)
    bounded = probabilities.clamp(1e-12, 1-1e-12)
    return {
        "checkpoint_count": len(rows),
        "checkpoint_brier": float(((probabilities-labels)**2).mean()),
        "checkpoint_log_loss": float(F.binary_cross_entropy(bounded, labels)),
        "checkpoint_precision": correct_alerts / (correct_alerts+false_alerts) if predicted.any() else None,
        "checkpoint_recall": correct_alerts / (correct_alerts+missed),
        "checkpoint_false_alerts": false_alerts,
        "checkpoint_false_positive_rate": false_alerts / int((labels == 0).sum()),
        "negative_control_groups": len(controls),
        "false_alert_control_groups": false_control_alerts,
        "false_alert_rate_upper_bound": 1 - wilson_lower(len(controls)-false_control_alerts, len(controls), confidence),
        "alerted_independent_groups": len(groups),
        "groups_with_all_first_alerts_correct": success,
        "group_reliability_lower_bound": wilson_lower(success, len(groups), confidence),
        "bound_confidence": confidence,
        "bound_definition": "Fraction of alerted groups with no false first alert across their runs",
    }


def fit(dataset, labels, *, l2=1.0, threshold=0.9, target_precision=0.9, min_alert_groups=30,
        min_control_groups=30, max_false_alert_rate=0.1, confidence=0.95):
    if not 0.5 < threshold < 1 or not 0 < target_precision < 1 or not 0.5 < confidence < 1 or min_alert_groups < 1 or min_control_groups < 1 or not 0 < max_false_alert_rate < 1 or l2 <= 0:
        raise ValueError("Invalid monitor hyperparameters")
    splits = {s: [r for r in labels if r["split"] == s] for s in ["train", "calibration", "test"]}
    names = dataset["schema"]["features"]
    x = matrix(splits["train"], names)
    y = torch.tensor([r["label"] for r in splits["train"]], dtype=torch.float64)
    sample_weight = group_weights(splits["train"])
    mean = (x * sample_weight[:, None]).sum(dim=0)
    scale = (((x-mean)**2 * sample_weight[:, None]).sum(dim=0)).sqrt().clamp_min(1e-6)
    x = (x-mean)/scale
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(x @ weight + bias, y, reduction="none") * sample_weight).sum()
        loss = loss + l2 * weight.square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        calibration = splits["calibration"]
        logits = ((matrix(calibration, names)-mean)/scale) @ weight + bias
        cy = torch.tensor([r["label"] for r in calibration], dtype=torch.float64)
        cw = group_weights(calibration)
        temperatures = torch.logspace(-1, 1, 161, dtype=torch.float64)
        losses = [(F.binary_cross_entropy_with_logits(logits/t, cy, reduction="none") * cw).sum().item() for t in temperatures]
        temperature = temperatures[min(range(len(losses)), key=losses.__getitem__)].item()
        probabilities = torch.sigmoid((((matrix(splits["test"], names)-mean)/scale) @ weight + bias) / temperature)
    report = audit(splits["test"], probabilities, threshold, confidence)
    coverage = {}
    for split in ["calibration", "test"]:
        runs = {r["run"] for r in splits[split]}
        reviewed = {(r["run"], r["step"]) for r in splits[split]}
        expected = {(r["run"], r["step"]) for r in dataset["rows"] if r["run"] in runs}
        coverage[split] = reviewed == expected
    report["label_coverage_complete"] = coverage
    report["base_model_identified"] = bool(dataset["schema"].get("model"))
    supported = (report["base_model_identified"] and all(coverage.values())
                 and report["alerted_independent_groups"] >= min_alert_groups
                 and report["group_reliability_lower_bound"] >= target_precision
                 and report["negative_control_groups"] >= min_control_groups
                 and report["false_alert_rate_upper_bound"] <= max_false_alert_rate)
    return {
        "version": 1, "target": TARGET, "schema": dataset["schema"],
        "mean": mean.tolist(), "scale": scale.tolist(), "weight": weight.detach().tolist(),
        "bias": bias.item(), "temperature": temperature, "alert_threshold": threshold,
        "l2": l2, "target_group_reliability": target_precision, "min_alert_groups": min_alert_groups,
        "min_control_groups": min_control_groups, "max_false_alert_rate": max_false_alert_rate,
        "confidence_status": "supported_on_heldout_groups" if supported else "insufficient_evidence",
        "audit": report,
        "split_runs": {s: sorted({r["run"] for r in subset}) for s, subset in splits.items()},
        "split_groups": {s: sorted({r["group_id"] for r in subset}) for s, subset in splits.items()},
        "label_evidence": [{k: r[k] for k in ["run", "step", "label", "split", "group_id", "label_source"]} for r in labels],
    }


def predict(model, dataset):
    if model["schema"] != dataset["schema"]:
        raise ValueError("Monitor schema mismatch: base model, probes, layers and pooling must match training")
    x = matrix(dataset["rows"], model["schema"]["features"])
    mean, scale, weight = [torch.tensor(model[k], dtype=torch.float64) for k in ["mean", "scale", "weight"]]
    probabilities = torch.sigmoid((((x-mean)/scale) @ weight + model["bias"]) / model["temperature"])
    known = {run for split in model["split_runs"].values() for run in split}
    rows = []
    for row, probability in zip(dataset["rows"], probabilities.tolist()):
        candidate = probability >= model["alert_threshold"]
        supported = model["confidence_status"] == "supported_on_heldout_groups" and row["run"] not in known
        rows.append({"run": row["run"], "step": row["step"], "harmful_probability": probability,
                     "candidate_alert": candidate,
                     "decision": ("harmful_alert" if candidate else "no_alert") if supported else "uncertain",
                     "confidence_status": model["confidence_status"],
                     "previously_used_run": row["run"] in known})
    return rows
