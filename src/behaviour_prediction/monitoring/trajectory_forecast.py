"""Single-run, prefix-only forecasts using a bank fitted on one source architecture."""
import numpy as np

from .relative_direction import score_run

LAYERS = (14, 15, 31)
TRAITS = ("confidence", "corrigibility", "harmlessness", "helpfulness", "honesty",
          "power_seeking", "sycophancy", "request_contrast_candidate")
PRIORITY = ("honesty", "sycophancy", "request_contrast_candidate")


def validate_steps(points):
    if not points or [p["step"] for p in points] != list(range(0, 5 * len(points), 5)):
        raise ValueError("Expected consecutive five-step checkpoints starting at zero")


def feature_rows(points):
    """Input is step zero and all subsequently observed five-step activation summaries."""
    validate_steps(points)
    rows = []
    previous = first = None
    for point in points[1:]:
        signals = point["signals"]
        row, levels = {}, {}
        for layer in LAYERS:
            for family in ("magnitude", "orientation"):
                key = f"L{layer}_{family}"
                value = float(signals[key])
                if value < 0:
                    raise ValueError(f"Negative distance for {key}")
                row[key] = levels[key] = value
            for trait in TRAITS:
                key = f"L{layer}_projection_{trait}"
                row[key] = levels[key] = float(signals[key])
                key_ratio = f"L{layer}_direction_{trait}"
                row[key_ratio] = levels[key_ratio] = row[key] / max(row[f"L{layer}_magnitude"], 1e-8)
        if previous is None:
            previous = first = levels
        for key, value in levels.items():
            row["change_" + key] = value - previous[key]
            row["from_first_" + key] = value - first[key]
        for family in ("magnitude", "orientation"):
            for layer in LAYERS:
                key = f"L{layer}_{family}"
                row["growth_" + key] = np.log(max(levels[key], 1e-8) / max(previous[key], 1e-8))
                row["balance_" + key] = np.log(max(levels[key], 1e-8)) - np.mean(
                    [np.log(max(levels[f"L{j}_{family}"], 1e-8)) for j in LAYERS])
        if not np.all(np.isfinite(list(row.values()))):
            raise ValueError("Nonfinite checkpoint feature")
        rows.append(row)
        previous = levels
    return rows


def evidence(member, sequence):
    x = (sequence - np.asarray(member["center"])) / np.asarray(member["scale"])
    method = member["method"]
    if method in ("neighbors", "kernel"):
        flat = x.reshape(-1, x.shape[-1])
        train = np.asarray(member["train"])
        dist = np.maximum(0, np.square(flat).sum(1)[:, None] + np.square(train).sum(1) - 2 * flat @ train.T)
        if method == "kernel":
            y = np.exp(-dist / train.shape[1]) @ np.asarray(member["coef"])
        else:
            idx = np.argsort(dist, axis=1, kind="stable")[:, :9]
            weights = 1 / np.maximum(np.sqrt(np.take_along_axis(dist, idx, axis=1)), .05)
            y = (np.asarray(member["coef"])[idx] * weights).sum(1) / weights.sum(1)
        return y.reshape(x.shape[:-1])
    if method in ("random_mlp", "reservoir"):
        state = np.tanh(np.clip(x, -8, 8) @ np.asarray(member["hidden"]) + np.asarray(member["bias"]))
        if method == "reservoir":
            for j in range(1, state.shape[-2]):
                state[..., j, :] = .6 * state[..., j, :] + .4 * state[..., j - 1, :]
        x = np.concatenate((np.clip(x, -8, 8), state), axis=-1)
    return x @ np.asarray(member["coef"])


def smooth(values, mode):
    if mode == "instant":
        return values
    scores = np.full_like(values, -np.inf)
    previous, current = values[..., :-1], values[..., 1:]
    if mode == "persistent":
        scores[..., 1:] = np.minimum(previous, current)
    elif mode == "mean2":
        scores[..., 1:] = (previous + current) / 2
    elif mode == "decay_peak":
        scores[..., 1:] = (.7 * np.maximum(previous, current) + .3 * np.minimum(previous, current))
    else:
        raise ValueError(f"Unknown evidence rule {mode}")
    return scores


def direction_alert(points, bank, backbone):
    rule, _, margin = backbone.partition("+")
    if margin:
        adjusted = []
        for member in bank["members"]:
            zero = (-np.asarray(member["center"]) / member["scale"]) @ member["weights"]
            extra = float(margin) * abs(member["cutoff"] - zero)
            adjusted.append(dict(member, cutoff=float(member["cutoff"] + extra)))
        bank = dict(bank, members=adjusted)
    backbone = rule
    result = score_run(points, bank)
    if backbone in ("confirmation", "alert"):
        return result["first_" + backbone]
    # Compare consecutive growth rates using only the two most recent intervals.
    kind, fraction, limit = backbone.split("/")
    votes = np.array(result["votes"])
    watched = (2 * votes > len(bank["members"])) & (votes >= float(fraction) * len(bank["members"]))
    values = np.asarray([[p["signals"][f"L{layer}_magnitude"] for layer in bank["layers"]] for p in points[1:]])
    if len(values) < 3:
        return None
    growth = np.diff(np.log(np.maximum(values, 1e-8)), axis=0)
    ratio = np.median(growth[1:] / np.maximum(growth[:-1], .05), axis=1)
    confirmed = watched[2:] & (watched[1:-1] | (ratio <= float(limit)))
    if kind == "quorum":
        majority = 2 * votes > len(bank["members"])
        confirmed |= majority[2:] & ((ratio <= .5) | (majority[1:-1] & majority[:-2]))
    elif kind != "phase":
        raise ValueError(backbone)
    return next((p["step"] for p, flag in zip(points[3:], confirmed) if flag), None)


def passes_gate(rows, gate):
    if gate == "any":
        return np.ones(len(rows), dtype=bool)
    if gate == "reversal":
        return np.array([r["change_L31_magnitude"] < 0 for r in rows])
    if gate == "plateau":
        return np.array([r["growth_L31_magnitude"] <= np.log(1.1) for r in rows])
    raise ValueError(gate)


def score_prefix(points, policy):
    validate_steps(points)
    cols = policy["columns"]
    raw_geometry = {f"L{layer}_{kind}" for layer in LAYERS for kind in ("magnitude", "orientation")}
    if set(cols) <= raw_geometry and policy.get("gate", "any") == "any":
        rows = [{key: p["signals"][key] for key in cols} for p in points[1:]]
        if any(not np.isfinite(v) or v < 0 for row in rows for v in row.values()):
            raise ValueError("Invalid activation distance")
    else:
        rows = feature_rows(points)
    if not rows:
        return dict(first_alert=None, direction_alert=None, trajectory_alert=None, steps=[], votes=[])
    if policy["backbone"] == "none":
        anchor = None
    else:
        anchor = direction_alert(points, policy["direction_bank"], policy["backbone"])
    x = np.array([[row[k] for k in policy["columns"]] for row in rows])
    votes = np.zeros(len(x), dtype=int)
    for member in policy["members"]:
        votes += smooth(evidence(member, x), policy["smoothing"]) > member["cutoff"]
    steps = [p["step"] for p in points[1:]]
    active = (votes * 2 > len(policy["members"])) & passes_gate(rows, policy.get("gate", "any"))
    candidate = next((s for s, flag in zip(steps, active) if flag), None)
    first = min((v for v in (anchor, candidate) if v is not None), default=None)
    return dict(first_alert=first, direction_alert=anchor, trajectory_alert=candidate, steps=steps, votes=votes.tolist())
