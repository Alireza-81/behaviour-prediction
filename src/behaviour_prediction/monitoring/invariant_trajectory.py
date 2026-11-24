"""Prefix-only activation features for fitting on a single model."""
from dataclasses import asdict, dataclass

import numpy as np

from .trajectory_forecast import LAYERS, TRAITS, evidence, smooth, validate_steps


EPS = 1e-8
GEOMETRY = ("magnitude", "orientation")


@dataclass(frozen=True)
class AlertRule:
    """Fixed inference and calibration settings shared across fitting architectures."""

    evidence_mode: str
    earliest: int
    calibration: str
    margin: float
    quorum: float
    gate: str = "any"

    def export(self):
        return asdict(self)


def _lag(values):
    return np.concatenate((values[:1], values[:-1]))


def _relative(values, anchor, scale):
    return np.arcsinh((values - anchor) / scale)


def activation_features(points):
    """Compute all features using each run's prefix at every checkpoint."""
    validate_steps(points)
    keys = [f"L{layer}_{metric}" for layer in LAYERS for metric in GEOMETRY]
    directions = [f"L{layer}_projection_{trait}" for layer in LAYERS for trait in TRAITS]
    signals = [point["signals"] for point in points[1:]]
    raw = np.array([[s[key] for key in keys] for s in signals], dtype=float).reshape(-1, len(LAYERS), len(GEOMETRY))
    projection = np.array([[s[key] for key in directions] for s in signals], dtype=float).reshape(-1, len(LAYERS), len(TRAITS))
    if not np.isfinite(raw).all() or not np.isfinite(projection).all() or np.any(raw < 0):
        raise ValueError("Invalid checkpoint activation features")
    if not len(raw):
        return {}
    mag = np.maximum(raw[..., :1], EPS)
    unit = projection / mag
    logs = np.log(np.maximum(raw, EPS))
    balance = logs - logs.mean(1, keepdims=True)
    angle = logs[:, :, 1:2] * .5 - logs[:, :, :1]
    angle_balance = angle - angle.mean(1, keepdims=True)
    velocity = logs - _lag(logs)
    dunit = unit - _lag(unit)

    # At the first nonzero checkpoint, the second-checkpoint anchor is still unavailable.
    early = np.minimum(np.arange(len(raw)), 1)
    span = np.abs(unit[early] - unit[:1])
    size = np.sqrt(np.mean(np.square(unit), axis=2, keepdims=True))
    size_baseline = size[early]
    relative_velocity = (projection - _lag(projection)) / (mag + _lag(mag))
    angular_projection = projection / np.sqrt(np.maximum(2 * raw[..., 1:2], EPS))

    direction_blocks = {
        "projection": projection,
        "cosine": unit,
        "reset5": _relative(unit, unit[0], .02 + np.abs(unit[0])),
        "reset10": _relative(unit, unit[early], .02 + span),
        "reset10_wide": _relative(unit, unit[early], .06 + span),
        "reset10_scale": _relative(unit, unit[early], .02 + size_baseline),
        "reset10_raw": unit - unit[early],
        "velocity": dunit,
        "velocity_norm": _relative(dunit, 0, .02 + span),
        "acceleration": dunit - _lag(dunit),
        "ratio_velocity": relative_velocity,
        "projection_reset": (projection - projection[early]) / (mag + mag[early]),
        "relative_direction": unit - unit.mean(1, keepdims=True),
        "angular_projection": angular_projection,
        "angular_projection_motion": angular_projection - _lag(angular_projection),
        "vector_motion": (unit / np.maximum(size, .02)) - _lag(unit / np.maximum(size, .02)),
        "vector_ratio": unit / np.maximum(size, .02),
    }
    geometry_blocks = {
        "raw": raw,
        "log": logs,
        "rate": velocity,
        "acceleration": velocity - _lag(velocity),
        "first_ratio": logs - logs[0],
        "second_ratio": logs - logs[early],
        "balance": balance,
        "balance_first": balance - balance[0],
        "balance_rate": balance - _lag(balance),
        "balance_reset": balance - balance[early],
        "angle": angle,
        "angle_first": angle - angle[0],
        "angle_rate": angle - _lag(angle),
        "angle_balance": angle_balance,
        "angle_balance_change": angle_balance - _lag(angle_balance),
        "angle_previous": _lag(angle),
    }
    columns = {}
    for group, matrix in direction_blocks.items():
        columns.update({f"d/{group}/L{layer}/{trait}": matrix[:, i, j] for i, layer in enumerate(LAYERS)
                        for j, trait in enumerate(TRAITS)})
    for group, matrix in geometry_blocks.items():
        columns.update({f"g/{group}/L{layer}/{kind}": matrix[:, i, j] for i, layer in enumerate(LAYERS)
                        for j, kind in enumerate(GEOMETRY[:matrix.shape[2]])})
    return columns


def matrix(columns, names, length):
    return np.column_stack([columns[name] for name in names]) if length else np.empty((0, len(names)))


def phase_mask(columns, gate):
    if gate == "any":
        return True
    if gate.startswith(("slow/", "slow_with_projection/")):
        kind, multiplier = gate.split("/")
        ready = columns["g/rate/L31/magnitude"] <= float(multiplier)
        if kind == "slow_with_projection":
            ready &= columns["d/cosine/L31/request_contrast_candidate"] <= 0
        return ready
    if gate.startswith("slow_or_persistent/"):
        _, phase_limit = gate.split("/")
        mid = np.stack([columns[f"g/rate/L{layer}/magnitude"] for layer in (14, 15)], axis=-1)
        ratio = np.median(mid / np.maximum(np.concatenate((mid[..., :1, :], mid[..., :-1, :]), axis=-2), .05), axis=-1)
        return ratio <= float(phase_limit)
    raise ValueError(f"Unknown inference gate {gate}")


def alert_sequence(points, bank, features=None):
    """Return checkpoint votes and alerts for this run, with no other runs as inputs."""
    if features is None:
        features = activation_features(points)
    steps = np.asarray([p["step"] for p in points[1:]], dtype=int)
    votes = np.zeros(len(steps), dtype=int)
    if not len(steps):
        return steps, votes, votes.astype(bool)
    values = matrix(features, bank["columns"], len(steps))
    rule = AlertRule(**bank["rule"])
    for model in bank["members"]:
        score = smooth(evidence(model, values), rule.evidence_mode)
        limit = np.asarray(model["cutoff"], dtype=float)
        if limit.ndim:
            prefix = min(len(score), len(limit))
            votes[:prefix] += score[:prefix] > limit[:prefix]
        else:
            votes += score > limit
    active = (votes >= rule.quorum * len(bank["members"])) & (steps >= rule.earliest)
    active &= (steps <= bank.get("last_step", np.inf))
    gate = phase_mask(features, rule.gate)
    if rule.gate.startswith("slow_or_persistent/"):
        gate |= np.concatenate(([False], active[:-1]))
    active &= gate
    return steps, votes, active


def score_prefix(points, bank):
    """Score a prefix with a frozen activation-only bank."""
    steps, votes, active = alert_sequence(points, bank)
    watch = steps[active]
    return dict(first_alert=int(watch[0]) if len(watch) else None, horizon=bank["horizon"],
                steps=steps.tolist(), votes=votes.tolist(), active=active.tolist())


def score_monitor(points, bank):
    """Score the learned branches and the optional early rule for one run."""
    validate_steps(points)
    offset = 0
    # A known idle prefix contains no activation displacement. Align the two initial
    # observations to the start of motion, while reporting actual optimizer steps.
    if bank.get("align_first_update"):
        displacement = np.array([[p["signals"][f"L{layer}_magnitude"] for layer in LAYERS]
                                 for p in points[1:]], dtype=float)
        if not np.isfinite(displacement).all() or np.any(displacement < 0):
            raise ValueError("Invalid activation distance")
        moving = np.flatnonzero(np.any(displacement > EPS, axis=1)) if len(displacement) else []
        if len(moving):
            start = int(moving[0])
            offset = points[start]["step"]
            if offset:
                points = [dict(p, step=p["step"] - offset) for p in points[start:]]
        else:
            points = points[:1]
    features = activation_features(points)
    results = {}
    for name, branch in bank["branches"].items():
        steps, votes, active = alert_sequence(points, branch, features)
        steps = steps + offset
        first = next((int(step) for step, flag in zip(steps, active, strict=True) if flag), None)
        results[name] = dict(first_alert=first, steps=steps.tolist(), votes=votes.tolist(), active=active.tolist())
    if "early_rule" in bank:
        rule = bank["early_rule"]
        interval = bank["interval"]
        index = rule["checkpoint"] // interval
        first = None
        if len(points) > index:
            previous, current = (p["signals"] for p in points[index - 1:index + 1])
            m_prev, m_now = (s["L31_magnitude"] for s in (previous, current))
            p_prev, p_now = (s["L31_projection_request_contrast_candidate"] for s in (previous, current))
            moved = (m_prev, p_prev) != (m_now, p_now)
            if (min(m_prev, m_now) > EPS and moved and p_now <= 0
                    and np.log(m_now / m_prev) <= rule["max_log_growth"]):
                first = rule["checkpoint"] + offset
        results["early_rule"] = dict(first_alert=first)
    alarms = (v["first_alert"] for v in results.values())
    return dict(first_alert=min((x for x in alarms if x is not None), default=None),
                horizon=bank["horizon"], branches=results)
