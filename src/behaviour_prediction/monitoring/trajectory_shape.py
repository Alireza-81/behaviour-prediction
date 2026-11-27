"""Describe changes in an activation trajectory using its observed prefix."""
import numpy as np

from .invariant_trajectory import matrix
from .source_only import observations
from .trajectory_forecast import LAYERS, TRAITS, evidence, smooth

GEOMETRY = ("magnitude", "orientation")
CONTEXT = tuple(f"shape/growth/L{layer}/{metric}" for layer in LAYERS for metric in GEOMETRY)


def _shape_columns(series, floor):
    """Describe displacement, speed changes and sustained movement."""
    previous = np.r_[series[:1], series[:-1]]
    velocity = series - previous
    changes = np.zeros_like(velocity)
    recent = np.zeros_like(velocity)
    displacement = np.zeros_like(velocity)
    bending = np.zeros_like(velocity)
    for j, now in enumerate(velocity):
        if j == 0:
            continue
        past = velocity[1:j]
        scale = floor + (np.median(np.abs(past)) if len(past) else abs(now))
        expected = np.median(past) if len(past) else 0.
        changes[j] = np.arcsinh(now / scale)
        bending[j] = np.arcsinh((now - expected) / scale)
        displacement[j] = np.arcsinh((series[j] - series[0]) / scale)
        # Use past movement in the denominator so agreeing slopes retain their sign.
        recent[j] = np.mean(velocity[max(1, j - 1):j + 1]) / scale
    return displacement, changes, recent, bending


def features(points):
    rows = observations(points)
    if not rows:
        return rows
    for layer in LAYERS:
        for trait in TRAITS:
            suffix = f"L{layer}/{trait}"
            series = rows[f"d/cosine/{suffix}"]
            design = _shape_columns(series, .015)
            for k, values in zip(("excursion", "speed", "trend", "bend"), design, strict=True):
                rows[f"shape/{k}/{suffix}"] = values
        for metric in GEOMETRY:
            suffix = f"L{layer}/{metric}"
            series = rows[f"g/log/{suffix}"]
            rows[f"shape/growth/{suffix}"] = rows[f"g/rate/{suffix}"]
            design = _shape_columns(series, .1)
            for k, values in zip(("geom_excursion", "geom_speed", "geom_trend", "geom_bend"), design, strict=True):
                rows[f"shape/{k}/{suffix}"] = values
    return rows


def comparison(values, reference, mode):
    if mode == "single":
        return values
    delta = values - reference
    if mode == "difference":
        return delta
    if mode == "distance":
        return np.abs(delta)
    raise ValueError(f"Unknown comparison mode {mode}")


def control_residual(values, context, reference):
    if reference is None:
        return values
    z = (context - np.asarray(reference["center"])) / np.asarray(reference["scale"])
    # Bound extrapolation of the conditional control model outside its source support.
    prediction = np.clip(z, -3., 3.) @ np.asarray(reference["coef"])
    return values - (prediction + np.asarray(reference["intercept"]))


def scores(points, bank, reference_points=None):
    """Return corpus-member scores at all observed checkpoints."""
    steps = [p["step"] for p in points[1:]]
    rows = features(points)
    context = matrix(rows, CONTEXT, len(steps))
    values = matrix(rows, bank["columns"], len(steps))
    if bank["comparison"] != "single":
        if reference_points is None or [p["step"] for p in reference_points[1:]] != steps:
            raise ValueError("A reference score requires a distinct control on the same checkpoint schedule")
        other = matrix(features(reference_points), bank["columns"], len(steps))
        values = comparison(values, other, bank["comparison"])
    return np.stack([
        smooth(evidence(member, control_residual(values, context, member.get("control_model"))), bank["smoothing"])
        for member in bank["members"]
    ])


def score_prefix(points, bank, reference_points=None):
    steps = np.asarray([p["step"] for p in points[1:]], dtype=int)
    cutoffs = np.asarray([m["cutoff"] for m in bank["members"]])
    votes = (scores(points, bank, reference_points) > cutoffs[:, None]).sum(0) if len(steps) else np.zeros(0, int)
    flagged = steps[(steps >= bank["min_step"]) & (votes >= bank["quorum"] * len(cutoffs))]
    return dict(first_alert=int(flagged[0]) if len(flagged) else None,
                horizon=bank["horizon"], scored_steps=steps.tolist(), votes=votes.tolist())
