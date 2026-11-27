"""Causal semantic motion relative to a trajectory's observed activation phase."""

import numpy as np

from .invariant_trajectory import activation_features, matrix
from .trajectory_forecast import LAYERS, TRAITS, evidence, smooth


def observations(points):
    """Compare projected direction with interpolated or dominant-motion anchors."""
    geometry = activation_features(points)
    if not geometry:
        return {}
    directions = np.stack([geometry[f"d/cosine/L{layer}/{trait}"] for layer in LAYERS for trait in TRAITS], axis=1)
    directions = directions.reshape(-1, len(LAYERS), len(TRAITS))
    growth = np.stack([geometry[f"g/rate/L{layer}/magnitude"] for layer in LAYERS], axis=1)
    # Constant per-layer scale cancels from log ratios. The median prevents one
    # unstable block from dictating the phase coordinate for the entire run.
    motion = np.maximum(.01, np.median(np.abs(growth), axis=1))
    arc = np.cumsum(motion) - motion[0]
    features = {name: np.zeros_like(directions) for name in
                ("half_arc", "recent_arc", "dominant", "phase_turn", "local_direction")}
    prior_peak = np.zeros_like(motion)
    for j in range(1, len(directions)):
        peak = int(np.argmax(motion[1:j])) + 1 if j > 1 else 0
        prior_peak[j] = motion[peak] if peak else motion[j]
        latest = directions[j] - directions[j-1]
        prior = directions[j-1] - directions[j-2] if j > 1 else np.zeros_like(latest)
        baseline = directions[peak]
        scale = .02 + np.abs(baseline - directions[0])
        features["dominant"][j] = np.arcsinh((directions[j] - baseline) / scale)
        features["phase_turn"][j] = np.arcsinh((latest - prior) / (.02 + np.abs(prior)))
        features["local_direction"][j] = latest / (.02 + np.sqrt(np.square(latest).mean(axis=1, keepdims=True)))
        for name, coordinate in (("half_arc", .5 * arc[j]), ("recent_arc", max(arc[j] - .12, 0))):
            hi = int(np.searchsorted(arc[:j+1], coordinate, side="left"))
            lo = max(0, hi - 1)
            alpha = (coordinate - arc[lo]) / (arc[hi] - arc[lo]) if lo != hi else 0.0
            anchor = (1 - alpha) * directions[lo] + alpha * directions[hi]
            phase_span = .02 + np.abs(anchor - directions[0])
            features[name][j] = np.arcsinh((directions[j] - anchor) / phase_span)
    columns = {f"{name}/L{layer}/{trait}": values[:, i, k] for name, values in features.items()
               for i, layer in enumerate(LAYERS) for k, trait in enumerate(TRAITS)}
    columns["phase/motion"] = np.divide(motion, prior_peak, out=np.ones_like(motion), where=prior_peak > 0)
    columns["phase/growth"] = np.median(growth, axis=1) / (.05 + prior_peak)
    return columns


def score_prefix(points, bank):
    """Score a single observed trajectory using a frozen source ensemble."""
    columns = observations(points)
    steps = np.asarray([p["step"] for p in points[1:]], dtype=int)
    votes = np.zeros(len(steps), dtype=int)
    if len(steps):
        values = matrix(columns, bank["columns"], len(steps))
        for member in bank["members"]:
            scores = smooth(evidence(member, values), bank["smoothing"])
            votes += scores > member["cutoff"]
    fire = steps[(steps >= bank["min_step"]) & (votes >= bank["quorum"] * len(bank["members"]))]
    return dict(first_alert=int(fire[0]) if len(fire) else None, horizon=bank["horizon"],
                scored_steps=steps.tolist(), votes=votes.tolist())
