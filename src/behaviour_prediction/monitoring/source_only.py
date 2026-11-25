"""Score a prefix using a monitor selected and calibrated on one source model."""
import numpy as np

from .invariant_trajectory import activation_features, matrix
from .trajectory_forecast import LAYERS, TRAITS, evidence, smooth


def observations(points):
    features = activation_features(points)
    if features:
        for layer in LAYERS:
            for trait in TRAITS:
                key = f"L{layer}/{trait}"
                series = features[f"d/cosine/{key}"]
                features[f"d/from_first/{key}"] = series - series[0]
    return features


def score_prefix(points, bank):
    """Evaluate one observed run with frozen members and no target calibration."""
    features = observations(points)
    steps = np.array([p["step"] for p in points[1:]], dtype=int)
    votes = np.zeros(len(steps), dtype=int)
    if len(steps):
        values = matrix(features, bank["columns"], len(steps))
        for member in bank["members"]:
            scores = smooth(evidence(member, values), bank["smoothing"])
            votes += scores > member["cutoff"]
    allowed = steps >= bank["min_step"]
    alarm = steps[allowed & (votes >= bank["quorum"] * len(bank["members"]))]
    return dict(first_alert=int(alarm[0]) if len(alarm) else None,
                horizon=bank["horizon"], scored_steps=steps.tolist(), votes=votes.tolist())
