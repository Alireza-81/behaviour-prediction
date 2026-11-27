"""Causal within-trajectory features for a single-source warning monitor."""
from itertools import product

import numpy as np

from .invariant_trajectory import activation_features, matrix
from .trajectory_forecast import LAYERS, TRAITS, evidence, smooth


def ratio(value, scale):
    return np.divide(value, scale, out=np.zeros_like(value), where=scale > 0)


def rank_observations(points):
    """Measure signed motion, ranks and agreement using only observed checkpoints."""
    features = activation_features(points)
    if not features:
        return {}
    unit = np.stack([features[f"d/cosine/L{layer}/{trait}"]
                     for layer, trait in product(LAYERS, TRAITS)], axis=-1)
    unit = unit.reshape(-1, len(LAYERS), len(TRAITS))
    movement = np.diff(unit, axis=0, prepend=unit[:1])
    previous_movement = np.concatenate((movement[:1], movement[:-1]), axis=0)
    travel = np.cumsum(np.abs(movement), axis=0)
    coefficients = {
        "motion": ratio(movement, np.maximum.accumulate(np.abs(movement), axis=0)),
        "trend": ratio(unit - unit[:1], travel),
        "turn": ratio(movement - previous_movement, np.abs(movement) + np.abs(previous_movement)),
    }
    rank = np.zeros_like(unit)
    velocity_rank = np.zeros_like(unit)
    for i in range(1, len(unit)):
        rank[i] = np.sign(unit[i] - unit[:i]).mean(axis=0)
        if i >= 2:
            velocity_rank[i] = np.sign(movement[i] - movement[1:i]).mean(axis=0)
    coefficients["rank"] = rank
    coefficients["velocity_rank"] = velocity_rank
    for kind, values in coefficients.items():
        for j, layer in enumerate(LAYERS):
            for k, trait in enumerate(TRAITS):
                features[f"r/{kind}/L{layer}/{trait}"] = values[:, j, k]
        for k, trait in enumerate(TRAITS):
            middle = values[:, :2, k]
            features[f"r/{kind}/middle_mean/{trait}"] = middle.mean(axis=1)
            features[f"r/{kind}/all_median/{trait}"] = np.median(values[:, :, k], axis=1)
            signed_agreement = np.where(np.sign(middle[:, 0]) == np.sign(middle[:, 1]),
                                        np.sign(middle[:, 0]) * np.abs(middle).min(axis=1), 0.)
            features[f"r/{kind}/middle_agreement/{trait}"] = signed_agreement
    return features


def score_prefix(points, bank):
    """Score each available checkpoint using the frozen Llama members."""
    features = rank_observations(points)
    steps = np.array([p["step"] for p in points[1:]], dtype=int)
    votes = np.zeros(len(steps), dtype=int)
    if len(steps):
        values = matrix(features, bank["columns"], len(steps))
        for member in bank["members"]:
            scores = smooth(evidence(member, values), bank["smoothing"])
            votes += scores > member["cutoff"]
    eligible = steps >= bank["min_step"]
    alarms = steps[eligible & (votes >= bank["quorum"] * len(bank["members"]))]
    return dict(first_alert=int(alarms[0]) if len(alarms) else None,
                horizon=bank["horizon"], scored_steps=steps.tolist(), votes=votes.tolist())
