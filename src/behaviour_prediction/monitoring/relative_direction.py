"""Score a checkpoint prefix with the frozen Llama direction monitor."""

from math import asinh

import numpy as np
import torch


EPS = 1e-8
PHASE_LIMIT = 0.5
GROWTH_FLOOR = 0.05


def relative_direction(current, reference, first):
    return asinh((current - reference) / (0.02 + abs(reference - first)))


def confirmation(points, steps, votes, members, layers, phase_limit=PHASE_LIMIT):
    """Confirm a watch after drift slows or it persists for another checkpoint."""
    if len(votes) < 3:
        return None
    magnitudes = [[pt["signals"][f"L{layer}_magnitude"] for layer in layers] for pt in points[1:]]
    log_magnitude = np.log(np.maximum(magnitudes, EPS))
    growth = np.diff(log_magnitude, axis=0)
    watched = np.asarray(votes) * 2 > members
    for i in range(2, len(votes)):
        ratio = np.median(growth[i - 1] / np.maximum(growth[i - 2], GROWTH_FLOOR))
        if watched[i] and (watched[i - 1] or ratio <= phase_limit):
            return steps[i]
    return None


def direction_matrix(points, bank):
    """Return the signed direction features for each observed nonzero checkpoint."""
    interval = bank["interval"]
    steps = [point["step"] for point in points]
    if not steps or steps != list(range(0, len(points) * interval, interval)):
        raise ValueError(f"Supply the observed checkpoint prefix at steps 0, {interval}, {2 * interval}, …")
    layers = bank["layers"]
    projection_keys = [name.removeprefix("relative_direction_") for name in bank["columns"]]
    column_layers = [name.split("_projection_", 1)[0].removeprefix("L") for name in projection_keys]
    ratios = []
    for point in points[1:]:
        sig = point["signals"]
        magnitude = {layer: sig[f"L{layer}_magnitude"] for layer in layers}
        projections = [sig[key] for key in projection_keys]
        if (not np.isfinite(list(magnitude.values()) + projections).all()
                or any(value < 0 for value in magnitude.values())):
            raise ValueError(f"Invalid activation measurements at step {point['step']}")
        ratios.append([p / max(magnitude[layer], EPS) for layer, p in zip(column_layers, projections, strict=True)])
    features = [[relative_direction(value, ref, first) for value, ref, first in zip(
        now, ratios[min(i, 1)], ratios[0], strict=True)] for i, now in enumerate(ratios)]
    return torch.tensor(features, dtype=torch.float64).reshape(len(ratios), len(projection_keys))


def summarize_votes(points, bank, votes, members):
    steps = [point["step"] for point in points[1:]]
    watch = next((t for t, v in zip(steps, votes, strict=True) if v * 2 > members), None)
    confirmed = confirmation(points, steps, votes, members, bank["layers"], bank["phase_limit"])
    return dict(horizon=bank["horizon"], first_alert=watch, scored_steps=steps, votes=votes.tolist(),
                first_confirmation=confirmed)


def score_run(points, bank):
    """Score a single observed prefix with its signed projections and L2 magnitudes."""
    z = direction_matrix(points, bank)
    votes = np.zeros(len(z), dtype=int)
    for member in bank["members"]:
        scores = ((z - z.new_tensor(member["center"])) / z.new_tensor(member["scale"])) @ z.new_tensor(member["weights"])
        votes += scores.numpy() > member["cutoff"]
    return summarize_votes(points, bank, votes, len(bank["members"]))
