"""Measure matched prompt contrasts in the same checkpoint and frozen trait basis."""
import re
from collections import Counter, defaultdict

import torch

from .core import geometry

ROLES = ("safe", "pressure", "paraphrase")


def probe_panel(config):
    """Render the panel with stable IDs and explicit left-minus-right contrasts."""
    if config["version"] != 2:
        raise ValueError("Unsupported trait probe panel version")
    prompts, contrasts = {}, {}
    ids = set()
    groups = defaultdict(Counter)
    for record in config["tuples"]:
        key, split, behavior = (record[k] for k in ("id", "split", "behavior"))
        if key in ids or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
            raise ValueError(f"Invalid or duplicate tuple ID: {key}")
        if split not in ("development", "heldout"):
            raise ValueError(f"Unexpected scenario split for {behavior}")
        ids.add(key)
        groups[behavior][split] += 1
        variants = [record[k].strip() for k in ROLES]
        context = record["context"].strip()
        if not context or any(not v for v in variants) or len(set(variants)) != len(variants):
            raise ValueError(f"Expected one context and three distinct instructions: {key}")
        role_ids = {role: f"{key}/{role}" for role in ROLES}
        prompts.update({role_ids[role]: f"{context}\n\n{variant}"
                        for role, variant in zip(ROLES, variants, strict=True)})
        for kind, left in (("behavior", "pressure"), ("wording", "paraphrase")):
            contrasts[f"{key}/{kind}"] = dict(
                behavior=behavior, split=split, axis=record["axis"], kind=kind,
                left=role_ids[left], right=role_ids["safe"], scale=role_ids["pressure"],
            )
    if not groups or any(counts != {"development": 9, "heldout": 6} for counts in groups.values()):
        raise ValueError("Expected nine development and six held-out triplets per behavior")
    return prompts, contrasts


def legacy_panel(legacy):
    """Keep the historically monitored pairs and discard the newly authored scenarios."""
    prompts = {}
    pairs = defaultdict(dict)
    for r in legacy:
        if r["split"] == "monitor":
            if r["id"] in prompts or r["group"] in pairs[r["pair"]]:
                raise ValueError("Duplicate legacy monitoring role or ID")
            prompts[r["id"]] = r["prompt"]
            pairs[r["pair"]][r["group"]] = r["id"]
    if not pairs or any(set(p) != {"risky_request", "matched_benign"} for p in pairs.values()):
        raise ValueError("Expected matched historical monitoring pairs")
    definitions = {
        f"legacy/{key}": dict(behavior="legacy_request", split="legacy", axis="request_contrast_candidate",
                            kind="behavior", left=p["risky_request"], right=p["matched_benign"],
                            scale=p["risky_request"])
        for key, p in sorted(pairs.items())
    }
    return prompts, definitions


def vector_cosine(a, b):
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    cosine = (a * b).sum(dim=-1) / denominator.clamp_min(1e-12)
    return [v if d > 1e-12 else None for v, d in zip(cosine.clamp(-1, 1).tolist(), denominator.tolist(), strict=True)]


def measure_checkpoint(current, baseline, probe_ids, basis, definitions, *, per_prompt=True):
    """Return baseline-relative pair drift and signed scores on every frozen axis."""
    index = {key: i for i, key in enumerate(probe_ids)}
    names = basis["names"]
    if set(current) != set(baseline) or set(current) != set(basis["directions"]) or len(set(names)) != len(names):
        raise ValueError("Direction and activation layer or axis identities differ")
    if any(spec["axis"] not in names for spec in definitions.values()):
        raise ValueError("An intended historical axis is missing")
    measurements = {}
    left, right = ([index[v[key]] for v in definitions.values()] for key in ("left", "right"))
    pressure = [index[v["scale"]] for v in definitions.values()]
    for layer, h in current.items():
        h = h.to(dtype=torch.float64)
        h0 = baseline[layer].to(dtype=torch.float64)
        directions = basis["directions"][layer].to(dtype=torch.float64)
        probe_metrics = geometry(h, h0, directions)
        axes = torch.nn.functional.normalize(directions, dim=1)
        gap, gap0 = h[left] - h[right], h0[left] - h0[right]
        delta = gap - gap0
        scale = (h0[pressure].norm(dim=1) + h0[right].norm(dim=1)) / 2
        projection = (gap @ axes.T) / scale[:, None]
        projection_shift = (delta @ axes.T) / scale[:, None]
        length = delta.norm(dim=1)
        unit = (delta @ axes.T) / length.clamp_min(1e-12)[:, None]
        contrast_rows = {}
        for i, (key, orientation) in enumerate(zip(definitions, vector_cosine(gap, gap0), strict=True)):
            contrast_rows[key] = dict(
                gap_length=float(gap[i].norm() / scale[i]), drift_norm=float(length[i] / scale[i]),
                orientation_change=None if orientation is None else 1 - orientation,
                projection=dict(zip(names, projection[i].tolist(), strict=True)),
                projection_shift=dict(zip(names, projection_shift[i].tolist(), strict=True)),
                drift_alignment=dict(zip(names, unit[i].clamp(-1, 1).tolist(), strict=True))
                if length[i] > 1e-12 else None,
            )
        row = {"contrasts": contrast_rows}
        if per_prompt:
            row["prompts"] = {
                key: dict(magnitude=float(probe_metrics["relative_l2"][i]),
                          orientation=float(1 - probe_metrics["cosine"][i]),
                          projection_shift=dict(zip(names, probe_metrics["relative_projection_delta"][i].tolist(), strict=True)))
                for i, key in enumerate(probe_ids)
            }
        measurements[layer] = row
    return measurements
