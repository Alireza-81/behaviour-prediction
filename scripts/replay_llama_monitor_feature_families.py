"""Compare the activation families and layers used by the Llama-only ridge monitor."""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from finalize_cross_architecture_monitor import WATCH, confirmation
from search_cross_architecture_monitors import (
    AXES, HORIZON, LAYERS, OUTPUT, TARGETS, Archive, Experiment, column_catalog, load_cohorts, prefix_catalog, write_json,
)


FAMILIES = ("magnitude", "orientation", *AXES)
REPRESENTATIONS = ("within_run_normalized", "checkpoint_level")
STATISTICS = ("timely", "early", "late", "missed", "false_alert")


def nonempty_subsets(values):
    return [subset for n in range(1, len(values) + 1) for subset in combinations(values, n)]


def column(representation, layer, family):
    if family in ("magnitude", "orientation"):
        prefix = "relative_phase_" if representation == "within_run_normalized" else ""
        return f"{prefix}L{layer}_{family}"
    prefix = "relative_direction_" if representation == "within_run_normalized" else ""
    return f"{prefix}L{layer}_projection_{family}"


def designs(representation, *, add_to=()):
    groups = []
    for layers in nonempty_subsets(LAYERS):
        for families in nonempty_subsets(FAMILIES):
            columns = tuple(column(representation, layer, family) for family in families for layer in layers)
            groups.append(dict(design="family_subset", layers=layers, families=families, columns=columns))
    columns = groups[-1]["columns"]
    for extra in columns:
        if add_to and extra not in add_to:
            groups.append(dict(design="add_one_to_baseline", layers=(), families=(), columns=(*add_to, extra)))
    return columns, groups


def fold_scores(archive, matrix, training, calibration, target, groups):
    ids = archive.select(("llama",), training)
    y = archive.labels[ids]
    data, labels = matrix[ids][y >= 0], y[y >= 0]
    center, scale = data.mean(0), data.std(0, unbiased=False).clamp_min(1e-6)
    x = (data - center) / scale
    gram, rhs = x.T @ x / len(x), x.T @ (labels - labels.mean()) / len(x)
    reference = archive.select(("llama",), (calibration,))
    controls = reference[archive.clean[reference]]
    n_reference = len(controls) * len(archive.steps)
    z = (matrix[np.concatenate((controls, target))].flatten(end_dim=1) - center) / scale
    for result_ids, cols in groups:
        # Standardizing and slicing the full Gram matrix is equivalent to
        # fitting each ridge model separately on its chosen columns.
        matrix_subsets = gram[cols[:, :, None], cols[:, None, :]]
        weight = torch.linalg.solve(matrix_subsets + 0.5 * torch.eye(cols.shape[1], dtype=x.dtype), rhs[cols, None])
        scores = (z[:, cols].transpose(0, 1) @ weight).squeeze(-1)
        cutoff = scores[:, :n_reference].amax(1, keepdim=True)
        alerts = (scores[:, n_reference:] > cutoff).reshape(len(cols), len(target), len(archive.steps))
        yield result_ids, alerts.numpy()


def replay(archive, matrix, candidates, columns):
    column_ids = {name: i for i, name in enumerate(columns)}
    by_width = {}
    for i, candidate in enumerate(candidates):
        by_width.setdefault(len(candidate["columns"]), []).append((i, [column_ids[k] for k in candidate["columns"]]))
    batches = [(np.array(ids), torch.tensor(cols)) for ids, cols in (zip(*rows, strict=True) for rows in by_width.values())]
    votes = np.zeros((len(candidates), len(archive.ids), len(archive.steps)), dtype=np.uint8)
    for held_out in archive.corpora:
        targets = archive.select(tuple(TARGETS), (held_out,))
        donors = tuple(c for c in archive.corpora if c != held_out)
        for calibration in donors:
            training = tuple(c for c in donors if c != calibration)
            for ids, alerts in fold_scores(archive, matrix, training, calibration, targets, batches):
                votes[ids[:, None], targets] += alerts
    return votes


def summarize(archive, votes):
    steps = archive.steps
    watched = votes * 2 > len(archive.corpora) - 1
    phase_ready = np.zeros(watched.shape[1:], dtype=bool)
    for i in archive.select(tuple(TARGETS), archive.corpora):
        for j in range(2, len(steps)):
            solo_vote = np.zeros(len(steps), dtype=int)
            solo_vote[j] = 1
            phase_ready[i, j] = confirmation(archive.points[i][:len(steps) + 1], steps, solo_vote, 1, ("14", "15")) is not None
    persistent = np.pad(watched[..., :-1], ((0, 0), (0, 0), (1, 0)))
    confirmed = watched & (persistent | phase_ready)
    confirmed[..., :2] = False
    none = steps[-1] + HORIZON + 1
    onset = np.array([o["onset"] or none for o in archive.outcomes])
    result = {}
    for arch in TARGETS:
        ids = archive.select((arch,), archive.corpora)
        positive = onset[ids] != none
        for mode, alerts in (("warning", watched), ("confirmation", confirmed)):
            first = np.where(alerts[:, ids, :], steps, none).min(axis=-1)
            found = first != none
            lead = onset[ids] - first
            mask = (positive & found & (lead > 0) & (lead <= HORIZON), positive & found & (lead > HORIZON),
                    positive & found & (lead <= 0), positive & ~found, archive.clean[ids] & found)
            for name, values in zip(STATISTICS, mask, strict=True):
                result[f"{arch}_{mode}_{name}"] = values.sum(axis=1)
    return result


def verify_baseline(archive, votes):
    engine = Experiment(archive)
    for arch in TARGETS:
        for i, result in engine.evaluate(WATCH, ("llama",), arch).items():
            if votes[i].tolist() != result["votes"]:
                raise RuntimeError(f"Vectorized scores differ for {archive.ids[i]}")


def run(output):
    protocol, cohorts, strict, identity = load_cohorts()
    catalog = prefix_catalog(column_catalog())
    baseline = catalog[WATCH.features]
    specs = {}
    for representation in REPRESENTATIONS:
        columns, candidates = designs(representation, add_to=baseline if representation == REPRESENTATIONS[0] else ())
        catalog[representation] = columns
        specs[representation] = candidates
    archive = Archive(protocol, cohorts, strict, catalog)
    rows = []
    for representation, candidates in specs.items():
        votes = replay(archive, archive.matrix(representation), candidates, catalog[representation])
        if representation == REPRESENTATIONS[0]:
            baseline_id = next(i for i, r in enumerate(candidates) if r["columns"] == baseline)
            verify_baseline(archive, votes[baseline_id])
        metrics = summarize(archive, votes)
        for i, candidate in enumerate(candidates):
            rows.append(dict(representation=representation, **{
                key: ";".join(v) if isinstance(v, tuple) else v for key, v in candidate.items()
            }, **{key: int(value[i]) for key, value in metrics.items()}))
        print(f"Scored {len(candidates)} {representation} candidates", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "llama-feature-family-replay.csv"
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    write_json(output / "llama-feature-family-design.json", dict(
        archive=identity, families=FAMILIES, layers=LAYERS, representations=REPRESENTATIONS, candidates=len(rows),
        forecast_steps=HORIZON, baseline=baseline, confirmation_layers=["14", "15"],
        training="Llama only. Each tested corpus is held out of fitting and cutoff calibration.",
        normalized="Relative changes anchored at step 10 and scaled by early motion. Geometry uses log growth.",
        checkpoint_level="Direct captured magnitudes, orientation changes from base, and signed projections from base.",
    ))
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output)
