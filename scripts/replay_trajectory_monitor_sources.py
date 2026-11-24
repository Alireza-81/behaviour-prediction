"""Change only the fitting architecture of the frozen two-branch monitor."""
import argparse
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.relative_direction import direction_matrix
from behaviour_prediction.monitoring.trajectory_forecast import evidence, score_prefix, smooth
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_single_source_trajectories import CONFIG, NEVER, ROOTS, Cohort, classify, counts, fit, serial

ARCHITECTURES = ("llama", *ROOTS)
BANK = CONFIG / "llama-trajectory-v2.json"
BRANCHES = {"combined": "first_alert", "directions": "direction_alert", "geometry": "trajectory_alert"}


def refit(data, template, matrices, source, holdout, seeds):
    corpora = tuple(c for c in data.corpora if c != holdout)
    source_rows = (data.arch == source) & np.isin(data.corpus, corpora)
    if seeds:
        source_rows &= np.array([int(name.split("/")[2]) in seeds for name in data.names])
    clean = np.array([o["earliest_possible"] is None for o in data.outcomes])
    geom, direction = matrices
    geometry_members, direction_members = [], []
    for corpus in corpora:
        training = np.flatnonzero(source_rows & (data.corpus != corpus))
        controls = source_rows & clean & (data.corpus == corpus)
        if np.unique(data.labels[training][data.labels[training] >= 0]).size != 2 or not controls.any():
            raise ValueError(f"Missing eligible source examples for {source}, {holdout}, {corpus}")
        model = fit("ridge", geom, training, data)
        scores = smooth(evidence(model, geom[controls]), template["smoothing"])
        peak = scores.max()
        typical = np.median(scores[np.isfinite(scores)])
        geometry_members.append(dict(serial(model), calibration=corpus,
                                     cutoff=float(peak + template["margin"] * (peak - typical))))

        model = fit("ridge", direction, training, data)
        direction_members.append(dict(center=model["center"].tolist(), scale=model["scale"].tolist(),
                                      weights=model["coef"].tolist(), corpus_for_calibration=corpus,
                                      cutoff=float(evidence(model, direction[controls]).max())))
    direction_bank = dict(template["direction_bank"], members=direction_members,
                          source_architectures=[source], source_corpora=corpora)
    return dict(template, source=source, direction_bank=direction_bank, members=geometry_members)


def replay(data, template, matrices, seeds, out):
    result = {}
    for source in ARCHITECTURES:
        full = refit(data, template, matrices, source, None, seeds)
        held = {c: refit(data, template, matrices, source, c, seeds) for c in data.corpora}
        write_json(out / f"{source}-bank.json", full)
        result[source] = {}
        for scope in ("all_source_corpora", "test_corpus_excluded"):
            alarms = {key: np.full(len(data.names), NEVER) for key in BRANCHES}
            per_run = {}
            for i, run in enumerate(data.names):
                if scope == "all_source_corpora" and data.arch[i] == source:
                    continue
                policy = full if scope == "all_source_corpora" else held[data.corpus[i]]
                pred = score_prefix(data.points[i], policy)
                for branch, field in BRANCHES.items():
                    alarms[branch][i] = pred[field] if pred[field] is not None else NEVER
                if source == "llama" and not seeds and scope == "all_source_corpora":
                    previous = score_prefix(data.points[i], template)
                    if any(previous[field] != pred[field] for field in BRANCHES.values()):
                        raise AssertionError(f"Refit changed the published Llama bank decisions for {run}")
                per_run[run] = dict(onset=data.outcomes[i]["onset"], **{field: pred[field] for field in BRANCHES.values()})
            status = classify(alarms["combined"], data.outcomes)
            for i, run in enumerate(data.names):
                if run in per_run:
                    per_run[run]["status"] = status[i]
            profiles = {}
            for target in ARCHITECTURES:
                if scope == "all_source_corpora" and target == source:
                    continue
                mask = data.arch == target
                profiles[target] = dict(branches={key: counts(times, data.outcomes, mask) for key, times in alarms.items()},
                    conservative=counts(alarms["combined"], data.strict, mask) if target != "llama" else None)
            result[source][scope] = dict(targets=profiles, runs=per_run)
        print(f"Completed {source} refit and evaluation", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/trajectory-monitor-source-rotation"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = Cohort()
    template = read_json(BANK)
    matrices = (data.matrix(template["columns"]),
                np.stack([direction_matrix(p, template["direction_bank"]).numpy() for p in data.points]))
    results = replay(data, template, matrices, (), args.output / "banks")
    matched = replay(data, template, matrices, (101, 202), args.output / "banks-seeds-101-202")
    write_json(args.output / "report.json", dict(template_sha256=file_digest(BANK), archive=data.identity,
               all_available_source_seeds=results, source_seeds_101_202=matched))


if __name__ == "__main__":
    main()
