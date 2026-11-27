"""Audit the first possible warning step and replay alternative causal anchors."""

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from finalize_cross_architecture_monitor import BANK_FILENAME, WATCH, evaluate_result, score_run
from search_cross_architecture_monitors import (
    EPS, OUTPUT, TARGETS, Archive, Experiment, column_catalog, file_digest, load_cohorts, prefix_catalog, read_json,
    relative_direction, write_json,
)


def count_steps(runs, key):
    return dict(Counter("no_alert" if run[key] is None else str(run[key]) for run in runs))


def inspect_replay(archive, bank, reference):
    measurements = {}
    fraction = bank["interval"]
    for i in archive.select(tuple(TARGETS), archive.corpora):
        name = archive.ids[i]
        model = archive.domains[i]
        points = archive.points[i][:len(archive.steps) + 1]
        original = score_run(points, bank)
        if original["votes"] != reference[model]["watch"]["runs"][name]["votes"]:
            raise RuntimeError(f"Frozen scorer differs from archived votes for {name}")
        for n in range(1, len(points) + 1):
            prefix = score_run(points[:n], bank)
            if prefix["votes"] != original["votes"][:n - 1]:
                raise RuntimeError(f"Future checkpoint changed an earlier score for {name}, {points[n - 1]['step']}")
        original_alert = original["first_alert"]
        anchor = points[2]
        frozen = [*points[:3], *(dict(anchor, step=p["step"]) for p in points[3:])]
        shifted = [*points[:3], dict(anchor, step=anchor["step"] + fraction),
                   *(dict(p, step=p["step"] + fraction) for p in points[3:])]
        quiet = score_run(frozen, bank)
        delayed = score_run(shifted, bank)
        expected = original_alert + fraction if original_alert is not None else None
        if delayed["first_alert"] != expected or quiet["first_alert"] is not None:
            raise RuntimeError(f"Checkpoint intervention failed for {name}")
        measurements[name] = dict(onset=archive.outcomes[i]["onset"], first_alert=original_alert,
                                  first_confirmation=original["first_confirmation"],
                                  copied_anchor_alert=quiet["first_alert"], delayed_alert=delayed["first_alert"],
                                  early_votes=original["votes"][:2])
    return measurements


def anchor_features(archive, initial_columns):
    alternatives = {mode: tuple(f"{mode}_{k}" for k in initial_columns) for mode in ("step5_anchor", "rolling_anchor")}
    archive.catalog.update(alternatives)
    for points, rows in zip(archive.points, archive.features, strict=True):
        for j, row in enumerate(rows, start=1):
            for key, fixed, rolling in zip(initial_columns, *alternatives.values(), strict=True):
                signal = key.removeprefix("relative_direction_")
                magnitude = signal.partition("_projection_")[0] + "_magnitude"
                ratios = [point["signals"][signal] / max(point["signals"][magnitude], EPS)
                          for point in points[max(1, j - 2):j + 1]]
                first = points[1]["signals"][signal] / max(points[1]["signals"][magnitude], EPS)
                row[fixed] = relative_direction(ratios[-1], first, first)
                row[rolling] = relative_direction(ratios[-1], ratios[-2 if len(ratios) > 1 else -1], ratios[0])
    return alternatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    bank_path = args.output / BANK_FILENAME
    report_path = args.output / "locked-candidate-report.json"
    bank = read_json(bank_path)
    if bank["report_sha256"] != file_digest(report_path):
        raise ValueError("Bank points to a different replay report")
    protocol, cohorts, strict, identity = load_cohorts()
    catalog = prefix_catalog(column_catalog())
    if list(catalog[WATCH.features]) != bank["columns"]:
        raise ValueError("Features differ from the frozen monitor")
    archive = Archive(protocol, cohorts, strict, catalog)
    zero_inputs = int((archive.matrix(WATCH.features)[:, :2, :] == 0).sum().item())
    if zero_inputs != len(archive.ids) * 2 * len(bank["columns"]):
        raise RuntimeError("Early anchors are expected to yield exact zeros")
    replays = inspect_replay(archive, bank, read_json(report_path)["one_frozen_bank"])
    histograms = {arch: {kind: {key: count_steps((v for k, v in replays.items()
                                                 if k.startswith(f"{arch}/") and (v["onset"] is not None) == onset), key)
                               for key in ("first_alert", "delayed_alert", "copied_anchor_alert", "first_confirmation")}
                         for kind, onset in (("onset", True), ("control", False))} for arch in TARGETS}
    alternatives = anchor_features(archive, catalog[WATCH.features])
    engine = Experiment(archive)
    experiments = {}
    for mode in (WATCH.features, *alternatives):
        policy = replace(WATCH, features=mode)
        experiments[mode] = {}
        for arch in TARGETS:
            f = engine.evaluate(policy, ("llama",), arch)
            scored = evaluate_result(engine, f, 5)
            experiments[mode][arch] = dict(metrics=scored["metrics"], onset_warning_steps=count_steps(
                (r for r in scored["runs"].values() if r["onset"] is not None), "first_alert"), runs=scored["runs"])
    result = dict(identity=identity, bank_sha256=file_digest(bank_path), early_steps=archive.steps[:2].tolist(),
                  exact_zero_features=zero_inputs, frozen_replay=replays, histograms=histograms,
                  older_checkpoint_anchors=experiments, interventions={
                      "delayed_alert": "Duplicate step 10 at step 15, then delay observed later activations by 5 steps.",
                      "copied_anchor_alert": "Keep the activation measurements from step 10 unchanged at every later checkpoint.",
                  })
    path = args.output / "llama-monitor-timing-audit.json"
    write_json(path, result)
    print(f"Verified {len(replays)} frozen-bank runs and every truncated prefix. Saved {path}.")


if __name__ == "__main__":
    main()
