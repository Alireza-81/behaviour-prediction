"""Replay and export the checkpoint-normalized Llama warning monitor."""

import argparse
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from search_cross_architecture_monitors import (
    AXES, HORIZON, MODES, MID_LAYERS, OUTPUT, PRIORITY_AXES, TARGETS, Archive, Experiment, Policy,
    classify, column_catalog, file_digest, linear_weights, load_cohorts, prefix_catalog, read_json,
    summarize, write_json,
)


from behaviour_prediction.monitoring.relative_direction import PHASE_LIMIT, confirmation, score_run


WATCH = Policy("ridge", "reset_relative_mid", 1)
TWO_SOURCE_WATCH = Policy("knn", WATCH.features, 1)
BANK_FILENAME = "llama-relative-direction-bank.json"
COMPACT_BANK_FILENAME = "llama-three-projection-bank.json"
COMPACT_COLUMNS = (
    "relative_direction_L14_projection_honesty",
    "relative_direction_L15_projection_honesty",
    "relative_direction_L14_projection_request_contrast_candidate",
)
ORIGINAL = "geometry_and_traits-10"


def fit_bank(archive, references, policy=WATCH):
    sequence = archive.matrix(policy.features)
    columns = archive.catalog[policy.features]
    members = []
    for holdout in references:
        training = archive.select(("llama",), tuple(c for c in references if c != holdout))
        eligible = archive.labels[training] >= 0
        rows, labels = sequence[training][eligible], archive.labels[training][eligible]
        center = rows.mean(0)
        scale = rows.std(0, unbiased=False).clamp_min(1e-6)
        weights = linear_weights(policy.method, (rows - center) / scale, labels)
        calibration = archive.select(("llama",), (holdout,))
        controls = calibration[archive.clean[calibration]]
        cutoff = (((sequence[controls] - center) / scale) @ weights).max().item()
        members.append(dict(corpus_for_calibration=holdout, center=center.tolist(),
                            scale=scale.tolist(), weights=weights.tolist(), cutoff=cutoff))
    axes = tuple(dict.fromkeys(name.split("_projection_", 1)[1] for name in columns))
    return dict(version=1, candidate=policy.name, source_architectures=["llama"], source_corpora=references,
                horizon=HORIZON, interval=int(archive.steps[0]), axes=axes, layers=MID_LAYERS,
                columns=columns, phase_limit=PHASE_LIMIT, members=members)


def evaluate_result(engine, watches, members, *, use_confirmation=False):
    predictions = {}
    for i, result in watches.items():
        confirm = confirmation(engine.data.points[i][:len(result["votes"]) + 1], result["scored_steps"], result["votes"],
                               members, MID_LAYERS)
        first = confirm if use_confirmation else result["first_alert"]
        predictions[i] = dict(result, first_alert=first, first_confirmation=confirm)
    runs = engine.outcomes(predictions)
    positive_windows = negative_windows = positives_alerted = negatives_alerted = 0
    within_window = 0
    for i, result in predictions.items():
        labels = engine.data.labels[i].numpy()
        if use_confirmation:
            active = [(result["first_alert"] is not None and step >= result["first_alert"]) and vote * 2 > members
                      for step, vote in zip(result["scored_steps"], result["votes"], strict=True)]
        else:
            active = np.asarray(result["votes"]) * 2 > members
        positive_windows += int(np.sum(labels == 1))
        negative_windows += int(np.sum(labels == 0))
        positives_alerted += int(np.sum((labels == 1) & active))
        negatives_alerted += int(np.sum((labels == 0) & active))
        within_window += int(np.any((labels == 1) & active))
    return dict(metrics=summarize(runs), conservative=summarize(engine.outcomes(predictions, strict=True)), runs=runs,
                positive_windows=positive_windows, positive_windows_alerted=positives_alerted,
                negative_windows=negative_windows, negative_windows_alerted=negatives_alerted,
                onset_runs_with_alert_in_window=within_window)


def candidates(catalog):
    columns = catalog[WATCH.features]
    for count in range(1, len(PRIORITY_AXES)):
        for axes in combinations(PRIORITY_AXES, count):
            catalog[" + ".join(axes)] = tuple(k for k in columns if k.endswith(axes))
    for layer in (*MID_LAYERS, "31"):
        catalog[f"layer_{layer}"] = tuple(f"relative_direction_L{layer}_projection_{axis}" for axis in PRIORITY_AXES)
    catalog["all_axes_mid_relative"] = tuple(f"relative_direction_L{layer}_projection_{axis}"
                                            for axis in AXES for layer in MID_LAYERS)
    for name, prefix in (("raw_projection_mid", ""), ("current_direction_mid", "direction_"),
                         ("anchored_direction_mid", "reset_direction_"), ("anchored_projection_mid", "reset_projection_")):
        catalog[name] = tuple(k.replace("relative_direction_", prefix, 1) for k in columns)
    return (WATCH.features, "reset_relative_all", "all_axes_mid_relative", "normalized_geometry",
            "normalized_mid_layers", "checkpoint_number", "layer_14", "layer_15", "layer_31",
            "raw_projection_mid", "current_direction_mid", "anchored_direction_mid", "anchored_projection_mid",
            *(" + ".join(axes) for count in (1, 2) for axes in combinations(PRIORITY_AXES, count)))


def archived_mixture(bank):
    root = Path("outputs/mixed-advice-monitor-v1")
    path, report_path = root / "gpu-mixture-activation-data.json", root / "gpu-report.json"
    if not path.is_file() or not report_path.is_file():
        return None
    archive = read_json(path)
    if archive["report_sha256"] != file_digest(report_path):
        raise ValueError("Finance mixture activation archive identifies a different outcome report")
    report = read_json(report_path)["runs"]
    results, unavailable = {}, []
    for run, points in archive["runs"].items():
        if points[1]["step"] != bank["interval"]:
            unavailable.append(run)
            continue
        old = report[run]["forecasts"]["full"]["geometry_and_traits-10"]
        steps = [p for p in points if p["step"] <= points[-1]["step"] - bank["horizon"]]
        new = score_run(steps, bank)
        label = old["onset"], old["earliest_possible"]
        results[run] = dict(old=old, watch=classify(new, *label),
                            confirm=classify(dict(new, first_alert=new["first_confirmation"]), *label))
    return dict(activation_sha256=file_digest(path), report_sha256=file_digest(report_path), runs=results,
                unavailable_missing_step_five=unavailable)


def replay_feature_subsets(output):
    protocol, cohorts, conservative, identity = load_cohorts()
    catalog = prefix_catalog(column_catalog())
    full = catalog[WATCH.features]
    subsets = {" + ".join(columns): columns for count in range(1, len(full) + 1)
               for columns in combinations(full, count)}
    archive = Archive(protocol, cohorts, conservative, catalog | subsets)
    engine = Experiment(archive)
    results = []
    for name, columns in subsets.items():
        policy = replace(WATCH, features=name)
        models = {}
        for arch in TARGETS:
            votes = engine.evaluate(policy, ("llama",), arch)
            models[arch] = {}
            for mode in ("watch", "confirmation"):
                result = evaluate_result(engine, votes, len(archive.corpora) - 1, use_confirmation=mode == "confirmation")
                models[arch][mode] = {key: result[key] for key in ("metrics", "conservative", "onset_runs_with_alert_in_window")}
        results.append(dict(columns=columns, models=models))
    path = output / "llama-direction-subsets.json"
    write_json(path, dict(identity=identity, source_architectures=["llama"], full_features=full, horizon=HORIZON,
                          confirmation_layers=MID_LAYERS, phase_limit=PHASE_LIMIT, subsets=results,
                          evaluation="Fit and calibrate each subset on Llama with the evaluated corpus held out. "
                                     "For confirmation, activation magnitudes from layers 14 and 15 stay fixed."))
    print(f"Replayed {len(results)} feature subsets. Saved {path}.")


def export_compact(output):
    protocol, cohorts, conservative, identity = load_cohorts()
    policy = replace(WATCH, features="three_projections")
    archive = Archive(protocol, cohorts, conservative, {policy.features: COMPACT_COLUMNS})
    engine = Experiment(archive)
    bank = fit_bank(archive, archive.corpora, policy)
    folds = {corpus: fit_bank(archive, tuple(c for c in archive.corpora if c != corpus), policy)
             for corpus in archive.corpora}
    report = dict(identity=identity, candidate=policy.name, corpus_held_out={}, one_frozen_bank={})
    checks = 0
    for arch in TARGETS:
        held_out, pooled = {}, {}
        for i, replay in engine.evaluate(policy, ("llama",), arch).items():
            points = archive.points[i][:len(replay["votes"]) + 1]
            fold = folds[archive.families[i]]
            held_out[i] = score_run(points, fold)
            pooled[i] = score_run(points, bank)
            for n in range(1, len(points) + 1):
                prefix = score_run(points[:n], fold)
                if prefix["votes"] != replay["votes"][:n - 1]:
                    raise RuntimeError(f"Compact scoring differs from batch replay on {archive.ids[i]}")
                checks += 1
        for mode, predictions, members in (("corpus_held_out", held_out, len(archive.corpora) - 1),
                                           ("one_frozen_bank", pooled, len(bank["members"]))):
            report[mode][arch] = {name: evaluate_result(engine, predictions, members, use_confirmation=confirm)
                                 for name, confirm in (("warning", False), ("confirmation", True))}
    report["mixtures"] = archived_mixture(folds["finance"])
    report["prefix_replays_verified"] = checks
    report_path = output / "llama-three-projection-report.json"
    write_json(report_path, report)
    write_json(output / COMPACT_BANK_FILENAME, dict(bank, report_sha256=file_digest(report_path),
        source_identity=identity, selection="Retrospective architecture-transfer results. All coefficients and cutoffs use Llama."))
    print(f"Verified {checks} compact scorer prefixes and saved {output / COMPACT_BANK_FILENAME}")


def run(output):
    protocol, cohorts, conservative, identity = load_cohorts()
    catalog = column_catalog()
    catalog.update(prefix_catalog(catalog))
    ablation_features = candidates(catalog)
    archive = Archive(protocol, cohorts, conservative, catalog)
    engine = Experiment(archive)
    results = dict(identity=identity, horizon=HORIZON, phase_limit=PHASE_LIMIT, watch=WATCH.name,
                   two_source_watch=TWO_SOURCE_WATCH.name, reference=ORIGINAL)
    for mode, targets in MODES.items():
        policy = WATCH if mode == "llama_only" else TWO_SOURCE_WATCH
        out = results[mode] = {}
        for target, sources in targets.items():
            w = engine.evaluate(policy, sources, target)
            out[target] = dict(source_architectures=sources, watch=evaluate_result(engine, w, 5),
                               confirm=evaluate_result(engine, w, 5, use_confirmation=True))
    results["reference_metrics"] = {}
    for model, root in TARGETS.items():
        ref = read_json(root / "report.json")["runs"]
        results["reference_metrics"][model] = summarize({key: value["decisions"][ORIGINAL] for key, value in ref.items()})
    results["ablations"] = {}
    for name in ablation_features:
        p = Policy(WATCH.method, name, 1)
        results["ablations"][name] = {
            arch: summarize(engine.outcomes(engine.evaluate(p, ("llama",), arch))) for arch in TARGETS}
    bank = fit_bank(archive, archive.corpora)
    results["one_frozen_bank"] = {}
    checks = 0
    for target in TARGETS:
        all_source_decisions = {}
        for corpus in archive.corpora:
            reference = tuple(c for c in archive.corpora if c != corpus)
            fold = fit_bank(archive, reference)
            replay = engine.forecast(WATCH, ("llama",), target, corpus, reference)
            for i, expected in replay.items():
                points = [dict(step=p["step"], signals={k: v for k, v in p["signals"].items()
                          if k.startswith(("L14_", "L15_"))}) for p in archive.points[i] if p["step"] <= archive.steps[-1]]
                full = score_run(points, fold)
                for size in range(1, len(points) + 1):
                    prefix = score_run(points[:size], fold)
                    if prefix["votes"] != expected["votes"][:size - 1]:
                        raise RuntimeError(f"Replay differs at {archive.ids[i]}, step {points[size - 1]['step']}")
                    expected_confirm = full["first_confirmation"]
                    if expected_confirm is not None and expected_confirm > points[size - 1]["step"]:
                        expected_confirm = None
                    if prefix["first_confirmation"] != expected_confirm:
                        raise RuntimeError(f"Confirmation changes with future data on {archive.ids[i]}")
                    checks += 1
                all_source_decisions[i] = score_run(points, bank)
        results["one_frozen_bank"][target] = dict(
            watch=evaluate_result(engine, all_source_decisions, len(bank["members"])),
            confirm=evaluate_result(engine, all_source_decisions, len(bank["members"]), use_confirmation=True),
        )
    results["mixtures"] = archived_mixture(fit_bank(archive, tuple(c for c in archive.corpora if c != "finance")))
    results["prefix_replays_verified"] = checks
    write_json(output / "locked-candidate-report.json", results)
    bank.update(report_sha256=file_digest(output / "locked-candidate-report.json"), source_identity=identity,
                scope="Uses all six source corpora. The archive's held-out-corpus scores use separate banks.",
                selection="Selected on the retrospective OLMo and Qwen archive. Weights and cutoffs use Llama only.")
    write_json(output / BANK_FILENAME, bank)
    print(f"Verified {checks} prefixes against the batch replay. Wrote reports and a scoreable Llama bank to {output}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bank", type=Path, help="Fitted monitor JSON to use with --score-run")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--score-run", type=Path, help="JSON list of observed activation checkpoints for one run")
    actions.add_argument("--feature-subsets", action="store_true", help="Replay all nonempty subsets of the six ridge inputs")
    actions.add_argument("--export-compact", action="store_true", help="Export the Llama monitor with three projections")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.score_run is not None:
        print(json.dumps(score_run(read_json(args.score_run), read_json(args.bank or args.output / BANK_FILENAME)), indent=2))
    elif args.feature_subsets:
        replay_feature_subsets(args.output)
    elif args.export_compact:
        export_compact(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
