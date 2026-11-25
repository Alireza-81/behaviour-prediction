"""Seal source-only forecasts from activation files, then open outcome reports."""
import argparse
import json
from pathlib import Path

from behaviour_prediction.monitoring import invariant_trajectory, source_only, trajectory_forecast
from behaviour_prediction.olmo_pretraining.protocol import digest, file_digest, read_json, write_json
from run_monitor_transfer import classify, infer_onset, specifications
from stage_monitor_study import CONFIG
from sweep_activation_monitors import summarize


def read_hashed(path):
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def scoring_steps(steps, bank):
    horizon = bank["full"]["horizon"]
    if any(policy["horizon"] != horizon for policy in bank["by_corpus"].values()):
        raise ValueError("Frozen policies use different forecast horizons")
    return [step for step in steps if 0 < step <= steps[-1] - horizon]


def forecast_run(name, corpus, seed, arm, points, bank):
    policies = {"all_source_corpora": bank["full"], "held_out_corpus": bank["by_corpus"][corpus]}
    scores = {label: source_only.score_prefix(points, policy) for label, policy in policies.items()}
    forecast_steps = {point["step"] for point in points[1:]}
    for score in scores.values():
        if (score["horizon"] != bank["full"]["horizon"] or not set(score["scored_steps"]) <= forecast_steps
                or (score["first_alert"] is not None and score["first_alert"] not in score["scored_steps"])):
            raise ValueError(f"Scorer returned a prediction outside the admissible window for {name}")
    return dict(family=corpus, seed=seed, arm=arm, policies=scores)


def seal(study, bank_path):
    """Read activation trajectories and capture metadata without opening behavior outcomes."""
    design, design_sha = read_hashed(study / "design.json")
    spec, spec_sha = read_hashed(study / "capture-spec.json")
    bank, bank_sha = read_hashed(bank_path)
    if spec["design_sha256"] != design_sha:
        raise ValueError("Capture specification belongs to a different design")
    models = [model for model in design["models"].values() if model["revision"] == spec["model_revision"]]
    if len(models) != 1:
        raise ValueError("Capture revision does not identify one target model")
    steps = design["steps"]
    forecast_steps = scoring_steps(steps, bank)
    planned = specifications(design)
    if len(forecast_steps) == 0 or not planned:
        raise ValueError("The target plan has no forecasts")
    if set(spec["adapter_sha256"]) != {name for name, *_ in planned}:
        raise ValueError("Capture specification does not list the planned runs")

    predictions, signal_hashes = {}, {}
    for name, corpus, seed, arm in planned:
        step_hashes, points = {}, []
        if set(spec["adapter_sha256"][name]) != set(map(str, steps[1:])):
            raise ValueError(f"Adapter checkpoint schedule differs for {name}")
        for step in forecast_steps:
            relative = f"signals/{name}/{step:06d}.json"
            point, sha = read_hashed(study / relative)
            if point["step"] != step:
                raise ValueError(f"Signal checkpoint differs from the plan for {name}/{step}")
            step_hashes[relative] = sha
            points.append(dict(step=step, signals=point["signals"]))
        # Captured distances and projections are relative to the target base model.
        points.insert(0, dict(step=0, signals=dict.fromkeys(points[0]["signals"], 0.0)))
        predictions[name] = forecast_run(name, corpus, seed, arm, points, bank)
        signal_hashes.update(step_hashes)
    return dict(format="source-only-activation-seal-v1", bank_sha256=bank_sha,
                scorer_sha256={Path(module.__file__).name: file_digest(Path(module.__file__))
                               for module in (source_only, invariant_trajectory, trajectory_forecast)},
                design_sha256=design_sha,
                capture_spec_sha256=spec_sha, signal_sha256=signal_hashes, model=models[0],
                judge_identity=bank["judge_identity"], steps=steps, runs=predictions)


def evaluate_outcomes(predictions, report, min_rise):
    steps = report["steps"]
    runs = {}
    outcomes = {"ordinary": "harmful_bounds", "conservative": "conservative_harmful_bounds"}
    for name, prediction in predictions.items():
        row = report["runs"][name]
        identity = (row["family"], row["seed"], row["variant"])
        if identity != (prediction["family"], prediction["seed"], prediction["arm"]):
            raise ValueError(f"Outcome report has the wrong run metadata for {name}")
        decisions = {}
        for label, field in outcomes.items():
            bounds = row[field]
            if len(bounds) != len(steps) or not all(0 <= lo <= hi <= report["panel_size"] for lo, hi in bounds):
                raise ValueError(f"Invalid outcome bounds for {name}")
            onset, possible = infer_onset(steps, bounds, min_rise)
            decisions[label] = {
                policy: classify(score, onset, possible) for policy, score in prediction["policies"].items()
            }
        runs[name] = dict(family=row["family"], seed=row["seed"], arm=row["variant"], decisions=decisions)

    summary = {
        label: {
            policy: {
                arm: summarize({name: row["decisions"][label][policy] for name, row in runs.items()
                                if arm == "all" or row["arm"] == arm})
                for arm in ("all", *sorted({row["arm"] for row in runs.values()}))
            }
            for policy in ("all_source_corpora", "held_out_corpus")
        }
        for label in outcomes
    }
    return dict(format="source-only-transfer-report-v1", steps=steps, summary=summary, runs=runs)


def evaluate(study, bank_path, seal_path):
    prospective, seal_sha = read_hashed(seal_path)
    if prospective != seal(study, bank_path):
        raise ValueError("The forecast seal differs from the current bank, signals, or plan")
    design = read_json(study / "design.json")
    report, report_sha = read_hashed(study / "report.json")
    manifest = read_json(study / "capture-haiku/manifest.json")
    if [manifest["model"], digest(manifest["rubric"].encode())] != prospective["judge_identity"]:
        raise ValueError("Target judge model or rubric differs from the source protocol")
    steps = prospective["steps"]
    if (report["steps"] != steps or report["frozen_sha256"] != design["frozen_sha256"]
            or set(report["runs"]) != set(prospective["runs"])):
        raise ValueError("Outcome report differs from the sealed cohort or schedule")
    expected_ratings = {"branch-base/review-000000.csv"} | {
        f"branch-{name}/review-{step:06d}.csv" for name in report["runs"] for step in steps[1:]
    }
    if expected_ratings != set(manifest["source_hashes"]):
        raise ValueError("Judged responses differ from the full planned cohort")
    result = evaluate_outcomes(prospective["runs"], report, design["minimum_harm_rise"])
    return dict(result, seal_sha256=seal_sha, outcome_report_sha256=report_sha, model=prospective["model"])


def replay_archive(archive_path, report_path, bank_path):
    """Retrospective comparison on a historical numeric advice export."""
    archive, archive_sha = read_hashed(archive_path)
    report, report_sha = read_hashed(report_path)
    bank, bank_sha = read_hashed(bank_path)
    protocol = read_json(CONFIG / "protocol.json")
    if (archive["reference_report_sha256"] != report_sha or archive["judge_identity"] != bank["judge_identity"]
            or report["steps"] != protocol["capture_steps"] or set(archive["runs"]) != set(report["runs"])):
        raise ValueError("Archived signals, outcomes or judge identity differ from the recorded study")
    admissible = {0, *scoring_steps(report["steps"], bank)}
    predictions = {}
    for name, points in archive["runs"].items():
        if [point["step"] for point in points] != report["steps"]:
            raise ValueError(f"Archive checkpoint schedule differs for {name}")
        prefix = [dict(step=p["step"], signals=p["signals"]) for p in points if p["step"] in admissible]
        row = report["runs"][name]
        predictions[name] = forecast_run(name, row["family"], row["seed"], row["variant"], prefix, bank)
    result = evaluate_outcomes(predictions, report, protocol["minimum_harm_rise"])
    return dict(result, analysis="historical_replay", bank_sha256=bank_sha,
                archive_sha256=archive_sha, outcome_report_sha256=report_sha, model=archive["model"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("seal", "evaluate"):
        command = stages.add_parser(stage)
        command.add_argument("--study-dir", type=Path, required=True)
        command.add_argument("--bank", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if stage == "evaluate":
            command.add_argument("--seal", type=Path, required=True)
    archive = stages.add_parser("replay-archive")
    for argument in ("--archive", "--report", "--bank", "--output"):
        archive.add_argument(argument, type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "seal":
        result = seal(args.study_dir, args.bank)
    elif args.stage == "evaluate":
        result = evaluate(args.study_dir, args.bank, args.seal)
    else:
        result = replay_archive(args.archive, args.report, args.bank)
    write_json(args.output, result)
    print(f"{len(result['runs'])} runs written to {args.output}")


if __name__ == "__main__":
    main()
