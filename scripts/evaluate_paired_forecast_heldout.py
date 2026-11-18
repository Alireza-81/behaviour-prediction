"""Replay the fixed development paired forecasters on the other study corpora."""
import argparse
from pathlib import Path

from evaluate_monitor_study import CONFIG, OUT, dataset, digest, load, read_split, save_json
from fit_paired_activation_forecast import METHODS, online_rows, report_runs, score_runs, totals


def majority_vote(predictions):
    """Count which of three frozen models alert at each checkpoint."""
    votes = {}
    for model, cutoff in predictions:
        for run, trajectory in model.items():
            run_votes = votes.setdefault(run, {point["step"]: 0 for point in trajectory})
            for point in trajectory:
                run_votes[point["step"]] += point["score"] > cutoff
    return {run: [dict(step=step, score=count) for step, count in ordered.items()]
            for run, ordered in votes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, default=OUT / "report/paired-loss-development.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fit_path = args.fit_report
    protocol = load(CONFIG / "protocol.json")
    frozen = load(fit_path)
    for name, expected in frozen["source_sha256"].items():
        if digest((OUT / "summaries/development" / f"{name}.json").read_bytes()) != expected:
            raise ValueError(f"Development source differs from the fitted artifact: {name}")
    inputs = read_split(protocol, "heldout", set())
    rows = online_rows(inputs, protocol)
    horizon = protocol["forecast_steps"]
    results = {}
    for variant, fit in frozen["results"].items():
        _, outcomes = dataset(inputs, protocol, variant != "primary")
        folds = [fold for fold in fit["folds"] if fold["rotation"] == "main"]
        if len(folds) != 3:
            raise ValueError("Expected exactly three main-rotation development fits")
        methods = {}
        for name in METHODS:
            members, predictions = [], []
            for fold in folds:
                model = fold["models"][name]
                scores = score_runs(rows, model)
                cutoff = model["cutoff"]
                observed = report_runs(scores, outcomes, cutoff, horizon)
                members.append(dict(training=fold["training"], calibration=fold["calibration"],
                                    cutoff=cutoff, summary=totals(observed.values()), runs=observed))
                predictions.append((scores, cutoff))
            votes = majority_vote(predictions)
            majority = report_runs(votes, outcomes, 1, horizon)
            for run in majority.values():
                run["votes_by_step"] = [dict(step=p["step"], votes=p["score"])
                                        for p in run.pop("scores")]
            methods[name] = dict(members=members,
                                 consensus=dict(summary=totals(majority.values()), runs=majority))
            print(variant, name, [r["summary"] for r in members], "majority", totals(majority.values()))
        results[variant] = methods

    output_name = "paired-loss-heldout-replay.json"
    if "layers" in frozen:
        output_name = "paired-loss-heldout-replay-layers-" + "-".join(map(str, frozen["layers"])) + ".json"
    output = args.output or OUT / "report" / output_name
    save_json(output, dict(
        development_artifact_sha256=digest(fit_path.read_bytes()),
        status="Retrospective replay on already-inspected held-out corpora, excluded from all fits and cutoffs.",
        policy="Report each main-rotation fit and an equal-weight two-of-three checkpoint vote. First alert is timely only if it precedes onset by 1 to 10 steps.",
        last_scored_step=frozen["last_scored_step"],
        source_sha256={
            f"{corpus}-{seed}": digest((OUT / "summaries/heldout" / f"{corpus}-{seed}.json").read_bytes())
            for corpus, seed in inputs
        },
        results=results,
        **({"layers": frozen["layers"]} if "layers" in frozen else {}),
    ))


if __name__ == "__main__":
    main()
