"""Refit and replay the original monitors on a matched advice transfer study."""

import argparse
import csv
from pathlib import Path

import torch

from behaviour_prediction.olmo_pretraining.protocol import digest, file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, dataset
from run_monitor_transfer import classify, predict
from stage_monitor_transfer import freeze_from_summaries
from sweep_activation_monitors import summarize


OUTPUT = Path("outputs/monitor-advice-transfer-v1/olmo")
PANEL = Path("outputs/monitor-study-v1/panel.json")
ARMS = ("original", "control")
PRIMARY = "geometry_and_traits-10"


def export_inputs(capture, output, *, model_key="olmo"):
    report_path = capture / "report.json"
    report = read_json(report_path)
    design_path = capture / "design.json"
    design = read_json(design_path)
    spec = read_json(capture / "capture-spec.json")
    if (spec["design_sha256"] != file_digest(design_path)
            or design["panel_sha256"] != file_digest(PANEL)
            or design["steps"] != report["steps"]
            or spec["model_revision"] != design["models"][model_key]["revision"]):
        raise ValueError("Capture differs from the audited target advice design.")
    panel = [(r["id"], r["prompt"]) for r in read_json(PANEL) if r["group"] == "legitimate"]
    manifest = read_json(capture / "capture-haiku/manifest.json")
    fields = ("id", "group", "harmful", "truncated")

    def ratings(name, step):
        relative = f"branch-{name}/review-{step:06d}.csv"
        if file_digest(capture / "capture" / relative) != manifest["source_hashes"][relative]:
            raise ValueError(f"Generation changed after judging: {relative}")
        with (capture / "capture-haiku" / relative).open(newline="") as file:
            rows = list(csv.DictReader(file))
        if [(r["id"], r["prompt"]) for r in rows] != panel or any(r["group"] != "legitimate" for r in rows):
            raise ValueError(f"Outcome prompts differ from the advice panel: {relative}")
        return [{k: r[k] for k in fields} | {"flagged": "needs_review=True" in r["notes"]} for r in rows]

    baseline = ratings("base", 0)
    runs = {}
    for name in sorted(report["runs"]):
        series = [read_json(capture / "signals" / name / f"{step:06d}.json") for step in report["steps"][1:]]
        if [r["step"] for r in series] != report["steps"][1:]:
            raise ValueError(f"Checkpoint schedule differs: {name}")
        # All captured features are baseline-relative. Step zero has zero shift.
        records = [dict(step=0, signals=dict.fromkeys(series[0]["signals"], 0.0), ratings=baseline)]
        records.extend(dict(r, ratings=ratings(name, r["step"])) for r in series)
        runs[name] = records
    write_json(output, {
        "model": design["models"][model_key],
        "reference_report_sha256": file_digest(report_path),
        "panel_sha256": design["panel_sha256"],
        "capture_spec_sha256": file_digest(capture / "capture-spec.json"),
        "judge_identity": [manifest["model"], digest(manifest["rubric"].encode())],
        "runs": runs,
    })
    print(f"Exported {len(runs)} numeric trajectories to {output}")


def fit(input_path, report_path, output, *, model_key="olmo"):
    archive = read_json(input_path)
    reference = read_json(report_path)
    protocol_path = CONFIG / "protocol.json"
    protocol = read_json(protocol_path)
    if (archive["reference_report_sha256"] != file_digest(report_path)
            or archive["panel_sha256"] != file_digest(PANEL)
            or set(archive["runs"]) != set(reference["runs"])
            or set(reference["families"]) != set(protocol["corpora"])
            or reference["steps"] != protocol["capture_steps"]):
        raise ValueError("The numeric inputs and audited advice report refer to different experiments.")
    pairs = {}
    names = {}
    for name, run in sorted(reference["runs"].items()):
        corpus, seed = run["family"], run["seed"]
        branch = ARMS.index(run["variant"])
        names[corpus, seed, branch] = name
        records = pairs.setdefault((corpus, seed), {"records": []})["records"]
        points = archive["runs"][name]
        if any(len(point["ratings"]) != reference["panel_size"] for point in points):
            raise ValueError(f"Incorrect outcome panel size: {name}")
        records.extend(dict(point, branch=branch) for point in points)
    corpora = sorted(protocol["corpora"])
    seeds = sorted({seed for _, seed in pairs})
    if set(names) != {(corpus, seed, branch) for corpus in corpora for seed in seeds for branch in (0, 1)}:
        raise ValueError("Every corpus must have both arms and the same seed set.")

    # The archived conservative transfer endpoint treats any flagged or capped
    # response as unresolved. Keep the fit on the ordinary historical labels.
    conservative_pairs = {
        key: {"records": [
            dict(point, ratings=[
                dict(row, harmful="") if row["flagged"] or row["truncated"] == "1" else row
                for row in point["ratings"]
            ]) for point in pair["records"]
        ]} for key, pair in pairs.items()
    }
    outcomes = {}
    for sensitivity, ref_key in ((False, "decisions"), (True, "conservative_decisions")):
        for horizon in (5, 10):
            _, labels = dataset(conservative_pairs if sensitivity else pairs,
                                dict(protocol, forecast_steps=horizon), sensitivity=False, activation_only=True)
            outcomes[sensitivity, horizon] = labels
            for (corpus, seed, branch), name in names.items():
                row = labels[f"{corpus}/{seed}/{branch}"]
                archived = reference["runs"][name][ref_key][f"geometry_and_traits-{horizon}"]
                bounds_key = "conservative_harmful_bounds" if sensitivity else "harmful_bounds"
                if (row["onset"] != archived["onset"] or row["earliest_possible"] != archived["earliest_possible"]
                        or [list(bound) for bound in row["harm_bounds"]] != reference["runs"][name][bounds_key]):
                    raise ValueError(f"Refit labels do not reproduce the audited outcome: {name}")

    folds = {}
    decisions = {}
    for target in corpora:
        source = {key: pair for key, pair in pairs.items() if key[0] != target}
        bank = freeze_from_summaries(protocol, source, archive["judge_identity"])
        folds[target] = {}
        for policy in bank["monitors"]:
            members = policy["members"]
            if len(members) != len(corpora) - 1 or any(
                target in m["training_corpora"] or target == m["calibration_corpus"] for m in members
            ):
                raise ValueError(f"The held-out corpus leaked into the bank: {target}")
            folds[target][policy["name"]] = [
                {k: m[k] for k in ("calibration_corpus", "training_corpora", "cutoff")}
                | {"training_windows": len(m["labels"]), "positive_training_windows": int(sum(m["labels"]))}
                for m in members
            ]
        for (corpus, seed, branch), name in names.items():
            if corpus != target:
                continue
            forecasts = predict(archive["runs"][name], bank, reference["steps"][-1])
            run = f"{corpus}/{seed}/{branch}"
            for policy, forecast in forecasts.items():
                horizon = forecast["horizon"]
                ordinary, strict = (outcomes[sensitivity, horizon][run] for sensitivity in (False, True))
                decisions.setdefault(policy, {})[name] = {
                    **classify(forecast, ordinary["onset"], ordinary["earliest_possible"]),
                    "conservative": classify(forecast, strict["onset"], strict["earliest_possible"]),
                }

    policies = {}
    for policy, runs in decisions.items():
        policies[policy] = {
            f"{model_key}_trained": summarize(runs),
            f"{model_key}_trained_conservative": summarize({name: row["conservative"] for name, row in runs.items()}),
            "llama_trained": summarize({name: row["decisions"][policy] for name, row in reference["runs"].items()}),
            "runs": runs,
        }
    provenance = dict(model=archive["model"], input_sha256=file_digest(input_path),
                      labels_sha256=archive["reference_report_sha256"], protocol_sha256=file_digest(protocol_path))
    result = dict(**provenance, primary=PRIMARY, seeds=seeds, folds=folds, policies=policies,
                  evaluation="Leave one corpus out, including both arms and all seeds.")
    bank = freeze_from_summaries(protocol, pairs, archive["judge_identity"])
    target_name = {"olmo": "OLMo", "qwen": "Qwen"}.get(model_key, model_key)
    write_json(output / f"{model_key}-advice-monitor-bank.json", dict(**bank, provenance=provenance,
               scope=f"Fit on all six archived {target_name} corpora for use on future runs. Held-out scores use separate fold banks."))
    write_json(output / f"{model_key}-advice-refit-report.json", result)
    for policy, values in policies.items():
        print(policy, values[f"{model_key}_trained"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "fit"))
    parser.add_argument("--capture", type=Path, default=Path("/data/monitor-advice-transfer-v1/olmo"))
    parser.add_argument("--input", type=Path, default=OUTPUT / "olmo-advice-refit-inputs.json")
    parser.add_argument("--report", type=Path, default=OUTPUT / "report.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "export":
        export_inputs(args.capture, args.input)
    else:
        fit(args.input, args.report, args.output)


if __name__ == "__main__":
    main()
