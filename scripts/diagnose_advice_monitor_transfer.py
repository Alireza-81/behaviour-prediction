"""Ablate the Llama-trained middle-layer logistic monitor and export its bank."""

import argparse
from pathlib import Path
from statistics import median

import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, read_split
from fit_paired_activation_forecast import fit, forecast_scores
from rank_monitor_traits import attribution, report_attribution
from run_monitor_transfer import classify, predict
from stage_monitor_transfer import freeze_from_summaries
from sweep_activation_monitors import Candidate, persistent_scores
from transfer_advice_early_warning import (
    OUTPUT, RAW_BASELINE, TARGET, Replay, ablation, evaluate, feature_groups, feature_row,
    feature_sets, scoped_layers, target_outcomes, target_pairs,
)


LAYERS = ("14", "15")
POLICY = Candidate("logistic", "normalized_mid_layers", 2)
HORIZON = 10


def evaluate_both(predictions, engine, outcomes):
    return {
        "llama": evaluate(predictions["llama"], engine.source_outcomes, HORIZON)[0],
        "olmo": evaluate(predictions["olmo"], outcomes["ordinary"], HORIZON)[0],
        "olmo_conservative": evaluate(predictions["olmo"], outcomes["conservative"], HORIZON)[0],
    }


def paired_discrimination(scores, outcomes):
    hits = ties = total = 0
    for run, row in scores.items():
        if not run.endswith("/0"):
            continue
        onset = outcomes[run]["onset"]
        control = scores[run[:-1] + "1"]
        for i, step in enumerate(row["scored_steps"]):
            if onset is None or not 0 < onset - step <= HORIZON:
                continue
            point_index = control["scored_steps"].index(step)
            hits += row["votes"][i] > control["votes"][point_index]
            ties += row["votes"][i] == control["votes"][point_index]
            total += 1
    return dict(higher_votes=hits, tied_votes=ties, windows=total)


def bank_for(engine, candidate, references):
    members = []
    for calibration in references:
        training = tuple(c for c in references if c != calibration)
        rows = [r for corpus in training for r in engine.fit[corpus]]
        learned = fit(rows, pair_weight=0.0, features=engine.sets[candidate.features])
        members.append(dict(model=learned, calibration_corpus=calibration, training_corpora=training,
                            cutoff=engine.cutoff(candidate, training, calibration)))
    return dict(horizon=engine.horizon, capture_interval=5, persistence=candidate.persistence, members=members,
                required_layers=LAYERS)


def score_bank(points, bank):
    """Apply the frozen bank to any observed checkpoint prefix from one run."""
    interval = bank["capture_interval"]
    if len(points) < 2 or [p["step"] for p in points] != list(range(0, len(points) * interval, interval)):
        raise ValueError("Expected step 0 followed by equally spaced, observed checkpoints")
    rows = [feature_row(points[0], points[1], previous, point, layers=tuple(bank["required_layers"]))
            for previous, point in zip(points, points[1:])]
    votes = [0] * len(rows)
    for member in bank["members"]:
        scores = forecast_scores(rows, member["model"])
        for i, score in enumerate(persistent_scores(scores, bank["persistence"])):
            votes[i] += score is not None and score > member["cutoff"]
    steps = [r["step"] for r in rows]
    first = next((step for step, count in zip(steps, votes, strict=True)
                  if count * 2 > len(bank["members"])), None)
    return dict(horizon=bank["horizon"], first_alert=first, votes=votes, scored_steps=steps)


def initial_drift(signals):
    return (signals["L14_magnitude"] + signals["L15_magnitude"]) / 2


def source_scale_band(engine, references):
    # Include both training arms. Padding the observed range makes the switch
    # insensitive to small scale differences between Llama source corpora.
    drift = [initial_drift(rows[0]) for corpus in references
             for rows in engine.online["llama"][corpus].values()]
    return [min(drift) / 2, 2 * max(drift)]


def route(drift, band):
    return "raw_knn" if band[0] <= drift <= band[1] else "normalized_logistic"


def score_routed(points, bank):
    if len(points) < 2 or [p["step"] for p in points] != list(range(0, len(points) * 5, 5)):
        raise ValueError("Expected step 0 followed by checkpoints every five steps")
    selected = route(initial_drift(points[1]["signals"]), bank["source_scale_band"])
    if selected == "raw_knn":
        raw = predict(points, bank[selected], points[-1]["step"] + HORIZON)
        return dict(raw["geometry_and_traits-10"], route=selected)
    return dict(score_bank(points, bank[selected]), route=selected)


def routing_experiment(engine, source, outcomes, protocol, signatures, output, full_bank):
    bands = {corp: source_scale_band(engine, engine.references[corp]) for corp in engine.corpora}
    all_scale_band = source_scale_band(engine, engine.corpora)
    predictions, routing = {}, {}
    for domain in ("llama", "olmo"):
        raw = engine.fixed(domain, RAW_BASELINE)
        fallback = engine.fixed(domain, POLICY)
        predictions[domain], routing[domain] = {}, {}
        for corpus, paths in engine.online[domain].items():
            for run, rows in paths.items():
                drift = initial_drift(rows[0])
                selected = route(drift, bands[corpus])
                predictions[domain][run] = raw[run] if selected == "raw_knn" else fallback[run]
                routing[domain][run] = dict(route=selected, initial_drift=drift)
    raw_bank = freeze_from_summaries(protocol, source, next(iter(signatures)))
    raw = [policy for policy in raw_bank["monitors"] if policy["name"] == "geometry_and_traits-10"]
    bank = dict(source_scale_band=all_scale_band, raw_knn={"monitors": raw}, normalized_logistic=full_bank,
                scope="Both branches fitted and calibrated on all six Llama corpora.",
                search_report_sha256=file_digest(output / "report-10.json"))
    write_json(output / "routed-llama-monitor-bank.json", bank)
    label_sets = {"llama": engine.source_outcomes, "olmo": outcomes["ordinary"]}
    per_run = {domain: evaluate(rows, label_sets[domain], HORIZON)[1] for domain, rows in predictions.items()}
    result = dict(
        **evaluate_both(predictions, engine, outcomes), runs=per_run, routes=routing, scale_band_by_corpus=bands,
        rule="At step 5, use mean relative L2 drift in layers 14 and 15. Use raw kNN within the Llama "
             "training range expanded by a factor of two in both directions, and normalized logistic elsewhere.",
        evaluation="Retrospective routing proposal developed after inspecting model-transfer results.",
        bank_sha256=file_digest(output / "routed-llama-monitor-bank.json"),
    )
    write_json(output / "routed-llama-monitor-report.json", result)
    print("Retrospective scale-routed monitor:", result["llama"], result["olmo"], flush=True)


def diagnostics(target, output):
    torch.set_num_threads(1)
    protocol = dict(read_json(CONFIG / "protocol.json"), forecast_steps=HORIZON)
    signatures = set()
    source = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    archive_path = target / "olmo-advice-refit-inputs.json"
    archive = read_json(archive_path)
    audited = read_json(target / "report.json")
    study = read_json(output / "report-10.json")
    design_path = output / "design.json"
    design = read_json(design_path)
    if (study["design_sha256"] != file_digest(design_path)
            or design["identity"]["olmo_features_sha256"] != file_digest(archive_path)
            or signatures != {tuple(archive["judge_identity"])}):
        raise ValueError("The diagnostic inputs differ from the frozen search.")
    groups, axes = feature_groups()
    catalog = feature_sets(groups, axes)
    target_data = target_pairs(archive, audited)
    engine = Replay(source, target_data, protocol, catalog)
    outcomes = target_outcomes(target_data, audited, protocol)
    selections = dict.fromkeys(engine.corpora, POLICY)
    by_group = {name: scoped_layers(columns, LAYERS) for name, columns in groups.items()}
    full = engine.sets[POLICY.features]
    assert sum(by_group.values(), ()) == full

    subsets = {}
    for mask in range(1 << len(by_group)):
        columns = sum((coords for i, coords in enumerate(by_group.values()) if mask & (1 << i)), ())
        predictions = ablation(engine, f"logistic_mid_subset_{mask}", columns, selections)
        subsets[mask] = evaluate_both(predictions, engine, outcomes)
    full_index = len(subsets) - 1
    assert all(subsets[full_index][domain] == study["fixed"][POLICY.name][domain] for domain in subsets[full_index])
    shapley = {
        domain: report_attribution(attribution({m: row[domain] for m, row in subsets.items()}, tuple(by_group)))
        for domain in subsets[0]
    }
    variations = {
        "without_recent_derivatives": tuple(k for k in full if not k.startswith(("rate_", "direction_rate_"))),
        "without_first_checkpoint_changes": tuple(k for k in full if not k.startswith(("growth_", "direction_change_", "shape_change_"))),
        "without_all_directed_projections": tuple(k for k in full if "_projection_" not in k),
        "raw_middle_layers": catalog["raw_mid_layers"],
    }
    geometry = variations["without_all_directed_projections"]
    for axis, columns in axes.items():
        block = scoped_layers(columns, LAYERS)
        variations[f"geometry_plus_{axis}"] = geometry + block
        variations[f"drop_{axis}"] = tuple(key for key in full if key not in block)
    for layer in ("14", "15", "31"):
        variations[f"only_layer_{layer}"] = scoped_layers(catalog["normalized_full"], (layer,))
    variant_results = {
        name: evaluate_both(ablation(engine, f"logistic_mid_{name}", columns, selections), engine, outcomes)
        for name, columns in variations.items()
    }

    baseline_step = 15
    measurement = ("L14_magnitude", "L15_magnitude", "L31_magnitude", "L31_orientation")
    control_shift = {
        domain: {key: median(row[key] for paths in cohorts.values() for run, rows in paths.items()
                             if run.endswith("/1") for row in rows if row["step"] == baseline_step)
                 for key in measurement}
        for domain, cohorts in engine.online.items()
    }
    tested = study["fixed"]
    development = tuple(k for k, role in protocol["corpora"].items() if role == "development")
    heldout = tuple(k for k, role in protocol["corpora"].items() if role == "heldout")

    def restrict(runs, cohorts):
        return {run: row for run, row in runs.items() if run.split("/")[0] in cohorts}

    cohort_counts = {
        candidate: {name: evaluate(restrict(values["runs"], corpora), outcomes["ordinary"], HORIZON)[0]
                    for name, corpora in (("development", development), ("heldout", heldout))}
        for candidate, values in tested.items()
    }
    eligible = [key for key, row in tested.items() if row["llama"]["false_alert"] == row["olmo"]["false_alert"] == 0]
    reliable = min(eligible, key=lambda key: (-tested[key]["olmo"]["timely"], -tested[key]["llama"]["timely"], key))
    dev_winner = min(tested, key=lambda key: (cohort_counts[key]["development"]["false_alert"],
                        -cohort_counts[key]["development"]["timely"], -cohort_counts[key]["development"]["early"],
                        len(catalog[key.split("/")[1]]), key))

    pairs = {}
    for corpus in engine.corpora:
        fold = bank_for(engine, POLICY, engine.references[corpus])
        for (domain, by_corpus) in (("llama", source), ("olmo", target_data)):
            for (name, seed), record in by_corpus.items():
                if name != corpus:
                    continue
                for arm in (0, 1):
                    run = f"{name}/{seed}/{arm}"
                    points = [p for p in record["records"] if p["branch"] == arm and p["step"] <= 40]
                    expected = engine.predict(POLICY, domain, corpus, engine.references[corpus])[run]
                    # Prove that the portable version uses only the claimed layers.
                    prefixes = tuple(f"L{layer}_" for layer in LAYERS)
                    mids = [dict(point, signals={k: x for k, x in point["signals"].items() if k.startswith(prefixes)})
                            for point in points]
                    assert score_bank(mids, fold) == expected
                    assert score_bank(mids[:4], fold)["votes"] == expected["votes"][:3]
        pairs[corpus] = fold

    full_bank = bank_for(engine, POLICY, engine.corpora)
    write_json(output / "portable-llama-logistic-bank.json", dict(
        candidate=POLICY.name, **full_bank, search_report_sha256=file_digest(output / "report-10.json"),
        input="Observed layer 14 and 15 activation summaries, measured against this run's step-0 base.",
        scope="All six Llama source corpora. Candidate selected retrospectively; archive holdouts use separate banks.",
    ))
    mixture_path = Path("outputs/mixed-advice-monitor-v1/gpu-mixture-activation-data.json")
    mixed = {}
    if mixture_path.is_file():
        activation = read_json(mixture_path)
        mix_report_path = mixture_path.with_name("gpu-report.json")
        assert activation["report_sha256"] == file_digest(mix_report_path)
        mixtures = read_json(mix_report_path)["runs"]
        for run, points in activation["runs"].items():
            if run.startswith("finance"):
                observation = mixtures[run]["forecasts"]["full"]["geometry_and_traits-10"]
                raw = score_bank([p for p in points if p["step"] <= 40], pairs["finance"])
                mixed[run] = classify(raw, observation["onset"], observation["earliest_possible"])

    result = dict(
        candidate=POLICY.name, search_report_sha256=file_digest(output / "report-10.json"),
        ablation_groups=by_group, ablation_subsets=subsets, full=subsets[full_index], shapley=shapley,
        drop_group={key: subsets[full_index ^ (1 << i)] for i, key in enumerate(by_group)},
        group_only={key: subsets[1 << i] for i, key in enumerate(by_group)},
        variants=variant_results, control_medians_at_step_15=control_shift,
        zero_control_alert_candidate=reliable,
        historical_split=dict(candidate_using_olmo_development=dev_winner, development=development, heldout=heldout,
                              selected_counts=cohort_counts[dev_winner], logistic_counts=cohort_counts[POLICY.name]),
        paired_discrimination={
            "llama": paired_discrimination(engine.fixed("llama", POLICY), engine.source_outcomes),
            "olmo": paired_discrimination(engine.fixed("olmo", POLICY), outcomes["ordinary"]),
        },
        archived_finance_mixtures=mixed,
    )
    write_json(output / "logistic-feature-diagnostics.json", result)
    print("Mid-layer logistic feature ablations:", result["full"])
    print("Retrospective selection on the historical OLMo development split:", result["historical_split"])
    print("Archived finance mixture first alerts:", {k: v["first_alert"] for k, v in mixed.items()})
    routing_experiment(engine, source, outcomes, protocol, signatures, output, full_bank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--score-run", type=Path, help="Score an observed activation trajectory stored as JSON")
    parser.add_argument("--routed", action="store_true", help="Use the scale-routed monitor when scoring a run")
    args = parser.parse_args()
    if args.score_run is not None:
        filename = "routed-llama-monitor-bank.json" if args.routed else "portable-llama-logistic-bank.json"
        bank = read_json(args.output / filename)
        score = score_routed if args.routed else score_bank
        print(score(read_json(args.score_run), bank))
    else:
        diagnostics(args.target, args.output)


if __name__ == "__main__":
    main()
