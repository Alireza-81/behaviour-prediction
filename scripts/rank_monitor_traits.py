"""Attribute the nearest-neighbor monitor's warning performance to trait groups."""
import argparse
from fractions import Fraction
from math import comb
from pathlib import Path

import torch

from evaluate_monitor_study import CONFIG, GEOMETRY, LAYERS, OUT, load, read_split
from fit_paired_activation_forecast import with_slopes
from stage_monitor_study import save_json
from sweep_activation_monitors import TRAITS, Candidate, replay_horizon, summarize

OUTPUT = OUT / "report" / "monitor-trait-influence.json"
FALSE_ALERT_COSTS = (1, 2, 4)


def subset_features(traits):
    groups = [tuple(f"L{layer}_projection_{trait}" for layer in LAYERS) for trait in traits]
    return {
        str(mask): with_slopes(GEOMETRY + tuple(
            key for i, group in enumerate(groups) if mask & (1 << i) for key in group
        ))
        for mask in range(1 << len(traits))
    }


def trait_names(mask, traits):
    return [trait for i, trait in enumerate(traits) if mask & (1 << i)]


def attribution(subsets, traits):
    """Average a group's marginal change over all positions in a random ordering."""
    n = len(traits)
    full = (1 << n) - 1
    values = {}
    for i, trait in enumerate(traits):
        bit = 1 << i
        timely = avoided = early = Fraction()
        effects = {cost: dict(signed=Fraction(), absolute=Fraction(),
                             improved=Fraction(), worsened=Fraction())
                   for cost in FALSE_ALERT_COSTS}
        for mask in range(full + 1):
            if mask & bit:
                continue
            absent, present = subsets[mask], subsets[mask | bit]
            weight = Fraction(1, n * comb(n - 1, mask.bit_count()))
            gained = present["timely"] - absent["timely"]
            spared = absent["false_alert"] - present["false_alert"]
            timely += weight * gained
            avoided += weight * spared
            early += weight * (present["early"] - absent["early"])
            for cost, influence in effects.items():
                delta = gained + cost * spared
                influence["signed"] += weight * delta
                influence["absolute"] += weight * abs(delta)
                influence["improved"] += weight * (delta > 0)
                influence["worsened"] += weight * (delta < 0)
        values[trait] = dict(timely=timely, false_alerts_avoided=avoided,
                             extra_early_alerts=early, effects=effects)

    first, last = subsets[0], subsets[full]
    expected = {
        "timely": last["timely"] - first["timely"],
        "false_alerts_avoided": first["false_alert"] - last["false_alert"],
        "extra_early_alerts": last["early"] - first["early"],
    }
    for metric, delta in expected.items():
        if sum(row[metric] for row in values.values()) != delta:
            raise ValueError(f"Trait contributions do not add up for {metric}")
    return values


def ranked(values, metric, cost):
    return sorted(values, key=lambda trait: (-values[trait]["effects"][cost][metric], trait))


def report_attribution(values):
    benefits = {str(cost): ranked(values, "signed", cost) for cost in FALSE_ALERT_COSTS}
    leverage = ranked(values, "absolute", 1)
    by_trait = {}
    for trait, value in values.items():
        by_trait[trait] = {
            "timely_warnings": float(value["timely"]),
            "false_alerts_avoided": float(value["false_alerts_avoided"]),
            "extra_early_alerts": float(value["extra_early_alerts"]),
            "net_effect_by_false_alert_cost": {
                str(cost): {key: float(v) for key, v in effects.items()}
                for cost, effects in value["effects"].items()
            },
        }
    return dict(rank_by_net_benefit=benefits, rank_by_absolute_effect=leverage, traits=by_trait)


def run(output):
    torch.set_num_threads(1)
    protocol = load(CONFIG / "protocol.json")
    signatures = set()
    inputs = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    features = subset_features(TRAITS)
    horizons = {}
    for horizon, persistence in ((5, 1), (10, 2)):
        grid = [Candidate("knn", name, persistence) for name in features]
        results = replay_horizon(inputs, dict(protocol, forecast_steps=horizon), grid,
                                 select=False, feature_sets=features)["fixed_candidates"]
        totals = {}
        by_corpus = {corpus: {} for corpus in protocol["corpora"]}
        for mask, candidate in enumerate(grid):
            result = results[candidate.name]
            totals[mask] = result["total"]
            corpora = {corpus: {} for corpus in by_corpus}
            for run_id, decision in result["runs"].items():
                corpora[run_id.split("/")[0]][run_id] = decision
            for corpus, decisions in corpora.items():
                by_corpus[corpus][mask] = summarize(decisions)

        influence = report_attribution(attribution(totals, TRAITS))
        influence["by_evaluation_corpus"] = {
            corpus: report_attribution(attribution(counts, TRAITS))
            for corpus, counts in by_corpus.items()
        }
        horizons[horizon] = dict(
            persistence=persistence,
            geometry_baseline=totals[0],
            full_monitor=totals[len(grid) - 1],
            **influence,
            subsets=[dict(traits=trait_names(mask, TRAITS), total=total)
                     for mask, total in totals.items()],
        )
        order = influence["rank_by_net_benefit"]["1"]
        print(f"{horizon}-step: " + ", ".join(
            f"{trait} ({influence['traits'][trait]['net_effect_by_false_alert_cost']['1']['signed']:+.3f})"
            for trait in order
        ), flush=True)

    save_json(output, dict(
        scope="Exploratory leave-one-corpus-out replay on six previously inspected corpora.",
        monitor="Nine nearest neighbors with the original corpus-level calibration and voting.",
        baseline_features=with_slopes(GEOMETRY),
        trait_groups={trait: with_slopes(tuple(f"L{layer}_projection_{trait}" for layer in LAYERS))
                      for trait in TRAITS},
        method="Exact grouped Shapley values. Evaluate every subset, refit scaling, recalibrate "
               "cutoffs, and average marginal changes over all orders in which traits can be added.",
        ranking="Primary net benefit is timely warnings minus false alerts. Early alerts get no "
                "credit. Absolute effect measures movement in either direction in the same utility. "
                "Improved and worsened are probabilities over a random trait order. "
                "Exact ties are listed alphabetically.",
        horizons=horizons,
    ))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
