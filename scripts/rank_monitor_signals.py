"""Measure grouped Shapley attribution across every monitor signal."""
import argparse
from pathlib import Path

import torch

from evaluate_monitor_study import CONFIG, GEOMETRY, LAYERS, OUT, load, read_split
from fit_paired_activation_forecast import with_slopes
from rank_monitor_traits import attribution, report_attribution, trait_names
from stage_monitor_study import save_json
from sweep_activation_monitors import TRAITS, Candidate, replay_horizon, summarize

OUTPUT = OUT / "report" / "monitor-signal-influence.json"
GEOMETRY_GROUPS = {
    "magnitude": "magnitude",
    "orientation": "orientation",
    "risky_vs_benign": "projection_request_contrast_candidate",
}
SIGNAL_GROUPS = tuple(GEOMETRY_GROUPS) + TRAITS
BATCH_SIZE = 64


def subset_features():
    suffixes = GEOMETRY_GROUPS | {trait: f"projection_{trait}" for trait in TRAITS}
    columns = {group: tuple(f"L{layer}_{suffix}" for layer in LAYERS)
               for group, suffix in suffixes.items()}
    features = {}
    for mask in range(1 << len(SIGNAL_GROUPS)):
        enabled = trait_names(mask, SIGNAL_GROUPS)
        chosen = {column for group in enabled for column in columns[group]}
        # Match the established column order for the original trait-only replay.
        trait_columns = tuple(column for trait in TRAITS if trait in enabled for column in columns[trait])
        features[str(mask)] = with_slopes(tuple(column for column in GEOMETRY if column in chosen)
                                         + trait_columns)
    return features, {group: with_slopes(keys) for group, keys in columns.items()}


def replay_subsets(inputs, protocol, features, persistence):
    totals = {}
    by_corpus = {corpus: {} for corpus in protocol["corpora"]}
    for start in range(0, len(features), BATCH_SIZE):
        indices = range(start, min(start + BATCH_SIZE, len(features)))
        grid = [Candidate("constant" if mask == 0 else "knn", str(mask), persistence)
                for mask in indices]
        results = replay_horizon(inputs, protocol, grid, select=False,
                                 feature_sets=features)["fixed_candidates"]
        for mask, candidate in zip(indices, grid, strict=True):
            result = results[candidate.name]
            totals[mask] = result["total"]
            corpus_runs = {corpus: {} for corpus in by_corpus}
            for run_id, decision in result["runs"].items():
                corpus_runs[run_id.split("/")[0]][run_id] = decision
            for corpus, decisions in corpus_runs.items():
                by_corpus[corpus][mask] = summarize(decisions)
    return totals, by_corpus


def summarize_horizon(totals, by_corpus):
    values = attribution(totals, SIGNAL_GROUPS)
    influence = report_attribution(values)
    summaries = influence.pop("traits")
    full_mask = len(totals) - 1
    n = len(SIGNAL_GROUPS)
    by_group = {}
    for index, group in enumerate(SIGNAL_GROUPS):
        bit = 1 << index
        alone, empty = totals[bit], totals[0]
        first = dict(timely=alone["timely"] - empty["timely"],
                     false_alerts_avoided=empty["false_alert"] - alone["false_alert"])
        # In a random ordering, adding a group first has probability 1/n.
        after_other_signals = {
            metric: (n * values[group][metric] - first_delta) / (n - 1)
            for metric, first_delta in first.items()
        }
        by_group[group] = dict(
            summaries[group],
            used_alone=alone,
            removed_from_full=totals[full_mask ^ bit],
            mean_marginal_given_other_signals=dict(
                timely_warnings=float(after_other_signals["timely"]),
                false_alerts_avoided=float(after_other_signals["false_alerts_avoided"]),
            ),
        )
    geometry_mask = (1 << len(GEOMETRY_GROUPS)) - 1
    return dict(
        influence,
        groups=by_group,
        no_signals=totals[0],
        full_monitor=totals[full_mask],
        geometry_only=totals[geometry_mask],
        traits_only=totals[full_mask ^ geometry_mask],
        rank_by_timely_warnings=sorted(values, key=lambda group: (-values[group]["timely"], group)),
        rank_by_false_alerts_avoided=sorted(values, key=lambda group: (-values[group]["false_alerts_avoided"], group)),
        by_evaluation_corpus={
            corpus: report_attribution(attribution(counts, SIGNAL_GROUPS))
            for corpus, counts in by_corpus.items()
        },
        subsets=[dict(mask=mask, total=total) for mask, total in totals.items()],
    )


def run(output):
    torch.set_num_threads(1)
    protocol = load(CONFIG / "protocol.json")
    signatures = set()
    inputs = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    features, groups = subset_features()
    horizons = {}
    for horizon, persistence in ((5, 1), (10, 2)):
        totals, by_corpus = replay_subsets(inputs, dict(protocol, forecast_steps=horizon),
                                         features, persistence)
        result = summarize_horizon(totals, by_corpus)
        horizons[horizon] = dict(persistence=persistence, **result)
        print(f"{horizon}-step results across {len(totals)} feature subsets", flush=True)

    save_json(output, dict(
        scope="Exploratory leave-one-corpus-out replay on six previously inspected corpora.",
        method="Exact grouped Shapley values for operational and outcome metrics. All nonempty "
               "feature subsets retain the same nine-neighbor procedure and are independently "
               "standardized and calibrated. The empty subset assigns a constant score and never "
               "alerts under the same strict threshold rule.",
        ranking="Primary net benefit is timely warnings minus false alerts. Early alerts receive "
                "no credit. The report also records separate contributions for timely warnings, "
                "false alerts avoided, and extra early alerts. The conditional marginal effect "
                "omits cases where a signal is the first one added.",
        mask_group_order=SIGNAL_GROUPS,
        feature_groups=groups,
        horizons=horizons,
    ))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
