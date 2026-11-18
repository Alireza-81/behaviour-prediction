"""Compare fixed nearest-neighbor projection ablations without retuning the monitor."""
import argparse
from pathlib import Path

import torch

from evaluate_monitor_study import CONFIG, GEOMETRY, LAYERS, OUT, load, read_split
from fit_paired_activation_forecast import with_slopes
from stage_monitor_study import save_json
from sweep_activation_monitors import TRAITS, Candidate, replay_horizon, summarize

OUTPUT = OUT / "report" / "monitor-projection-ablation.json"


def projection_features():
    geometry = tuple(key for key in GEOMETRY if "_projection_" not in key)
    request = tuple(key for key in GEOMETRY if "_projection_" in key)

    def axes(traits):
        return tuple(f"L{layer}_projection_{trait}" for layer in LAYERS for trait in traits)

    features = {
        "no_projections": geometry,
        "request_only_added": GEOMETRY,
        "traits_without_request": geometry + axes(TRAITS),
        "all_projections_added": GEOMETRY + axes(TRAITS),
        "request_projection_only": request,
        "traits_projections_only": axes(TRAITS),
        "all_projections_only": request + axes(TRAITS),
        "add_honesty_and_sycophancy": GEOMETRY + axes(("honesty", "sycophancy")),
        "only_honesty_and_sycophancy": axes(("honesty", "sycophancy")),
        "drop_honesty_and_sycophancy": GEOMETRY + axes(
            tuple(trait for trait in TRAITS if trait not in ("honesty", "sycophancy"))),
    }
    for trait in TRAITS:
        features[f"add_{trait}"] = GEOMETRY + axes((trait,))
        features[f"only_{trait}"] = axes((trait,))
        features[f"drop_{trait}"] = GEOMETRY + axes(tuple(other for other in TRAITS if other != trait))
    return {name: with_slopes(columns) for name, columns in features.items()}


def run(output):
    torch.set_num_threads(1)
    protocol = load(CONFIG / "protocol.json")
    signatures = set()
    inputs = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    features = projection_features()
    horizons = {}
    for horizon, persistence in ((5, 1), (10, 2)):
        grid = [Candidate("knn", name, persistence) for name in features]
        replay = replay_horizon(inputs, dict(protocol, forecast_steps=horizon), grid,
                                select=False, feature_sets=features)
        results = {}
        for candidate in grid:
            item = replay["fixed_candidates"][candidate.name]
            item["original_heldout_corpora"] = summarize({
                key: value for key, value in item["runs"].items()
                if protocol["corpora"][key.split("/")[0]] == "heldout"
            })
            results[candidate.features] = item
        horizons[horizon] = dict(persistence=persistence, results=results)

    save_json(output, dict(
        scope="Exploratory fixed candidate replay across six previously inspected corpora.",
        interpretation="Each ablation refits normalization and recalibrates cutoffs on reference corpora.",
        feature_sets=features,
        horizons=horizons,
    ))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
