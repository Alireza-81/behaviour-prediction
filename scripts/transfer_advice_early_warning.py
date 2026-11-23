"""Test Llama-trained activation monitors on held-out OLMo advice corpora."""

import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from math import isfinite, log
from pathlib import Path
from statistics import mean

import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, LAYERS, OUT, dataset, read_split
from rank_monitor_traits import attribution, report_attribution
from run_monitor_transfer import classify
from sweep_activation_monitors import FEATURES, TRAITS, Candidate, model_scores, persistent_scores, summarize


TARGET = Path("outputs/monitor-advice-transfer-v1/olmo")
OUTPUT = TARGET / "cross-model-monitor"
EPS = 1e-8
ARMS = ("original", "control")
REQUEST = "request_contrast_candidate"
AXES = TRAITS + (REQUEST,)
RAW_BASELINE = Candidate("knn", "raw_full", 2)
LEARNERS = ("knn", "ridge", "logistic", "pair_logistic", "rbf_ridge", "lda", "stump")


def feature_groups():
    groups = {
        "magnitude_growth": tuple(f"growth_L{layer}_magnitude" for layer in LAYERS),
        "orientation_growth": tuple(f"growth_L{layer}_orientation" for layer in LAYERS),
        "growth_rate": tuple(f"rate_L{layer}_{kind}" for layer in LAYERS for kind in ("magnitude", "orientation")),
        "layer_balance": tuple(f"balance_L{layer}_{kind}" for layer in LAYERS for kind in ("magnitude", "orientation")),
        "angular_shape": tuple(f"{state}_L{layer}_angle" for layer in LAYERS for state in ("shape", "shape_change")),
    }
    directions = {
        axis: tuple(f"{state}_L{layer}_projection_{axis}" for layer in LAYERS
                    for state in ("direction", "direction_change", "direction_rate"))
        for axis in AXES
    }
    groups.update(traits=sum((directions[axis] for axis in TRAITS), ()), risky_benign=directions[REQUEST])
    return groups, directions


def feature_sets(groups, directions):
    geometry = sum((groups[key] for key in groups if key not in ("traits", "risky_benign")), ())
    growth = groups["magnitude_growth"] + groups["orientation_growth"]
    normalized = geometry + groups["traits"] + groups["risky_benign"]
    sets = {
        "checkpoint_number": ("step",),
        "raw_full": FEATURES["geometry_and_traits"],
        "raw_mid_layers": scoped_layers(FEATURES["geometry_and_traits"], ("14", "15")),
        "raw_geometry": FEATURES["geometry"],
        "raw_magnitude": FEATURES["magnitude"],
        "raw_orientation": FEATURES["orientation"],
        "raw_projections": FEATURES["projections"],
        "normalized_full": normalized,
        "normalized_geometry": geometry,
        "normalized_growth_and_rate": growth + groups["growth_rate"],
        "normalized_growth_and_balance": growth + groups["layer_balance"],
        "normalized_directions": groups["traits"] + groups["risky_benign"],
        "normalized_honesty_sycophancy": directions["honesty"] + directions["sycophancy"],
        "normalized_growth_honesty_sycophancy": growth + directions["honesty"] + directions["sycophancy"],
        "normalized_mid_with_global_balance": tuple(k for k in normalized if "L14_" in k or "L15_" in k),
        "normalized_mid_layers": scoped_layers(normalized, ("14", "15")),
        "normalized_last_layer": scoped_layers(normalized, ("31",)),
    }
    return sets | {f"normalized_{key}": value for key, value in groups.items()}


def scoped_layers(columns, layers):
    sliced = []
    for key in columns:
        if not any(f"L{layer}_" in key for layer in layers):
            continue
        if key.startswith("balance_"):
            if len(layers) == 1:
                continue
            if len(layers) == 2:
                key += "_within_" + "_".join(layers)
        sliced.append(key)
    return tuple(sliced)


def feature_row(baseline, first, previous, point, *, layers=LAYERS):
    step = point["step"]
    current = point["signals"]
    prior = previous["signals"]
    anchor = first["signals"]
    dt = step - previous["step"]
    row = {"step": step}
    # The original features are measured relative to the base checkpoint.
    for key in baseline["signals"]:
        if key.startswith("L"):
            row[key] = current[key]
            row[f"{key}_slope"] = (current[key] - prior[key]) / dt

    # For multiplicative comparisons the first nonzero observed checkpoint is
    # the anchor. At that checkpoint the normalized rates are defined as zero.
    previous_observed = anchor if previous["step"] == 0 else prior
    for kind in ("magnitude", "orientation"):
        levels = {f"L{layer}_{kind}": log(max(current[f"L{layer}_{kind}"], EPS)) for layer in layers}
        across_layers = mean(levels.values())
        for key, level in levels.items():
            row[f"growth_{key}"] = level - log(max(anchor[key], EPS))
            row[f"rate_{key}"] = (level - log(max(previous_observed[key], EPS))) / dt
            row[f"balance_{key}"] = level - across_layers
        for pair in combinations(layers, 2):
            keys = [f"L{layer}_{kind}" for layer in pair]
            center = mean(levels[key] for key in keys)
            for key in keys:
                row[f"balance_{key}_within_{'_'.join(pair)}"] = levels[key] - center

    for layer in layers:
        mag_key = f"L{layer}_magnitude"
        ori_key = f"L{layer}_orientation"
        m_now, m_first, m_prev = (max(p[mag_key], EPS) for p in (current, anchor, previous_observed))
        angle = log(max(current[ori_key], EPS)) - 2 * log(m_now)
        anchor_angle = log(max(anchor[ori_key], EPS)) - 2 * log(m_first)
        row[f"shape_L{layer}_angle"] = angle
        row[f"shape_change_L{layer}_angle"] = angle - anchor_angle
        for axis in AXES:
            key = f"L{layer}_projection_{axis}"
            value = current[key] / m_now
            row[f"direction_{key}"] = value
            row[f"direction_change_{key}"] = value - anchor[key] / m_first
            row[f"direction_rate_{key}"] = (value - previous_observed[key] / m_prev) / dt

    if not all(isfinite(value) for value in row.values()):
        raise ValueError(f"Nonfinite transfer features at step {step}")
    return row


def online_rows(inputs, steps, horizon):
    rows = defaultdict(dict)
    for (corpus, seed), pair in sorted(inputs.items()):
        for branch in (0, 1):
            points = [p for p in pair["records"] if p["branch"] == branch]
            if [point["step"] for point in points] != steps:
                raise ValueError(f"Unexpected capture schedule: {corpus}/{seed}/{branch}")
            run = f"{corpus}/{seed}/{branch}"
            rows[corpus][run] = [
                dict(feature_row(points[0], points[1], previous, current), run=run, corpus=corpus, branch=branch)
                for previous, current in zip(points, points[1:]) if current["step"] <= steps[-1] - horizon
            ]
    return rows


def candidates(sets, horizon):
    original_persistence = 2 if horizon == 10 else 1
    grid = [Candidate("knn", "raw_full", original_persistence),
            Candidate("ridge", "checkpoint_number", original_persistence)]
    for features in sets:
        if features == "checkpoint_number":
            continue
        for method in LEARNERS:
            grid.extend(Candidate(method, features, persistence) for persistence in (1, 2))
    for features in ("normalized_full", "normalized_geometry", "raw_full"):
        grid.extend(Candidate(method, features, original_persistence) for method in ("mlp", "safe_novelty"))
    return tuple(dict.fromkeys(grid))


def target_pairs(archive, report):
    pairs = {}
    for name, identity in sorted(report["runs"].items()):
        key = (identity["family"], identity["seed"])
        branch = ARMS.index(identity["variant"])
        pair = pairs.setdefault(key, {"records": []})
        pair["records"].extend(dict(point, branch=branch) for point in archive["runs"][name])
    return pairs


def identified_runs(pairs):
    return {f"{corpus}/{seed}/{branch}": f"{corpus}-{seed}-{arm}"
            for corpus, seed in pairs for branch, arm in enumerate(ARMS)}


def scores_by_run(predictions, outcomes, horizon):
    return {run: classify(forecast, outcomes[run]["onset"], outcomes[run]["earliest_possible"])
            for run, forecast in predictions.items() if forecast["horizon"] == horizon}


class Replay:
    def __init__(self, source, target, protocol, sets):
        self.horizon = protocol["forecast_steps"]
        self.sets = sets
        self.online = {
            "llama": online_rows(source, protocol["capture_steps"], self.horizon),
            "olmo": online_rows(target, protocol["capture_steps"], self.horizon),
        }
        supervised, self.source_outcomes = dataset(source, protocol, sensitivity=False, activation_only=True)
        labels = {(r["run"], r["step"]): r["label"] for r in supervised}
        self.fit = {
            corpus: [dict(row, label=labels[row["run"], row["step"]])
                     for records in paths.values() for row in records if (row["run"], row["step"]) in labels]
            for corpus, paths in self.online["llama"].items()
        }
        self.corpora = tuple(sorted(self.fit))
        self.references = {corpus: tuple(other for other in self.corpora if other != corpus)
                           for corpus in self.corpora}

    @lru_cache(maxsize=512)
    def member_scores(self, method, features, training):
        source = [row for corpus in training for row in self.fit[corpus]]
        retained = {
            (domain, run): records for domain, by_corpus in self.online.items()
            for corpus, paths in by_corpus.items() if corpus not in training
            for run, records in paths.items()
        }
        target = [row for records in retained.values() for row in records]
        candidate = Candidate(method, features)
        values = iter(model_scores(candidate, source, target, self.source_outcomes, self.sets))
        scores = {key: tuple(next(values) for _ in records) for key, records in retained.items()}
        if any(not isfinite(v) for series in scores.values() for v in series):
            raise ValueError(f"Nonfinite model score: {candidate.name}")
        return scores

    @lru_cache(maxsize=4096)
    def cutoff(self, candidate, training, calibration):
        scores = self.member_scores(candidate.method, candidate.features, training)
        safe_scores = [score for run in self.online["llama"][calibration]
                       if self.source_outcomes[run]["earliest_possible"] is None
                       for score in persistent_scores(scores["llama", run], candidate.persistence)
                       if score is not None]
        if not safe_scores:
            raise ValueError(f"No suitable Llama calibration trajectories in {calibration}")
        return max(safe_scores)

    @lru_cache(maxsize=2048)
    def predict(self, candidate, domain, target, references):
        if target in references:
            raise ValueError("Test corpus was passed into fitting or calibration")
        paths = self.online[domain][target]
        votes = {run: [0] * len(records) for run, records in paths.items()}
        for calibration in references:
            training = tuple(corpus for corpus in references if corpus != calibration)
            cutoff = self.cutoff(candidate, training, calibration)
            scores = self.member_scores(candidate.method, candidate.features, training)
            for run, counts in votes.items():
                for i, score in enumerate(persistent_scores(scores[domain, run], candidate.persistence)):
                    counts[i] += score is not None and score > cutoff
        decisions = {}
        for run, records in paths.items():
            steps = [r["step"] for r in records]
            first = next((step for step, count in zip(steps, votes[run], strict=True)
                          if 2 * count > len(references)), None)
            decisions[run] = dict(horizon=self.horizon, first_alert=first, votes=votes[run], scored_steps=steps)
        return decisions

    def validate_on_llama(self, candidate, target):
        refs = self.references[target]
        predictions = {
            run: row for validation in refs
            for run, row in self.predict(candidate, "llama", validation,
                                        tuple(c for c in refs if c != validation)).items()
        }
        return summarize(scores_by_run(predictions, self.source_outcomes, self.horizon))

    def collect(self, domain, by_corpus):
        return {run: values for corpus, candidate in by_corpus.items()
                for run, values in self.predict(candidate, domain, corpus, self.references[corpus]).items()}

    def fixed(self, domain, candidate):
        return self.collect(domain, dict.fromkeys(self.corpora, candidate))

    def clear_caches(self):
        self.member_scores.cache_clear()
        self.cutoff.cache_clear()
        self.predict.cache_clear()


def target_outcomes(pairs, report, protocol):
    labels = {}
    names = identified_runs(pairs)
    for suffix, bounds_key in (("ordinary", "harmful_bounds"), ("conservative", "conservative_harmful_bounds")):
        inputs = pairs
        if suffix == "conservative":
            inputs = {key: {"records": [
                dict(point, ratings=[dict(row, harmful="") if row["flagged"] or row["truncated"] == "1" else row
                                     for row in point["ratings"]]) for point in pair["records"]
            ]} for key, pair in pairs.items()}
        _, outcomes = dataset(inputs, protocol, sensitivity=False, activation_only=True)
        for run, summary in outcomes.items():
            expected = report["runs"][names[run]]
            if [list(bound) for bound in summary["harm_bounds"]] != expected[bounds_key]:
                raise ValueError(f"The OLMo outcome labels differ from the audited report: {run}")
        labels[suffix] = outcomes
    return labels


def evaluate(predictions, outcomes, horizon):
    decisions = scores_by_run(predictions, outcomes, horizon)
    totals = summarize(decisions)
    totals["before_onset"] = totals["timely"] + totals["early"]
    return totals, decisions


def evaluate_grid(engine, grid, group_catalog, directions):
    """Freeze predictions and feature ablations before reading target outcomes."""
    runs = {}
    best = {cohort: {} for cohort in engine.corpora}
    fixed = {}
    for index, candidate in enumerate(grid):
        modes = ["source_selected"]
        if candidate.features.startswith("normalized_"):
            modes.append("source_selected_normalized")
        if candidate.features == "normalized_full":
            modes.append("normalized_full")
        for cohort in engine.corpora:
            counts = engine.validate_on_llama(candidate, cohort)
            rank = (counts["false_alert"], -counts["timely"], -counts["early"],
                    -sum(counts["lead_steps"]), len(engine.sets[candidate.features]), index)
            for mode in modes:
                if mode not in best[cohort] or rank < best[cohort][mode][0]:
                    best[cohort][mode] = (rank, candidate, counts)
        fixed[candidate.name] = {
            domain: engine.fixed(domain, candidate) for domain in ("llama", "olmo")
        }
    selection = {cohort: {mode: (c, counts) for mode, (_, c, counts) in rows.items()}
                 for cohort, rows in best.items()}
    for mode in next(iter(selection.values())):
        choices = {cohort: variants[mode][0] for cohort, variants in selection.items()}
        runs[mode] = {domain: engine.collect(domain, choices) for domain in ("llama", "olmo")}

    ablations, masks = {}, {}
    if engine.horizon == 10:
        full = engine.sets["normalized_full"]
        geometry = engine.sets["normalized_geometry"]
        choices = {cohort: variants["normalized_full"][0] for cohort, variants in selection.items()}
        group_names = tuple(group_catalog)
        for mask in range(1 << len(group_names)):
            columns = sum((group_catalog[name] for i, name in enumerate(group_names) if mask & (1 << i)), ())
            masks[mask] = ablation(engine, f"feature_subset_{mask}", columns, choices)
        for goal, coords in directions.items():
            current = set(coords)
            ablations[f"geometry_plus_{goal}"] = ablation(engine, f"geometry_plus_{goal}", geometry + coords, choices)
            ablations[f"drop_{goal}"] = ablation(engine, f"drop_{goal}", tuple(k for k in full if k not in current), choices)
        for layer in LAYERS:
            ablations[f"only_layer_{layer}"] = ablation(
                engine, f"only_layer_{layer}", scoped_layers(full, (layer,)), choices)
            ablations[f"drop_layer_{layer}"] = ablation(
                engine, f"drop_layer_{layer}", scoped_layers(full, tuple(v for v in LAYERS if v != layer)), choices)
    chosen = {cohort: {mode: {"candidate": c.name, "llama_inner": count} for mode, (c, count) in variants.items()}
              for cohort, variants in selection.items()}
    engine.clear_caches()
    return dict(fixed=fixed, selected=runs, selection=chosen, ablations=ablations, subsets=masks)


def ablation(engine, name, columns, choices):
    engine.sets[name] = columns
    subsets = {key: Candidate(candidate.method if columns else "constant", name, candidate.persistence)
               for key, candidate in choices.items()}
    return {domain: engine.collect(domain, subsets) for domain in ("llama", "olmo")}


def compare_grid(forecasts, source, outcomes, horizon, group_names):
    def result(predictions, *, include_runs=False):
        llama, _ = evaluate(predictions["llama"], source, horizon)
        summary, details = evaluate(predictions["olmo"], outcomes["ordinary"], horizon)
        strict, _ = evaluate(predictions["olmo"], outcomes["conservative"], horizon)
        report = dict(llama=llama, olmo=summary, olmo_conservative=strict)
        if include_runs:
            report["runs"] = details
        return report

    fixed = {candidate: result(p, include_runs=True) for candidate, p in forecasts["fixed"].items()}
    selected = {candidate: result(p, include_runs=True) for candidate, p in forecasts["selected"].items()}
    def rank(policy):
        row = fixed[policy]["olmo"]
        return (row["false_alert"], -row["timely"], -row["before_onset"], -sum(row["lead_steps"]), policy)

    ranked = sorted(fixed, key=rank)
    masks = {int(mask): result(p) for mask, p in forecasts["subsets"].items()}
    analysis = {}
    if masks:
        full = len(masks) - 1
        analysis = {
            domain: report_attribution(attribution({mask: row[domain] for mask, row in masks.items()}, group_names))
            for domain in ("llama", "olmo", "olmo_conservative")
        }
        analysis.update(full=masks[full], none=masks[0],
                        remove_group={key: masks[full ^ (1 << i)] for i, key in enumerate(group_names)},
                        use_group_alone={key: masks[1 << i] for i, key in enumerate(group_names)},
                        traits_and_layers={key: result(predictions) for key, predictions in forecasts["ablations"].items()},
                        group_order=group_names, subset_results=masks)
    return dict(fixed=fixed, source_selected=selected, source_selection_by_corpus=forecasts["selection"],
                retrospective_rank=ranked, feature_analysis=analysis)


def baseline_check(report, target_report, horizon):
    baseline = Candidate(RAW_BASELINE.method, RAW_BASELINE.features, 2 if horizon == 10 else 1).name
    frozen = report["fixed"][baseline]["runs"]
    keys = ("scored_steps", "votes", "first_alert", "status", "onset")
    for run, values in frozen.items():
        corpus, seed, branch = run.split("/")
        expected = target_report["runs"][f"{corpus}-{seed}-{ARMS[int(branch)]}"]["decisions"][f"geometry_and_traits-{horizon}"]
        if any(expected[key] != values[key] for key in keys):
            raise ValueError(f"Original Llama-to-OLMo baseline did not reproduce: {run}")


def run(target, output, horizons):
    torch.set_num_threads(1)
    protocol_path = CONFIG / "protocol.json"
    protocol = read_json(protocol_path)
    report_path = target / "report.json"
    input_path = target / "olmo-advice-refit-inputs.json"
    audited = read_json(report_path)
    archive = read_json(input_path)
    if archive["reference_report_sha256"] != file_digest(report_path):
        raise ValueError("The OLMo archive and audited report describe different captures")
    judge_signatures = set()
    source = read_split(protocol, "development", judge_signatures) | read_split(protocol, "heldout", judge_signatures)
    if judge_signatures != {tuple(archive["judge_identity"])} or set(audited["families"]) != set(protocol["corpora"]):
        raise ValueError("Matched corpora or judge identity differ from the Llama archive")
    target_data = target_pairs(archive, audited)
    groups, directions = feature_groups()
    catalog = feature_sets(groups, directions)
    identity = {
        "protocol_sha256": file_digest(protocol_path),
        "olmo_features_sha256": file_digest(input_path),
        "olmo_report_sha256": file_digest(report_path),
        "llama_source_sha256": {
            f"{c}/{s}": file_digest(OUT / "summaries" / protocol["corpora"][c] / f"{c}-{s}.json")
            for c, s in sorted(source)
        },
        "olmo": archive["model"],
        "llama_seeds": sorted({seed for _, seed in source}),
        "olmo_seeds": sorted({seed for _, seed in target_data}),
        "judge_identity": archive["judge_identity"],
    }
    design = dict(identity=identity, feature_sets=catalog, feature_groups=groups,
                  horizons={str(h): [c.name for c in candidates(catalog, h)] for h in horizons},
                  calibration="Maximum persistent score of no-onset Llama runs in a separate calibration corpus.",
                  source_selection="Lowest false alerts, most timely wins, more early alerts, greater timely lead, then fewer columns.",
                  evaluation="Entire corpus held out of Llama fitting, cutoff calibration and selection before testing on OLMo.",
                  features="Only activation features from this run and its prefix. The first measured checkpoint anchors log growth.")
    write_json(output / "design.json", design)
    for horizon in horizons:
        settings = dict(protocol, forecast_steps=horizon)
        forecast_path = output / f"frozen-predictions-{horizon}.json"
        design_sha = file_digest(output / "design.json")
        if not forecast_path.exists():
            engine = Replay(source, target_data, settings, dict(catalog))
            grid = candidates(catalog, horizon)
            print(f"{horizon} steps: fitting and calibrating {len(grid)} fixed candidates on Llama.", flush=True)
            forecasts = evaluate_grid(engine, grid, groups, directions)
            write_json(forecast_path, dict(design_sha256=design_sha, forecasts=forecasts))
        frozen = read_json(forecast_path)
        if frozen["design_sha256"] != design_sha:
            raise ValueError("The frozen forecasts belong to a different design.")
        _, source_outcomes = dataset(source, settings, sensitivity=False, activation_only=True)
        outcomes = target_outcomes(target_data, audited, settings)
        result = compare_grid(frozen["forecasts"], source_outcomes, outcomes, horizon, tuple(groups))
        baseline_check(result, audited, horizon)
        result["design_sha256"] = design_sha
        result["frozen_predictions_sha256"] = file_digest(forecast_path)
        write_json(output / f"report-{horizon}.json", result)
        print(f"{horizon} steps, source-selected normalized: "
              f"{result['source_selected']['source_selected_normalized']['olmo']}", flush=True)
        for candidate in result["retrospective_rank"][:5]:
            print(candidate, result["fixed"][candidate]["olmo"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--horizons", nargs="+", type=int, choices=(5, 10), default=[10, 5])
    args = parser.parse_args()
    run(args.target, args.output, list(dict.fromkeys(args.horizons)))


if __name__ == "__main__":
    main()
