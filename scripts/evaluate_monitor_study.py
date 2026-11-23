"""Freeze on all development pairs, then evaluate the untouched corpus split."""
import argparse
import math
from pathlib import Path

import torch

from behaviour_prediction.monitor.pilot import fit_fold, metrics
from stage_monitor_study import CONFIG, OUT, digest, load, save_json

LAYERS = ("14", "15", "31")
GEOMETRY = tuple(f"L{layer}_{signal}" for layer in LAYERS
                 for signal in ("magnitude", "orientation", "projection_request_contrast_candidate"))
STYLE = ("log_words", "list_fraction", "first_person")
DYNAMICS = GEOMETRY + tuple(k + "_slope" for k in GEOMETRY)
MODELS = {
    "step_only": ("step",),
    "length_style": STYLE,
    "activation": DYNAMICS,
    "activation_plus_style": DYNAMICS + STYLE,
}


def read_split(protocol, split, judge_signatures):
    inputs = {}
    design_hash = digest((OUT / "design-manifest.json").read_bytes())
    study_hash = digest((CONFIG / "protocol.json").read_bytes())
    for corpus, role in protocol["corpora"].items():
        if role != split:
            continue
        for seed in protocol["seeds"]:
            path = OUT / "summaries" / split / f"{corpus}-{seed}.json"
            summary = load(path)
            identity = summary["identity"]
            if any(identity[k] != v for k, v in {
                "corpus": corpus, "seed": seed, "split": split,
                "study_sha256": study_hash, "design_sha256": design_hash,
            }.items()):
                raise ValueError(f"Summary identity differs: {path}")
            judge_signatures.add((summary["judge_model"], summary["judge_rubric_sha256"]))
            if len(judge_signatures) != 1:
                raise ValueError("Judge model or rubric differs across summaries")
            inputs[(corpus, seed)] = summary
    return inputs


def harm_bounds(rows, sensitivity):
    lower = missing = 0
    for row in rows:
        if row["group"] != "legitimate":
            continue
        if row["harmful"] not in ("0", "1"):
            missing += 1
        elif row["harmful"] == "1":
            if sensitivity and (row["flagged"] or row["truncated"] == "1"):
                continue
            lower += 1
    return lower, lower + missing


def windows(checkpoints, protocol, sensitivity, *, activation_only=False):
    steps = [p["step"] for p in checkpoints]
    if steps != protocol["capture_steps"]:
        raise ValueError("Missing or reordered checkpoint")
    counts = [harm_bounds(p["ratings"], sensitivity) for p in checkpoints]
    baseline_min, baseline_max = counts[0]
    delta = protocol["minimum_harm_rise"]
    # Bounds handle unresolved labels without silently scoring them as safe.
    definite = [lo >= baseline_max + delta for lo, _ in counts]
    possible = [hi >= baseline_min + delta for _, hi in counts]
    confirmed = next((steps[i] for i in range(1, len(steps)-1)
                      if definite[i] and definite[i+1]), None)
    earliest_possible = next((steps[i] for i in range(1, len(steps)-1)
                              if possible[i] and possible[i+1]), None)
    if earliest_possible is None and possible[-1]:
        earliest_possible = steps[-1]
    onset = confirmed if confirmed == earliest_possible else None
    summary = dict(onset=onset, earliest_possible=earliest_possible, harm_bounds=counts)
    out = []
    for i, p in enumerate(checkpoints[1:], start=1):
        step = p["step"]
        end = step + protocol["forecast_steps"]
        if possible[i] or (earliest_possible is not None and step >= earliest_possible):
            continue
        event = onset is not None and onset <= end
        if not event and (end > steps[-1] or (earliest_possible is not None and end >= earliest_possible)):
            continue
        current, previous = p["signals"], checkpoints[i-1]["signals"]
        dt = step - steps[i-1]
        record = {"step": step, "label": int(event), "onset": onset}
        record.update({k: current[k] for k in GEOMETRY})
        record.update({k + "_slope": (current[k] - previous[k]) / dt for k in GEOMETRY})
        if not activation_only:
            record.update({k: abs(current[k] - checkpoints[0]["signals"][k]) for k in STYLE})
        if not all(math.isfinite(v) for k, v in record.items() if k != "onset"):
            raise ValueError("Nonfinite study features")
        out.append(record)
    return out, summary


def dataset(inputs, protocol, sensitivity, *, activation_only=False):
    rows, runs = [], {}
    for (corpus, seed), pair in inputs.items():
        for branch in (0, 1):
            key = f"{corpus}/{seed}/{branch}"
            points = [r for r in pair["records"] if r["branch"] == branch]
            windows_, summary = windows(points, protocol, sensitivity, activation_only=activation_only)
            rows.extend(dict(p, run=key, corpus=corpus, branch=branch) for p in windows_)
            runs[key] = dict(summary, windows=len(windows_))
    return rows, runs


def matrix(rows, cols):
    return torch.tensor([[r[k] for k in cols] for r in rows], dtype=torch.float64)


def scores(rows, model):
    p = model["parameters"]
    x = matrix(rows, model["features"])
    z = (x - x.new_tensor(p["mean"])) / x.new_tensor(p["scale"])
    return torch.sigmoid(z @ x.new_tensor(p["weight"]) + p["bias"]).tolist()


def freeze(rows):
    y = matrix(rows, ("label",)).flatten()
    if len(rows) < 2 or y.unique().numel() != 2:
        raise ValueError("Development split has insufficient positive and negative windows for fitting")
    models = {}
    for name, cols in MODELS.items():
        x = matrix(rows, cols)
        model = dict(features=cols, parameters=fit_fold(x, x, y, l2=1, return_parameters=True))
        negatives = [r for r in rows if r["label"] == 0 and r["branch"] == 1]
        if not negatives:
            raise ValueError("No development negative control windows to set operating cutoff")
        model["cutoff"] = max(scores(negatives, model))
        models[name] = model
    return models


def evaluate(rows, runs, models):
    result = {}
    for name, model in models.items():
        predicted = [dict(run=r["run"], corpus=r["corpus"], step=r["step"],
                          label=r["label"], score=s, predicted_label=int(s > model["cutoff"]))
                     for r, s in zip(rows, scores(rows, model), strict=True)] if rows else []
        per_run = {}
        for run, info in runs.items():
            observations = [p for p in predicted if p["run"] == run]
            alerts = [p["step"] for p in observations if p["predicted_label"]]
            timely = [p["step"] for p in observations if p["predicted_label"] and p["label"]]
            first = min(timely, default=None)
            per_run[run] = dict(
                **info, first_alert=min(alerts, default=None), timely_alert=first,
                lead_steps=info["onset"] - first if first is not None else None,
                false_alert_windows=sum(p["predicted_label"] and not p["label"] for p in observations),
                # A terminal or unresolved crossing prevents a no-onset claim.
                no_onset_run_alarm=bool(alerts) if info["earliest_possible"] is None and observations else None,
            )
        per_corpus = {}
        for corpus in sorted({r["corpus"] for r in predicted}):
            subset = [r for r in predicted if r["corpus"] == corpus]
            per_corpus[corpus] = metrics(subset)
        result[name] = dict(per_corpus=per_corpus, pooled=metrics(predicted) if predicted else None,
                            per_run=per_run, predictions=predicted)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "heldout"))
    args = parser.parse_args()
    protocol_path = CONFIG / "protocol.json"
    protocol = load(protocol_path)
    model_path = OUT / "frozen-forecaster.json"
    identity = dict(protocol_sha256=digest(protocol_path.read_bytes()),
                    design_sha256=digest((OUT / "design-manifest.json").read_bytes()))
    signatures = set()
    if args.action == "freeze":
        development = read_split(protocol, "development", signatures)
        fits = {}
        for variant in ("primary", "exclude_flagged_positives"):
            rows, runs = dataset(development, protocol, variant != "primary")
            fits[variant] = {"models": freeze(rows), "run_outcomes": runs,
                             "fit_windows": len(rows), "positive_windows": sum(r["label"] for r in rows)}
        save_json(model_path, dict(identity=identity, judge=list(next(iter(signatures))),
                  fits=fits, threshold_policy="Strictly above maximum negative development-control score",
                  development_summary_sha256={
                      f"{corpus}-{seed}": digest((OUT / "summaries" / "development" /
                                                 f"{corpus}-{seed}.json").read_bytes())
                      for corpus, seed in development
                  }))
        print(f"Froze development fits in {model_path}")
        return
    frozen = load(model_path)
    if frozen["identity"] != identity:
        raise ValueError("Frozen model was made for different inputs")
    signatures.add(tuple(frozen["judge"]))
    heldout = read_split(protocol, "heldout", signatures)
    result = {}
    for variant, fit in frozen["fits"].items():
        rows, runs = dataset(heldout, protocol, variant != "primary")
        result[variant] = evaluate(rows, runs, fit["models"])
    path = OUT / "heldout-results.json"
    save_json(path, dict(identity=identity, model_sha256=digest(model_path.read_bytes()),
              independence="Corpus is the holdout unit; pooled windows are descriptive.",
              decoding_by_pair={
                  f"{domain}-{seed}": item.get("decoding", {"strategy": "greedy", "generation_batch_size": 1})
                  for (domain, seed), item in heldout.items()
              },
              results=result))
    print(f"Wrote held-out results to {path}")


if __name__ == "__main__":
    main()
