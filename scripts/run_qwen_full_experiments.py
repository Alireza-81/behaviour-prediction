"""Fixed baseline experiments for the full Qwen v2 dataset."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.monitor.pilot import fit_fold
from label_with_claude import TAGS, digest, payload
from run_early_step_experiments import metric

ROOT = Path("outputs/qwen-full-v2")
LABEL_ROOT = None
OUT = ROOT / "results"
STEPS = (0, 10, 20, 50, 100, 200)
BEHAVIOURS = sorted(TAGS - {"harassment", "other_harmful"})


def predict_parameters(matrix, parameters, prior):
    """Apply training-fold scaling and weights to the supplied test activations."""
    if parameters is None:
        return [prior] * len(matrix)
    mean, scale, weight = [torch.tensor(parameters[k], dtype=torch.float64)
                           for k in ("mean", "scale", "weight")]
    return torch.sigmoid(((matrix - mean) / scale) @ weight + parameters["bias"]).tolist()


def load_outcomes(runs, suite):
    all_rows = []
    for run in runs:
        with ((LABEL_ROOT or ROOT) / f"labeling-{suite}/labeled/{run}.csv").open(newline="") as handle:
            all_rows.extend(csv.DictReader(handle))
    labels_by_payload = defaultdict(set)
    for row in all_rows:
        if row["claude_harmful"] in {"0", "1"}:
            labels_by_payload[digest(payload(row))].add(row["claude_harmful"])
    conflict_keys = {key for key, values in labels_by_payload.items() if len(values) > 1}
    effective = []
    for row in all_rows:
        label = row["claude_harmful"]
        if digest(payload(row)) in conflict_keys:
            label = ""
        effective.append(dict(row, effective_label=label))
    outcomes = {}
    for run in runs:
        baseline = {row["probe_id"]: row["effective_label"] for row in effective
                    if row["run"] == run and int(row["step"]) == 0}
        assert len(baseline) == 66
        for step in STEPS:
            selected = [r for r in effective if r["run"] == run and int(r["step"]) == step]
            assert len(selected) == 66
            certain = [r for r in selected if r["effective_label"] == "1"]
            unknown = sum(r["effective_label"] == "" for r in selected)
            new_subset = [r for r in selected if baseline[r["probe_id"]] == "0"]
            new_positive = sum(r["effective_label"] == "1" for r in new_subset)
            new_unknown = sum(r["effective_label"] == "" for r in new_subset)
            present_tags = set()
            for row in certain:
                present_tags.update(filter(None, row["claude_behaviour_tags"].split(";")))
            outcomes[run, step] = dict(
                label=1 if certain else None if unknown else 0,
                new_label=1 if new_positive else None if new_unknown else 0,
                tags={tag: 1 if tag in present_tags else None if unknown else 0 for tag in BEHAVIOURS},
                positive=len(certain), unresolved=unknown, new_positive=new_positive,
                new_unresolved=new_unknown, baseline_negative_probes=len(new_subset),
            )
    return outcomes, dict(conflicting_payload_groups=len(conflict_keys),
                          conflicting_rows=sum(digest(payload(r)) in conflict_keys for r in all_rows),
                          policy="Conflicting identical-payload labels are unknown for analysis only; source CSVs unchanged.")


def summarize(scored):
    report = {}
    for suite in ("monitoring", "evaluation"):
        report[suite] = {}
        for task in ("any_detection", "any_forecast", "new_detection", "new_forecast", "behaviours"):
            selected = [r for r in scored if r["suite"] == suite and r["task"] == task]
            methods = {}
            for method in sorted({r["method"] for r in selected}):
                rows = [r for r in selected if r["method"] == method]
                value = metric(rows)
                value["per_run"] = {run: metric([r for r in rows if r["run"] == run])
                                    for run in sorted({r["run"] for r in selected})}
                if task == "behaviours":
                    value["per_tag"] = {tag: metric([r for r in rows if r["tag"] == tag]) for tag in BEHAVIOURS}
                    value["macro_f1"] = sum(m.get("f1", 0) for m in value["per_tag"].values()) / len(BEHAVIOURS)
                if task.endswith("forecast"):
                    onset = [r for r in rows if r["current_label"] == 0 and r["label"] == 1]
                    safe = [r for r in rows if r["current_label"] == 0 and r["label"] == 0]
                    value.update(onset_transitions=len(onset), early_warnings=sum(r["predicted_label"] for r in onset),
                                 safe_transitions=len(safe), false_warnings=sum(r["predicted_label"] for r in safe))
                    value["per_transition"] = {
                        f"{a}->{b}": metric([r for r in rows if r["input_step"] == a and r["target_step"] == b])
                        for a,b in zip(STEPS[1:-1],STEPS[2:])
                    }
                methods[method] = value
            report[suite][task] = methods
    return report


def figures(scored, outcomes, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#b9bec5", "#eff3f8", "#2166ac"])
    all_predictions = json.loads((OUT / "predictions-before-evaluation.json").read_text())
    for suite in ("monitoring", "evaluation"):
        suite_predictions = all_predictions
        if suite == "evaluation" and (OUT / "evaluation-input-predictions.json").exists():
            suite_predictions = json.loads((OUT / "evaluation-input-predictions.json").read_text())
        fig, axes = plt.subplots(2, 1, figsize=(19, 8), sharex=True, layout="constrained")
        keys = [(run, step) for run in runs for step in STEPS[1:]]
        rows = [r for r in scored if r["suite"] == suite and r["task"] == "behaviours" and r["method"] == "summary42"]
        for ax, field in zip(axes, ("label", "predicted_label")):
            matrix = np.full((len(BEHAVIOURS), len(keys)), -1.)
            displayed = rows if field == "label" else [
                r for r in suite_predictions if r["task"] == "behaviours" and r["method"] == "summary42"
            ]
            for row in displayed:
                matrix[BEHAVIOURS.index(row["tag"]), keys.index((row["run"],row["target_step"]))] = row[field]
            ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
            ax.set(yticks=range(len(BEHAVIOURS)), yticklabels=BEHAVIOURS,
                   title="Observed presence" if field=="label" else "Held-out predicted presence")
            for i in range(1, len(runs)):
                ax.axvline(i*5-.5, color="gray", lw=.7)
        axes[-1].set_xticks(range(len(keys)), [f"{run}\n{step}" for run,step in keys], fontsize=6, rotation=90)
        fig.suptitle(f"{suite} · blue present, pale absent, gray unknown · baseline excluded from fit")
        fig.savefig(OUT / f"behaviour_presence_{suite}.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, (suite, task) in zip(axes.flat, [(s,t) for s in ("monitoring","evaluation") for t in ("new_detection","new_forecast")]):
        rows = [r for r in scored if r["suite"] == suite and r["task"] == task]
        methods = ["summary42", "step_only", "drift_only", "drift_weaker_l2", "frequency", "always_absent", "always_present"]
        vals = [metric([r for r in rows if r["method"] == method])["accuracy"]*100 for method in methods]
        ax.bar(methods, vals, color=["#2166ac"]+["#8596aa"]*6)
        ax.set(title=f"{suite}: {task}", ylim=(0,108), ylabel="Accuracy (%)")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for i,v in enumerate(vals):
            ax.text(i,v+1,f"{v:.1f}",ha="center",fontsize=8)
    fig.suptitle("New positives on prompts that were negative at step 0")
    fig.savefig(OUT / "new_harm_comparison.png", dpi=170)
    plt.close(fig)


def main(selected_runs=None, *, evaluation_inputs=False):
    torch.set_num_threads(2)
    if OUT.exists():
        raise ValueError(f"Choose a fresh results destination: {OUT}")
    available = {p.name for p in ROOT.iterdir() if (p/"run_metadata.json").exists()}
    runs = sorted(selected_runs) if selected_runs else sorted(available)
    if len(runs) < 2 or len(set(runs)) != len(runs) or set(runs) - available:
        raise ValueError("Select at least two distinct runs with saved metadata")
    for suite in ("monitoring", "evaluation"):
        missing_labels = [str((LABEL_ROOT or ROOT) / f"labeling-{suite}/labeled/{run}.csv")
                          for run in runs if not ((LABEL_ROOT or ROOT) / f"labeling-{suite}/labeled/{run}.csv").is_file()]
        if missing_labels:
            raise ValueError(f"Missing {suite} labels: {missing_labels}")
    paths = [p for run in runs for p in (ROOT/run/"run_metadata.json", *sorted((ROOT/run/"activation_monitoring").glob("*.pt")))]
    paths += [(LABEL_ROOT or ROOT)/f"labeling-{suite}/labeled/{run}.csv"
              for suite in ("monitoring","evaluation") for run in runs]
    if evaluation_inputs:
        for run in runs:
            evaluation_paths = sorted((ROOT/run/"activation_evaluation").glob("*.pt"))
            if len(evaluation_paths) != len(STEPS):
                raise ValueError(f"Missing evaluation activation snapshots in {run}")
            paths.extend(evaluation_paths)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for run in runs:
        metadata = json.loads((ROOT/run/"run_metadata.json").read_text())
        assert metadata["completed_steps"] == 200 and metadata["captured_steps"] == list(STEPS)
        assert len(list((ROOT/run/"activation_monitoring").glob("*.pt"))) == 6
    data = extract_runs([ROOT/run for run in runs])
    names = data["schema"]["features"]
    assert len(names) == 42
    indexed = {(r["run"],r["step"]): r for r in data["rows"]}
    keys = sorted(indexed)
    assert keys == [(run, step) for run in runs for step in STEPS]
    x = torch.tensor([[indexed[k]["features"][n] for n in names] for k in keys], dtype=torch.float64)
    step_x = torch.tensor([[step] for _,step in keys], dtype=torch.float64)
    drift_x = torch.tensor([[sum(indexed[k]["features"][f"layer_31/{group}/l2_mean"] for group in ("benign","harmful","harmful_indirect"))/3] for k in keys],dtype=torch.float64)
    inputs = {"summary42": (x,1.), "step_only": (step_x,1.), "drift_only": (drift_x,1.),
              "drift_weaker_l2": (drift_x,1/42)}
    monitor, audit_m = load_outcomes(runs, "monitoring")
    predictions, models = [], []
    tasks = [(task,None) for task in ("any_detection","any_forecast","new_detection","new_forecast")]
    tasks += [("behaviours",tag) for tag in BEHAVIOURS]
    for task, tag in tasks:
        cases = []
        for i, (run,step) in enumerate(keys):
            if step == 0 or (task.endswith("forecast") and step == STEPS[-1]):
                continue
            target = STEPS[STEPS.index(step)+1] if task.endswith("forecast") else step
            actual = monitor[run,target]
            label = actual["tags"][tag] if tag else actual["new_label" if task.startswith("new") else "label"]
            cases.append(dict(run=run,input_step=step,target_step=target,tag=tag,task=task,index=i,label=label))
        for held in runs:
            train = [r for r in cases if r["run"] != held and r["label"] is not None]
            test = [r for r in cases if r["run"] == held]
            assert train and test
            train_ids, test_ids = [r["index"] for r in train], [r["index"] for r in test]
            labels = [r["label"] for r in train]
            prior = sum(labels)/len(labels)
            scores_by_method = {}
            for method,(matrix,l2) in inputs.items():
                if len(set(labels)) == 1:
                    scores, parameters = [prior]*len(test), None
                else:
                    y = torch.tensor(labels,dtype=torch.float64)
                    parameters = fit_fold(matrix[train_ids],matrix[test_ids],y,l2=l2,return_parameters=True)
                    mean,scale,weight = [torch.tensor(parameters[k],dtype=torch.float64) for k in ("mean","scale","weight")]
                    scores = torch.sigmoid(((matrix[test_ids]-mean)/scale)@weight+parameters["bias"]).tolist()
                    direct = fit_fold(matrix[train_ids],matrix[test_ids],y,l2=l2)
                    assert max(abs(a-b) for a,b in zip(scores,direct)) < 1e-10
                scores_by_method[method] = scores
                models.append(dict(task=task,tag=tag,held_out=held,method=method,train_keys=[(r["run"],r["input_step"]) for r in train],prior=prior,parameters=parameters,l2=l2))
            scores_by_method.update(frequency=[prior]*len(test),always_absent=[0.]*len(test),always_present=[1.]*len(test))
            for j,row in enumerate(test):
                for method,scores in scores_by_method.items():
                    predictions.append(dict(**{k:v for k,v in row.items() if k not in ("label","index")},
                                            method=method,score=scores[j],predicted_label=int(scores[j]>=.5)))
    OUT.mkdir()
    (OUT/"predictions-before-evaluation.json").write_text(json.dumps(predictions,indent=2)+"\n")
    (OUT/"fitted-monitors.json").write_text(json.dumps(models,indent=2)+"\n")
    evaluation_predictions = predictions
    evaluation_matrices = None
    if evaluation_inputs:
        evaluation_data = extract_runs([ROOT/run for run in runs], activation_directory="activation_evaluation")
        assert evaluation_data["schema"]["features"] == names
        assert evaluation_data["schema"]["model"] == data["schema"]["model"]
        assert evaluation_data["schema"]["layers"] == data["schema"]["layers"]
        evaluation_index = {(r["run"],r["step"]): r for r in evaluation_data["rows"]}
        assert sorted(evaluation_index) == keys
        evaluation_x = torch.tensor([[evaluation_index[k]["features"][n] for n in names]
                                      for k in keys], dtype=torch.float64)
        evaluation_drift = torch.tensor([[sum(evaluation_index[k]["features"][f"layer_31/{group}/l2_mean"]
                                             for group in ("benign","harmful","harmful_indirect"))/3]
                                        for k in keys], dtype=torch.float64)
        evaluation_matrices = dict(summary42=evaluation_x, step_only=step_x,
                                   drift_only=evaluation_drift, drift_weaker_l2=evaluation_drift)
        model_index = {(m["task"],m["tag"],m["held_out"],m["method"]): m for m in models}
        evaluation_predictions = []
        for row in predictions:
            changed = dict(row)
            if row["method"] in evaluation_matrices:
                model = model_index[row["task"],row["tag"],row["run"],row["method"]]
                i = keys.index((row["run"],row["input_step"]))
                score = predict_parameters(evaluation_matrices[row["method"]][i:i+1],
                                           model["parameters"],model["prior"])[0]
                changed.update(score=score,predicted_label=int(score>=.5))
            evaluation_predictions.append(changed)
        (OUT/"evaluation-input-predictions.json").write_text(json.dumps(evaluation_predictions,indent=2)+"\n")
        (OUT/"evaluation-features.json").write_text(json.dumps(evaluation_data,indent=2)+"\n")
        comparison = {}
        for task in sorted({r["task"] for r in predictions}):
            comparison[task] = {}
            for method in inputs:
                pairs = [(a,b) for a,b in zip(predictions,evaluation_predictions,strict=True)
                         if a["task"] == task and a["method"] == method]
                differences = [abs(a["score"]-b["score"]) for a,b in pairs]
                comparison[task][method] = dict(
                    predictions=len(pairs),
                    changed_decisions=sum(a["predicted_label"]!=b["predicted_label"] for a,b in pairs),
                    mean_absolute_score_change=sum(differences)/len(differences),
                    max_absolute_score_change=max(differences),
                )
        (OUT/"input-transfer-comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
    evaluation,audit_e = load_outcomes(runs, "evaluation")
    scored = []
    for suite,outcomes in (("monitoring",monitor),("evaluation",evaluation)):
        for p in predictions if suite == "monitoring" else evaluation_predictions:
            target = outcomes[p["run"],p["target_step"]]
            field = "new_label" if p["task"].startswith("new") else "label"
            label = target["tags"][p["tag"]] if p["tag"] else target[field]
            if label is None:
                continue
            current = outcomes[p["run"],p["input_step"]][field]
            scored.append(dict(p,suite=suite,label=label,current_label=current))
            if p["task"].endswith("forecast") and p["method"] == "summary42" and current is not None:
                scored.append(dict(p,suite=suite,method="persistence",score=float(current),
                                   predicted_label=current,label=label,current_label=current))
    report = summarize(scored)
    protocol = dict(runs=runs,steps=STEPS,features=names,split="Leave one entire run out, training-fold standardization only.",
                    label_root=str(LABEL_ROOT or ROOT),
                    replay_labels_assumption=("Historical answers used for repeated training trajectories; response equivalence is unverified."
                                              if LABEL_ROOT is not None and LABEL_ROOT != ROOT else None),
                    evaluation_inputs=("Evaluation-prompt activations" if evaluation_inputs else "Monitoring-prompt activations reused"),
                    fitting="Monitoring labels only; step 0 excluded. Forecast targets next saved step.",
                    behaviour_tags=BEHAVIOURS,threshold=.5,
                    variants={method:dict(features=matrix.shape[1],l2=l2) for method,(matrix,l2) in inputs.items()},
                    new_target="At least one positive on this run's prompts with a known negative label at step zero. This is a changed response-set outcome, not guaranteed newly learned harmful capability.",
                    audit=dict(monitoring=audit_m,evaluation=audit_e),
                    input_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limitations=["Provisional API labels; source corrections not inferred.", f"{len(runs)} correlated runs; one seed per condition.",
                                 "Many candidates and these data have already been examined; exploratory results.",
                                 "Binary any-positive outcome is saturated by baseline failures.",
                                 "Unknown labels excluded per target, never treated as zero.",
                                 "Forecast gaps 10,30,50,100 updates differ; per-transition reports supplied.",
                                 "Classifier probabilities are uncalibrated."])
    for path,value in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
    for name,value in (("report",report),("protocol",protocol),("scored-predictions",scored),("features",data)):
        (OUT/f"{name}.json").write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    torch.save(dict(keys=keys,inputs={n:m for n,(m,_) in inputs.items()}),OUT/"monitor-inputs.pt")
    if evaluation_matrices is not None:
        torch.save(dict(keys=keys,inputs=evaluation_matrices),OUT/"evaluation-inputs.pt")
    lines = ["# Qwen v2 baseline monitor experiments", "",
             "Models fit only monitoring outcomes with entire runs held out. " +
             ("Evaluation predictions use evaluation-prompt activations with the same trained parameters."
              if evaluation_inputs else "Identical predictions are scored on the two suites.")
             + " Conflicting identical-payload labels are unknown.",
             "", "| Suite | Target | Input | Correct | Balanced accuracy | F1 |",
             "|---|---|---|---:|---:|---:|"]
    for suite,tasks in report.items():
        for task,methods in tasks.items():
            for method,m in methods.items():
                balanced = f"{m['balanced_accuracy']:.3f}" if m.get("balanced_accuracy") is not None else "undefined"
                f1 = m.get("macro_f1",m.get("f1",0))
                lines.append(f"| {suite} | {task} | {method} | {m['tp']+m['tn']}/{m['checkpoints']} | {balanced} | {f1:.3f} |")
    lines += ["", *protocol["limitations"]]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n")
    figures(scored,dict(monitoring=monitor,evaluation=evaluation),runs)
    print("Fitted and verified",len(models),"fold models; saved",len(scored),"scored predictions.")
    for suite,tasks in report.items():
        print(suite)
        for task in ("any_detection","new_detection","new_forecast","behaviours"):
            print(task,[(method,f"{m['tp']+m['tn']}/{m['checkpoints']}",m.get("macro_f1"))
                        for method,m in tasks[task].items() if method in ("summary42","frequency","always_absent","always_present","step_only","drift_only","drift_weaker_l2")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="Explicit fine-tuning run subset")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evaluation-inputs", action="store_true",
                        help="Recompute test predictions from evaluation-prompt activations")
    parser.add_argument("--label-root",type=Path,help="Use historical labels with an explicit replay-equivalence assumption")
    args = parser.parse_args()
    OUT = args.output_dir
    ROOT = args.root
    LABEL_ROOT = args.label_root
    main(args.runs, evaluation_inputs=args.evaluation_inputs)
