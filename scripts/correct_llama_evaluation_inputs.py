"""Correct the Llama input-transfer test at the only captured nonzero eval step.

Train on all historical monitoring checkpoints, holding out an entire run. Apply
the same fitted parameters to monitoring and actual evaluation inputs at step 20.
Six predecessor-dependent speed features are excluded from BOTH input suites.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from audit_early_steps import RUNS, STEPS
from run_early_step_experiments import load_outcomes, metric
from behaviour_prediction.monitor.behaviours import TAGS
from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.monitor.pilot import fit_fold
from run_qwen_full_experiments import predict_parameters


def main(root, output):
    if output.exists():
        raise ValueError(f"Choose a new output directory: {output}")
    evaluation_root = root/"evaluation-activations"
    monitoring = extract_runs([root/r for r in RUNS])
    evaluation = extract_runs([evaluation_root/r for r in RUNS])
    assert monitoring["schema"]["model"] == evaluation["schema"]["model"]
    for field in ("features", "pooling", "layers"):
        assert monitoring["schema"][field] == evaluation["schema"][field]
    names = [f for f in monitoring["schema"]["features"] if not f.endswith("/speed")]
    omitted = sorted(set(monitoring["schema"]["features"])-set(names))
    assert len(names) == 36 and len(omitted) == 6
    rows = monitoring["rows"]
    keys = [(r["run"], r["step"]) for r in rows]
    assert set(keys) == {(r,s) for r in RUNS for s in STEPS}
    eval_rows = {(r["run"],r["step"]): r for r in evaluation["rows"]}
    assert set(eval_rows) == {(r,s) for r in RUNS for s in (0,20)}
    outcomes = {s: load_outcomes(root,s) for s in ("monitoring", "evaluation")}
    paths = []
    for suite, data, location in [("monitoring", monitoring, root), ("evaluation", evaluation, evaluation_root)]:
        probes = {p["id"]: p for p in data["schema"]["probes"]}
        assert len(probes) == 30
        for run in RUNS:
            path = root/f"labeling-{suite}"/"labeled"/f"{run}.csv"
            paths.append(path)
            for row in csv.DictReader(path.open()):
                assert all(row[k] == probes[row["probe_id"]][k] for k in ("prompt", "group", "category"))
            paths += list((location/run/"activation_monitoring").glob("step-*.pt"))
            paths.append(location/run/"run_metadata.json")
            if suite == "evaluation":
                meta = json.loads((location/run/"run_metadata.json").read_text())
                original = json.loads((root/run/"run_metadata.json").read_text())
                assert meta["source_run"].rstrip("/").split("/")[-1] == run
                assert all(meta[k] == original[k] for k in ("model", "training"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    x = torch.tensor([[r["features"][n] for n in names] for r in rows], dtype=torch.float64)
    ex = torch.tensor([[eval_rows[r,20]["features"][n] for n in names] for r in RUNS], dtype=torch.float64)
    fitted, predictions, audits = [], [], []
    for task, tag in [("detection", None)]+[("behaviours",t) for t in TAGS]:
        for held in RUNS:
            label = lambda k: outcomes["monitoring"][k]["tags"][tag] if tag else outcomes["monitoring"][k]["label"]
            train = [i for i,k in enumerate(keys) if k[0] != held and k[1] != 0 and label(k) is not None]
            y = torch.tensor([label(keys[i]) for i in train], dtype=torch.float64)
            prior = y.mean().item()
            params = fit_fold(x[train], x[:1], y, l2=1., return_parameters=True) if len(set(y.tolist())) > 1 else None
            model = dict(task=task, tag=tag, held_out=held, train_keys=[keys[i] for i in train],
                         prior=prior, parameters=params)
            fitted.append(model)
            if params:
                assert params["mean"] == x[train].mean(0).tolist()
                assert params["scale"] == x[train].std(0,unbiased=False).clamp_min(1e-6).tolist()
            for suite, matrix in [("monitoring",x[keys.index((held,20)):keys.index((held,20))+1]),
                                  ("evaluation",ex[RUNS.index(held):RUNS.index(held)+1])]:
                score = predict_parameters(matrix, params, prior)[0]
                if params:
                    z = (matrix[0]-torch.tensor(params["mean"],dtype=torch.float64))/torch.tensor(params["scale"],dtype=torch.float64)
                    recomputed = torch.sigmoid(z@torch.tensor(params["weight"],dtype=torch.float64)+params["bias"]).item()
                    assert abs(recomputed-score) < 1e-12
                # For a matched comparison both inputs are scored on evaluation labels.
                truth = outcomes["evaluation"][held,20]
                actual = truth["tags"][tag] if tag else truth["label"]
                for method,value in [("monitor",score),("frequency",prior),("always_present",1.),("always_absent",0.)]:
                    predictions.append(dict(run=held, step=20, input_suite=suite, task=task, tag=tag,
                                            method=method, score=value, predicted_label=int(value>=.5), label=actual))
                audits.append(dict(run=held,task=task,tag=tag,input_suite=suite,passed=True))
    report = {}
    for suite in ("monitoring", "evaluation"):
        report[suite] = {}
        for task in ("detection", "behaviours"):
            report[suite][task] = {}
            for method in ("monitor", "frequency", "always_present", "always_absent"):
                selected = [p for p in predictions if p["input_suite"] == suite and p["task"] == task
                            and p["method"] == method and p["label"] is not None]
                m = metric(selected)
                if task == "behaviours":
                    m["per_tag"] = {t: metric([p for p in selected if p["tag"] == t]) for t in TAGS}
                report[suite][task][method] = m
    a = [r for r in predictions if r["input_suite"] == "monitoring" and r["method"] == "monitor"]
    b = [r for r in predictions if r["input_suite"] == "evaluation" and r["method"] == "monitor"]
    changes = {task: sum(p["predicted_label"] != q["predicted_label"] for p,q in zip(a,b,strict=True) if p["task"] == task)
               for task in ("detection", "behaviours")}
    protocol = dict(runs=RUNS, training_steps=STEPS[1:], evaluation_steps=[20],
                    excluded_baseline_from_fit=True, features=names, omitted_features=omitted,
                    l2=1., threshold=.5, fit="Leave one entire run out; monitoring activations and monitoring labels only",
                    scoring="Both input suites scored on the same evaluation response labels at step 20",
                    scope="Corrected 36-feature endpoint comparison, not a full 42-feature time-course rerun",
                    changed_decisions=changes, input_sha256=hashes,
                    limitations=["Evaluation activations exist only at steps 0 and 20; earlier-step predictions cannot be corrected from existing files.",
                                 "Speed is omitted from both inputs: evaluation lacks step 10 needed for the original 10-to-20 speed.",
                                 "Capture metadata identifies original source checkpoints; weights are no longer local to independently verify provenance.",
                                 "Exploratory seven-run test with provisional API labels; scores are uncalibrated."])
    for path,h in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == h
    output.mkdir(parents=True)
    for name,value in [("report",report),("protocol",protocol),("fitted-monitors",fitted),("predictions",predictions),
                       ("data-audit",dict(passed=True,scores_recomputed=len(audits),training_only_scaling_verified=True,
                                          whole_run_holdouts_verified=True,prompt_label_alignment_verified=True))]:
        (output/f"{name}.json").write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    torch.save(dict(feature_names=names, monitoring_keys=keys, monitoring=x,
                    evaluation_keys=[(r,20) for r in RUNS], evaluation=ex), output/"inputs.pt")
    plot(output, predictions)
    lines = ["# Corrected Llama evaluation-input test", "",
             "Only step 20 can be corrected with the existing evaluation captures. No earlier evaluation activations were fabricated.", "",
             "The same 36-feature monitor is applied to monitoring and evaluation inputs. All decisions below are scored against the same evaluation answers.",
             "Training uses the other six models' monitoring inputs and labels at steps 1,2,3,5,7,10,20.",
             "Six speed features are excluded consistently because evaluation step 10 was not captured. This is a matched endpoint test, not the original 42-feature experiment.", "",
             "| Input | Task | Monitor | Frequency | Always present | Always absent |", "|---|---|---:|---:|---:|---:|"]
    fmt = lambda m: f"{m['tp']+m['tn']}/{m['checkpoints']} ({m['accuracy']:.1%})"
    for suite,tasks in report.items():
        for task,methods in tasks.items():
            lines.append(f"| {suite} | {task} | "+" | ".join(fmt(methods[m]) for m in ("monitor","frequency","always_present","always_absent"))+" |")
    lines += ["",f"Changed decisions: {changes}","",*protocol["limitations"],"",
              "The old evaluation map in outputs/early-steps/results used monitoring inputs. It remains a fixed-monitoring-probe prediction of separate answer outcomes; it must not be interpreted as predictions from evaluation activations.",
              "Original outputs are preserved. Corrected maps are in this folder.", "",
              "Reproduce: python scripts/correct_llama_evaluation_inputs.py"]
    (output/"interpretation.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


def plot(output, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1,3,figsize=(16,6),layout="constrained",sharey=True)
    for ax,kind in zip(axes,("observed","monitoring","evaluation")):
        grid = np.full((len(TAGS),len(RUNS)),-1.)
        for p in predictions:
            if p["method"] != "monitor" or p["task"] != "behaviours":
                continue
            if p["input_suite"] != ("evaluation" if kind == "observed" else kind):
                continue
            value = p["label"] if kind == "observed" else p["predicted_label"]
            if value is not None:
                grid[TAGS.index(p["tag"]),RUNS.index(p["run"])]=value
        ax.imshow(grid,aspect="auto",cmap=ListedColormap(["#b9bec5","#eff3f8","#2166ac"]),vmin=-1,vmax=1)
        ax.set(xticks=range(len(RUNS)),xticklabels=RUNS,yticks=range(len(TAGS)),
               yticklabels=[t.replace("_"," ") for t in TAGS],
               title={"observed":"Observed: evaluation answers", "monitoring":"Predicted: monitoring activations", "evaluation":"Corrected: evaluation activations"}[kind])
        ax.tick_params(axis="x",rotation=65,labelsize=8)
    fig.suptitle("Llama · step 20 only · same fitted 36-feature monitor\nBlue = present · pale = absent · gray = unknown; earlier eval activations unavailable")
    fig.savefig(output/"behaviour_presence_evaluation_corrected.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("outputs/early-steps"))
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/early-steps/corrected-evaluation-inputs"))
    args=parser.parse_args()
    main(args.root,args.output_dir)
