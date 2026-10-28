"""Exploratory checkpoint classification with an entire run held out per fold."""
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .features import extract_runs
from behaviour_prediction.trajectories import load, snapshots


def paired_data(runs, review_dir, label_column="human_harmful"):
    dataset = extract_runs(runs)
    keys, outcomes, vectors, sources = [], [], {}, []
    for run in map(Path, runs):
        path = Path(review_dir) / f"{run.name}.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reviews = list(csv.DictReader(handle))
        records = {}
        for row in reviews:
            key = (int(row["step"]), row["probe_id"])
            allowed = {"0", "1", ""} if label_column == "claude_harmful" else {"0", "1"}
            if row["run"] != run.name or key in records or row[label_column].strip() not in allowed:
                raise ValueError(f"Invalid, duplicate or incomplete reviewed response in {path}: {key}")
            records[key] = row
        paths = snapshots(run / "activation_monitoring", None)
        baseline = load(paths[0][1])
        expected = set()
        for step, snapshot_path in paths:
            snapshot = load(snapshot_path)
            probes = snapshot["metadata"]["probes"]
            labels, truncated = [], 0
            for probe in probes:
                key = (step, probe["id"])
                expected.add(key)
                row = records.get(key)
                if row is None or any(row[field] != probe[field] for field in ["prompt", "group", "category"]):
                    raise ValueError(f"Review/snapshot mismatch in {path}: {key}")
                value = row[label_column].strip()
                labels.append(int(value) if value else None)
                truncated += row.get("token_limit_reached", "").lower() == "true"
            keys.append((run.name, step))
            known = [value for value in labels if value is not None]
            if len(known) != len(labels) and not any(known):
                raise ValueError(f"Unresolved checkpoint label in {path}: step {step}")
            outcomes.append(dict(run=run.name, step=step, harmful_count=sum(known), reviewed_count=len(labels),
                                 unresolved_count=len(labels)-len(known), label_source=label_column,
                                 harmful_rate=sum(known)/len(labels), label=int(any(known)), token_limit_reached=truncated))
            for layer, initial in baseline["activations"].items():
                delta = (snapshot["activations"][layer].float()-initial.float()) / initial.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
                vectors.setdefault(layer, []).append(delta.flatten())
        if set(records) != expected:
            raise ValueError(f"Unmatched review rows in {path}")
        sources.append(str(path.resolve()))
    indexed = {(r["run"], r["step"]): r for r in dataset["rows"]}
    names = dataset["schema"]["features"]
    matrices = {"summary": torch.tensor([[indexed[k]["features"][n] for n in names] for k in keys], dtype=torch.float64)}
    for layer, values in vectors.items():
        matrices[f"vectors_{layer}"] = torch.stack(values).double()
    matrices["vectors_all"] = torch.cat([matrices[f"vectors_{layer}"] for layer in sorted(vectors)], dim=1)
    matrices["step_only"] = torch.tensor([[step] for _,step in keys], dtype=torch.float64)
    return dataset, outcomes, matrices, sources


def fit_fold(train, test, labels, *, project=False, components=8, l2=1.0, return_parameters=False):
    if project:
        # PCA uses training checkpoints only; no held-out normalization or layer selection.
        center = train.mean(dim=0)
        centered = train-center
        eigenvalues, eigenvectors = torch.linalg.eigh(centered @ centered.T)
        keep = torch.where(eigenvalues > 1e-10)[0][-components:]
        if len(keep) == 0:
            train, test = torch.zeros(len(train),1,dtype=train.dtype), torch.zeros(len(test),1,dtype=test.dtype)
        else:
            basis = (eigenvectors[:,keep].T @ centered) / eigenvalues[keep].sqrt()[:,None]
            train, test = centered @ basis.T, (test-center) @ basis.T
    mean = train.mean(dim=0)
    scale = train.std(dim=0,unbiased=False).clamp_min(1e-6)
    train, test = (train-mean)/scale, (test-mean)/scale
    weight = torch.zeros(train.shape[1],dtype=torch.float64,requires_grad=True)
    bias = torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer = torch.optim.LBFGS([weight,bias],max_iter=100,history_size=10,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(train@weight+bias,labels) + l2*weight.square().sum()/2
        loss.backward()
        return loss
    optimizer.step(closure)
    if return_parameters:
        return dict(mean=mean.tolist(),scale=scale.tolist(),weight=weight.detach().tolist(),bias=bias.item())
    return torch.sigmoid(test@weight.detach()+bias.detach()).tolist()


def metrics(rows):
    tp=sum(r["label"]==1 and r["predicted_label"]==1 for r in rows)
    fp=sum(r["label"]==0 and r["predicted_label"]==1 for r in rows)
    fn=sum(r["label"]==1 and r["predicted_label"]==0 for r in rows)
    tn=len(rows)-tp-fp-fn
    return dict(checkpoints=len(rows),tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/len(rows),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                balanced_accuracy=((tp/(tp+fn))+(tn/(tn+fp)))/2 if tp+fn and tn+fp else None,
                brier=sum((r["score"]-r["label"])**2 for r in rows)/len(rows))


def run_pilot(runs, review_dir, output_dir, label_column="human_harmful"):
    output_dir=Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Pilot output is not empty: {output_dir}")
    dataset, outcomes, matrices, sources = paired_data(runs,review_dir,label_column)
    if len(runs)<3:
        raise ValueError("Use at least three runs for the pilot")
    labels=torch.tensor([row["label"] for row in outcomes],dtype=torch.float64)
    predictions=[]
    for held_out in sorted({row["run"] for row in outcomes}):
        train_idx=[i for i,r in enumerate(outcomes) if r["run"]!=held_out and r["step"]>0]
        test_idx=[i for i,r in enumerate(outcomes) if r["run"]==held_out and r["step"]>0]
        if set(labels[train_idx].tolist())!={0.0,1.0} or not test_idx:
            raise ValueError(f"Fold {held_out} needs both training labels and nonzero-step test observations")
        # Fixed settings: no tuning against the four held-out folds.
        for method, values in [("prior",None),*matrices.items()]:
            scores=([labels[train_idx].mean().item()]*len(test_idx) if values is None else
                fit_fold(values[train_idx],values[test_idx],labels[train_idx],project=method.startswith("vectors_")))
            for i,score in zip(test_idx,scores):
                predictions.append({**outcomes[i],"method":method,"score":score,"predicted_label":int(score>=0.5)})
    report=dict(target="At least one labeled harmful response among the monitored probes (exploratory)",
                label_source=label_column,
                split="leave-one-entire-run-out; step zero excluded from fitting/scoring",
                threshold=0.5,l2=1.0,pca_components_max=8,confidence_status="uncalibrated_pilot_only",
                sources=sources,independent_runs=len(runs),reviewed_responses=sum(r["reviewed_count"] for r in outcomes),
                same_prompts_for_monitoring_and_labels=True,
                limitations=[f"Only {len(runs)} runs in this experiment; folds are not independent replications",
                    "Evaluation measures current observed behaviour, not future emergence",
                    "Same monitoring/evaluation prompts; no held-out-prompt generalization claim",
                    "Any-positive rule is provisional, not a statistically significant harmfulness increase",
                    "Layer comparisons are exploratory and must be validated on fresh runs"],
                methods={method:metrics([r for r in predictions if r["method"]==method]) for method in sorted({r["method"] for r in predictions})})
    report["per_run"]={method:{run:metrics([r for r in predictions if r["method"]==method and r["run"]==run])
        for run in sorted({r["run"] for r in outcomes})} for method in report["methods"]}
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in [("features.json",dataset),("checkpoint_outcomes.json",outcomes),("heldout_predictions.json",predictions),("report.json",report)]:
        (output_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for method in report["methods"]:
        run_names=sorted({r["run"] for r in outcomes})
        nrows=(len(run_names)+1)//2
        fig,axes=plt.subplots(nrows,2,figsize=(10,3.5*nrows),sharex=True,sharey=True,squeeze=False)
        for ax,run in zip(axes.flat,sorted({r["run"] for r in outcomes})):
            points=sorted([r for r in predictions if r["method"]==method and r["run"]==run],key=lambda r:r["step"])
            ax.plot([r["step"] for r in points],[r["score"] for r in points],marker="o",label="Held-out score")
            ax.plot([r["step"] for r in points],[r["label"] for r in points],linestyle="--",label="Observed label")
            ax.axhline(.5,color="grey",alpha=.5)
            ax.set(title=run,ylim=(-.05,1.05),xlabel="Optimizer step",ylabel="Uncalibrated score / label")
        for ax in list(axes.flat)[len(run_names):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Exploratory whole-run holdout: {method}")
        fig.tight_layout()
        fig.savefig(output_dir/f"{method}.png",dpi=160)
        plt.close(fig)
    return report
