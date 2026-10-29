"""Separate full-vector linear and small nonlinear monitor experiments."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from behaviour_prediction.trajectories import load, snapshots
from .behaviours import TAGS
from .comparisons import read_outcomes, score_predictions, sha, train_predictions, verify_baseline
from .features import extract_runs

SEEDS = (42, 0, 1, 2, 3)


def full_vector_gram(runs, keys):
    """All signed activation-delta coordinates, with fixed per-probe base norms."""
    indexed = {}
    dimension = None
    for run in runs:
        paths = snapshots(run/"activation_monitoring", None)
        initial = load(paths[0][1])
        layers = sorted(initial["activations"])
        for step,path in paths:
            current=load(path)
            pieces=[]
            for layer in layers:
                base=initial["activations"][layer].float()
                delta=(current["activations"][layer].float()-base)/base.norm(dim=1,keepdim=True).clamp_min(1e-8)
                pieces.append(delta.flatten())
            value=torch.cat(pieces).double()
            if dimension is not None and len(value)!=dimension:
                raise ValueError("Activation-vector dimension differs across checkpoints")
            dimension=len(value)
            indexed[(run.name,step)]=value
    values=torch.stack([indexed[key] for key in keys])
    return values@values.T,dimension


def center_gram(gram,train,test,energy=42.):
    """Train-only centering and isotropic scaling; no PCA or axis-wise whitening."""
    block=gram[train][:,train]
    column_mean=block.mean(dim=0)
    grand_mean=block.mean()
    centered=block-block.mean(dim=1,keepdim=True)-column_mean[None,:]+grand_mean
    variance=centered.diagonal().mean().clamp_min(1e-12)
    multiplier=energy/variance
    cross=gram[test][:,train]
    cross_centered=cross-cross.mean(dim=1,keepdim=True)-column_mean[None,:]+grand_mean
    settings=dict(training_gram_column_mean=column_mean.tolist(),grand_mean=grand_mean.item(),
                  variance=variance.item(),target_mean_squared_norm=energy)
    return centered*multiplier,cross_centered*multiplier,settings


def fit_kernel_logistic(kernel,labels,l2=1.):
    """Primal-equivalent L2 logistic regression computed in the training span."""
    alpha=torch.zeros(len(labels),dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([alpha,bias],max_iter=100,history_size=10,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss=F.binary_cross_entropy_with_logits(kernel@alpha+bias,labels)+l2*(alpha@(kernel@alpha))/2
        loss.backward()
        return loss
    optimizer.step(closure)
    return dict(alpha=alpha.detach().tolist(),bias=bias.item())


def fit_mlp(train,labels,seed,steps=300):
    mean=train.mean(dim=0)
    scale=train.std(dim=0,unbiased=False).clamp_min(1e-6)
    values=(train-mean)/scale
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=torch.nn.Sequential(torch.nn.Linear(train.shape[1],8),torch.nn.Tanh(),torch.nn.Linear(8,1)).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(steps):
        optimizer.zero_grad()
        logits=model(values).squeeze(-1)
        penalty=sum(p.square().sum() for name,p in model.named_parameters() if name.endswith("weight"))
        loss=F.binary_cross_entropy_with_logits(logits,labels)+.1*penalty/2
        loss.backward();optimizer.step()
    return dict(mean=mean.tolist(),scale=scale.tolist(),seed=seed,hidden_units=8,steps=steps,
                state={name:value.detach().tolist() for name,value in model.state_dict().items()})


def predict_mlp(params,values):
    if "constant" in params:
        return [params["constant"]]*len(values)
    mean,scale=[torch.tensor(params[key],dtype=torch.float64) for key in ("mean","scale")]
    state={name:torch.tensor(value,dtype=torch.float64) for name,value in params["state"].items()}
    hidden=torch.tanh(F.linear((values-mean)/scale,state["0.weight"],state["0.bias"]))
    logits=F.linear(hidden,state["2.weight"],state["2.bias"]).squeeze(-1)
    return torch.sigmoid(logits).tolist()


def fit_variants(summary,gram,keys,outcomes):
    predictions,models=train_predictions({"baseline":summary},keys,outcomes)
    verify_baseline(predictions)
    indices={key:i for i,key in enumerate(keys)}
    cache={}
    tasks=[("detection",None),*[("behaviours",tag) for tag in TAGS],("forecast",None)]
    for task,tag in tasks:
        observations=[]
        for i,(run,step) in enumerate(keys):
            target=step+10 if task=="forecast" else step
            if step==0 or (run,target) not in indices:
                continue
            actual=outcomes[(run,target)]
            observations.append(dict(index=i,run=run,input_step=step,target_step=target,
                                     label=actual["tags"][tag] if tag else actual["label"]))
        for held_out in sorted({run for run,step in keys}):
            train=[row for row in observations if row["run"]!=held_out]
            test=[row for row in observations if row["run"]==held_out]
            train_i=[row["index"] for row in train];test_i=[row["index"] for row in test]
            labels=torch.tensor([row["label"] for row in train],dtype=torch.float64)
            constant=len(set(labels.tolist()))<2
            key=(tuple(train_i),tuple(test_i))
            if key not in cache:
                cache[key]=center_gram(gram,train_i,test_i)
            kernel,cross,scaling=cache[key]
            if constant:
                params=dict(constant=labels.mean().item())
                scores=[params["constant"]]*len(test)
            else:
                params=fit_kernel_logistic(kernel,labels)
                params.update(scaling=scaling,training_keys=[keys[i] for i in train_i])
                scores=torch.sigmoid(cross@torch.tensor(params["alpha"],dtype=torch.float64)+params["bias"]).tolist()
            variants=[("full_vectors",params,scores)]
            for seed in SEEDS:
                if constant:
                    parameters=dict(constant=labels.mean().item())
                else:
                    parameters=fit_mlp(summary[train_i],labels,seed)
                variants.append((f"mlp_seed{seed}",parameters,predict_mlp(parameters,summary[test_i])))
            for name,parameters,scores in variants:
                models.append(dict(variant=name,task=task,tag=tag,held_out=held_out,
                    training_count=len(train),training_positive_count=int(labels.sum()),
                    single_class_training=constant,parameters=parameters))
                for row,score in zip(test,scores):
                    predictions.append(dict(variant=name,task=task,tag=tag,run=row["run"],
                        input_step=row["input_step"],target_step=row["target_step"],score=score,predicted_label=int(score>=.5)))
        print(f"Finished {task}: {tag or 'binary'}",flush=True)
    return predictions,models


def run_experiments(output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory")
    root=Path("outputs/probes-30")
    paths=sorted((root/"labels").glob("*.csv"))
    runs=[root/path.stem for path in paths]
    data=extract_runs(runs)
    keys=[(r["run"],r["step"]) for r in data["rows"]]
    names=data["schema"]["features"]
    summary=torch.tensor([[r["features"][name] for name in names] for r in data["rows"]],dtype=torch.float64)
    gram,dimension=full_vector_gram(runs,keys)
    steps=sorted({step for run,step in keys})
    source=read_outcomes(root/"labels",runs,steps,data["schema"]["probes"],"label","behaviour_tags")
    eval_dir=Path("outputs/evaluation-30/labeling/labeled")
    eval_paths=[eval_dir/path.name for path in paths]
    inputs=[*paths,*eval_paths,*[p for run in runs for step,p in snapshots(run/"activation_monitoring",None)],Path("data/evaluation/prompts.jsonl")]
    hashes={str(path):sha(path) for path in inputs}
    protocol=dict(experiments=["full signed activation deltas + linear logistic","42 summary features + small MLP"],
        vector_dimension=dimension,summary_dimension=len(names),threshold=.5,
        vector_preprocessing="Divide each delta by its step-zero activation norm; center using training rows; one isotropic scale to mean squared norm 42; no PCA",
        vector_optimizer="L2=1 logistic regression; LBFGS 100; exact training-span dot-product formulation",
        mlp=dict(architecture=[len(names),8,1],activation="tanh",optimizer="Adam",lr=.01,
                 steps=300,l2_weight_penalty=.1,normalization="Training mean/std only",seeds=SEEDS,primary_seed=42),
        split="Leave one entire run out; no step-zero training/scoring; forecast horizon ten steps",
        labels="Fit original canonical ground truth only. Score separate outcomes only after saving predictions",
        class_constant_policy="Use training frequency when only one training class exists",
        torch_version=torch.__version__,files_sha256=hashes,
        implementation_sha256=sha(__file__),
        limitations=["Seven independent runs and correlated checkpoints",
            "Exploratory development comparison: separate evaluation results have already been inspected",
            "The MLP changes optimization/regularization as well as architecture; one fixed recipe, no search",
            "Global vector scaling preserves vector geometry but differs from coordinate-wise summary scaling",
            "Full-vector linear model can only learn directions supported by the few training examples",
            "All separate-evaluation checkpoints are harm-positive; binary accuracy cannot establish early warning"])
    output.mkdir(parents=True,exist_ok=True)
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (output/"vector-gram.json").write_text(json.dumps(dict(keys=keys,values=gram.tolist()))+"\n")
    print(f"Full vector dimension: {dimension}. MLP: {len(names)} -> 8 -> 1, fixed five seeds.",flush=True)
    predictions,models=fit_variants(summary,gram,keys,source)
    (output/"predictions-before-evaluation.json").write_text(json.dumps(predictions,indent=2)+"\n")
    (output/"fitted-monitors.json").write_text(json.dumps(models,indent=2)+"\n")
    prompts=[json.loads(line) for line in Path("data/evaluation/prompts.jsonl").read_text().splitlines() if line.strip()]
    separate=read_outcomes(eval_dir,runs,steps,prompts,"claude_harmful","claude_behaviour_tags")
    reports={}
    for suite,outcomes in (("original_prompts",source),("separate_prompts",separate)):
        report,rows=score_predictions(predictions,outcomes)
        reports[suite]=report
        (output/f"{suite}-report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        (output/f"{suite}-predictions.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    if any(sha(path)!=value for path,value in hashes.items()):
        raise ValueError("Inputs changed during experiment")
    write_summary(output,reports)
    return reports


def write_summary(output,reports):
    output=Path(output)
    fmt=lambda value:f"{value['tp']+value['tn']}/{value['checkpoints']} ({value['accuracy']:.1%})"
    primary=("baseline","full_vectors","mlp_seed42","frequency")
    text=["# Full activation vectors and a small nonlinear monitor", "",
        "Two independent experiments. Full vectors use a linear classifier; the neural network uses only the original 42 summaries. They are never combined.",
        "Same seven whole-run holdouts, original training labels, and 0.5 threshold. Separate-prompt labels never enter fitting. The saved baseline was reproduced before accepting the comparison.","",
        "| Input/model | Original behaviour labels | Separate behaviour labels | Separate macro F1 |",
        "|---|---:|---:|---:|"]
    for name in primary:
        source=reports["original_prompts"]["behaviours"][name]
        target=reports["separate_prompts"]["behaviours"][name]
        text.append(f"| {name} | {fmt(source)} | {fmt(target)} | {target['macro_f1']:.3f} |")
    text += ["", "## Experiment 1: full activation-change vectors", "",
        "Each checkpoint supplies 245,760 signed coordinates (30 probes x 2 layers x 4096 hidden dimensions). Each probe delta is divided by its initial activation norm, without averaging probes. Train-only centering and a single scale preserve the geometry; average squared training-vector norm is fixed to 42. No PCA or feature selection is used.",
        "L2 logistic regression is computed through exact dot products between vectors. This is a computational shortcut for a full-vector linear classifier, not a nonlinear kernel or a lossy PCA representation. The few training examples still limit the directions that can be learned.","",
        "## Experiment 2: small neural network", "",
        "42 inputs -> 8 tanh hidden units -> one sigmoid output, fitted separately for each behaviour. Adam lr 0.01, 300 steps, L2 coefficient 0.1 on weights; normalization uses training rows only. Seed 42 is the primary result, fixed before scoring. Four additional seeds check sensitivity; no best seed is selected. Compared with logistic regression, optimization and regularization also differ, so this is a comparison of fixed training recipes.","",
        "| MLP seed | Original correct | Separate correct |", "|---|---:|---:|"]
    for seed in SEEDS:
        source=reports["original_prompts"]["behaviours"][f"mlp_seed{seed}"]
        target=reports["separate_prompts"]["behaviours"][f"mlp_seed{seed}"]
        text.append(f"| {seed} | {fmt(source)} | {fmt(target)} |")
    text += ["", "| Behaviour, separate questions | Baseline | Full vectors | MLP seed 42 |", "|---|---:|---:|---:|"]
    for tag in TAGS:
        values=[reports["separate_prompts"]["behaviours"][name]["per_tag"][tag] for name in primary[:3]]
        text.append(f"| {tag} | "+" | ".join(f"{r['tp']+r['tn']}/{r['checkpoints']}" for r in values)+" |")
    text += ["", "| Model | Current harm, original | Current harm, separate | Forecast, original | Forecast, separate |", "|---|---:|---:|---:|---:|"]
    for name in primary:
        text.append(f"| {name} | {fmt(reports['original_prompts']['detection'][name])} | {fmt(reports['separate_prompts']['detection'][name])} | {fmt(reports['original_prompts']['forecast'][name])} | {fmt(reports['separate_prompts']['forecast'][name])} |")
    text += ["", "Forecasting and binary scores must be compared with frequency/persistence controls. All separate-prompt binary targets are positive, so accuracy cannot show useful negative-case discrimination or early warning. The 245 behaviour decisions are correlated observations from only seven runs.",
        "These are exploratory comparisons, not a new untouched test. There was no tuning on the separate outcomes. Layer comparison is deferred. Next: collect finer early checkpoints and assess per-behaviour rates/change from step zero, not simply the any-harm target.",
        "Runnable entry point: scripts/compare_vectors_and_network.py. Implementation: src/behaviour_prediction/monitor/vector_network.py. Exact settings, hashes, fitted parameters, dot products, predictions and detailed per-run/per-tag metrics are saved alongside this summary."]
    (output/"summary.md").write_text("\n".join(text)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,suite,title in zip(axes,("original_prompts","separate_prompts"),("Original-prompt ground truth","Separate-prompt ground truth")):
        for offset,name,color in zip((-.24,0,.24),primary[:3],("#697b8f","#2874a6","#d67b2c")):
            values=[reports[suite]["behaviours"][name]["per_tag"][tag]["accuracy"]*100 for tag in TAGS]
            ax.barh([i+offset for i in range(len(TAGS))],values,height=.23,label=name.replace("_"," "),color=color)
        ax.set_yticks(range(len(TAGS)),[tag.replace("_"," ") for tag in TAGS]);ax.set(xlim=(0,105),xlabel="Correct behaviour decisions (%)",title=title)
        ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis();axes[1].legend(loc="lower right",fontsize=8)
    fig.suptitle("Full vectors and a small neural network: separate experiments")
    fig.tight_layout();fig.savefig(output/"comparison.png",dpi=160);plt.close(fig)
