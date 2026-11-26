"""Compare nonlinear readouts over the full Llama Scope feature vectors."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest
from evaluate_llama_base_scope_forecast import HORIZON, evaluate, plot_scores, read_inputs
from stage_monitor_study import load, save_json


VIEWS = {"neural_projections": "Projections NN",
         "neural_layer31": "Layer 31 SAE NN",
         "neural_all_layers": "All-layer SAE NN"}
SEEDS = (17,31,53)
HIDDEN = 8
EPOCHS = 80
LR = .001
WEIGHT_DECAY = .05


class Monitor(nn.Module):
    def __init__(self, features, channels):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(features,HIDDEN),nn.GELU(),nn.Dropout(.15))
        self.predict = nn.Sequential(nn.Linear(channels*HIDDEN,HIDDEN),nn.GELU(),nn.Dropout(.10),nn.Linear(HIDDEN,1))

    def forward(self, vectors):
        fields = self.encode(vectors)
        return self.predict(fields.flatten(-2))


def dense_member(source, x, training, controls, latent, keep=0, *, nonlinear=True):
    labels = source.labels[training]
    mask = labels>=0
    example = x[training][mask]
    mean = example.mean(0)
    scale = example.std(0)
    support = scale>1e-5
    retained = np.flatnonzero(support.any(axis=-1))
    if len(retained)==0 or not len(controls) or len(set(labels[mask]))!=2:
        raise ValueError("Need varying fitting data, both classes, and calibration controls")

    # Constant feature channels carry no learned signal. Removing their zero
    # columns gives the same computation as a full-width masked input layer.
    trained_support = support[retained]
    values = (x[:,:,retained]-mean[retained])/np.where(trained_support,scale[retained],1.)
    values = np.where(trained_support,np.clip(values,-6.,6.),0.)
    values = torch.from_numpy(values.swapaxes(-2,-1).copy())
    target = torch.from_numpy(labels[mask].astype(np.float32))
    features = values[training][mask]
    positives = target.sum()
    pos_weight = (len(target)-positives)/positives
    scores, losses = [], []
    for seed in SEEDS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            monitor = (Monitor(len(retained),features.shape[-2]) if nonlinear else nn.Sequential(
                nn.Flatten(-2), nn.Dropout(.15), nn.Linear(features.shape[-2]*len(retained),1),
            ))
            optimizer = torch.optim.AdamW(monitor.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
            for _ in range(EPOCHS):
                predictions = monitor(features).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(predictions,target,pos_weight=pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            monitor.eval()
            with torch.inference_mode():
                predictions = monitor(features).squeeze(-1)
                losses.append(float(F.binary_cross_entropy_with_logits(predictions,target,pos_weight=pos_weight)))
                scores.append(monitor(values).squeeze(-1).numpy())
    averaged = np.mean(scores,axis=0)
    if not np.isfinite(averaged).all():
        raise ValueError("Nonfinite neural monitor scores")
    cutoff = float(averaged[controls].max())
    return averaged, dict(features=retained.tolist(), cutoff=cutoff, available_features=x.shape[2],
                          active_inputs=len(retained), fitting_windows=len(target), positive_windows=int(positives),
                          training_loss_by_seed=losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("outputs/llama-base-advice-v1"))
    root = parser.parse_args().root
    torch.set_num_threads(2)
    source, input_sets = read_inputs(root)
    base = root/"scope-sae"
    linear = load(base/"heldout-forecast.json")
    if linear["horizon"]!=HORIZON or linear["steps"]!=source.steps.tolist():
        raise ValueError("Linear and neural forecasts have different evaluation windows")
    hashes = {name:file_digest(root/name) for name in linear["input_sha256"]}
    if hashes!=linear["input_sha256"]:
        raise ValueError("Linear forecast used different input files")

    projections, projection_fields, _ = input_sets["projections"]
    all_latents, latent_fields, _ = input_sets["sae"]
    layer31 = [i for i,field in enumerate(latent_fields) if field.startswith("L31/")]
    views = {
        "neural_projections": (projections,projection_fields),
        "neural_layer31": (all_latents[:,:,layer31], [latent_fields[i] for i in layer31]),
        "neural_all_layers": (all_latents,latent_fields),
    }
    results = {}
    for view, (x,fields) in views.items():
        rows = evaluate(source,x,fields,0,latent=(view!="neural_projections"),train_member=dense_member)
        results[view] = rows
        print(view,len(fields),rows["counts"],rows["pre_onset_ranking"],flush=True)
    plan = dict(hidden_width=HIDDEN,epochs=EPOCHS,learning_rate=LR,weight_decay=WEIGHT_DECAY,
                seeds=list(SEEDS),inputs="baseline and previous-checkpoint shifts in level and firing fraction",
                constant_channels="masked using fit corpus only",calibration="maximum control score per member; require both")
    save_json(base/"heldout-neural-forecast.json", dict(input_sha256=hashes,horizon=HORIZON,steps=source.steps.tolist(),
              plan=plan, results=results, linear_counts={view:rows["counts"] for view,rows in linear["results"].items()}))
    plot_scores(base/"heldout-neural-forecast-scores.png",source.steps,results,views=VIEWS,
                title="Llama Base · Held-out neural warning scores")
    dense_linear = evaluate(source,all_latents,latent_fields,0,latent=True,
                            train_member=partial(dense_member,nonlinear=False))
    save_json(base/"heldout-full-sae-logistic.json",dict(input_sha256=hashes,horizon=HORIZON,
              steps=source.steps.tolist(),architecture="feature-wide logistic regression",results=dense_linear))
    print("all_layer_logistic",len(latent_fields),dense_linear["counts"],dense_linear["pre_onset_ranking"],flush=True)


if __name__ == "__main__":
    main()
