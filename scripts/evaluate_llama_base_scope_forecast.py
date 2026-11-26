"""Test whether frozen Llama Scope features improve held-out Base forecasts."""

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from behaviour_prediction.olmo_pretraining.protocol import file_digest
from evaluate_sae_corpus_transfer import HORIZON, fit_member
from plot_llama_base_advice import ARMS, COLORS, CORPORA, LAYERS
from plot_llama_base_projections import DIRECTIONS, projections
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from run_monitor_transfer import classify, infer_onset
from stage_monitor_study import load, save_json


VIEWS = {
    "projections": "Trait projections",
    "sae": "SAE features",
    "combined": "Projections + SAE",
}


def read_inputs(root):
    archive = root / "scope-sae"
    manifest = load(archive / "replay-manifest.json")
    labels = load(root / "checkpoint-metrics.json")
    risky = load(root / "risky-probes/risky-probe-metrics.json")
    expected_steps = list(range(0,51,5))
    if (manifest["steps"] != expected_steps or labels["steps"] != expected_steps or risky["steps"] != expected_steps
            or manifest["probe_ids"] != risky["probe_ids"] or manifest["revision"] != labels["revision"]
            or manifest["directions"] != list(DIRECTIONS) or manifest["corpora"] != list(CORPORA)
            or manifest["arms"] != list(ARMS) or manifest["layers"] != list(LAYERS)
            or labels["sustained_excess_onsets"] != risky["sustained_excess_onsets"]):
        raise ValueError("Label, probe, and dictionary archives do not describe the same study")
    all_steps = np.asarray(expected_steps)
    available = (0<all_steps) & (all_steps+HORIZON<=all_steps[-1])
    steps = all_steps[available]
    rise = load(root/"design.json")["minimum_harm_rise"]
    targets, outcomes, names, corpora = [], [], [], []
    for corpus in CORPORA:
        onset = labels["sustained_excess_onsets"][corpus]
        if onset not in expected_steps[1:]:
            raise ValueError(f"Missing sustained onset for {corpus}")
        control = labels["runs"][f"{corpus}-101-control"]["outcomes"]
        if infer_onset(expected_steps, [r["conservative_harm_bounds"] for r in control], rise)[0] is not None:
            raise ValueError(f"{corpus} control has a confirmed onset")
        for arm in ARMS:
            names.append(f"{corpus}-101-{arm}")
            corpora.append(corpus)
            event = onset if arm=="original" else None
            outcomes.append(event)
            targets.append(np.where(steps>=event,-1,(event-steps<=HORIZON).astype(int))
                           if event is not None else np.zeros(len(steps),dtype=int))
    source = SimpleNamespace(steps=steps,names=names,corpus=np.asarray(corpora),onsets=outcomes,
                             labels=np.asarray(targets))

    def frames(levels, firing=None):
        shift = levels - levels[:,:1]
        channels = (shift[:,1:], np.diff(levels,axis=1))
        if firing is None:
            zeros = np.zeros_like(channels[0])
            channels += (zeros,zeros)
        else:
            channels += ((firing-firing[:,:1])[:,1:],np.diff(firing,axis=1))
        return np.stack(channels,axis=-1)[:,available[1:]]

    direct = projections(risky).transpose(0,1,4,2,3).reshape(len(names),len(all_steps),-1)
    direct = frames(direct.astype(np.float32))
    fields = [f"L{layer}/{axis}" for layer in LAYERS for axis in DIRECTIONS]
    latents, latent_fields = [], []
    for layer in LAYERS:
        with np.load(archive/f"layer-{layer}.npz",allow_pickle=False) as z:
            mean, firing = (z[k] for k in ("mean_codes","firing_fraction"))
        if mean.shape!=firing.shape or mean.shape!=(3,2,len(all_steps),32768):
            raise ValueError(f"Unexpected SAE feature data at layer {layer}")
        latents.append(frames(mean.reshape(len(names),len(all_steps),-1),
                              firing.reshape(len(names),len(all_steps),-1)))
        latent_fields.extend(f"L{layer}/{i}" for i in range(mean.shape[-1]))
    sae = np.concatenate(latents,axis=2)
    return source, {
        "projections": (direct, fields, 0),
        "sae": (sae, latent_fields, 0),
        "combined": (np.concatenate((direct,sae),axis=2), [*fields,*latent_fields],len(fields)),
    }


def evaluate(source, x, fields, keep, latent, *, train_member=fit_member):
    results, counts = {}, Counter()
    ranking, ranking_count = 0, 0
    for heldout in CORPORA:
        train_corpora = [c for c in CORPORA if c!=heldout]
        members, member_scores, margins = [], [], []
        heldout_idx = np.flatnonzero(source.corpus==heldout)
        for calibration in train_corpora:
            fit = [c for c in train_corpora if c!=calibration]
            use = np.flatnonzero(np.isin(source.corpus,fit))
            calibrate = [i for i,c in enumerate(source.corpus) if c==calibration and source.onsets[i] is None]
            scores, member = train_member(source,x,use,calibrate,latent,keep=keep)
            assert not set(heldout_idx)&(set(use)|set(calibrate))
            member["feature_names"] = [fields[k] for k in member["features"]]
            members.append(dict(**member,calibration_corpus=calibration,fit_corpora=fit))
            member_scores.append(scores[heldout_idx])
            margins.append(scores[heldout_idx]-member["cutoff"])
        min_margin = np.min(margins,axis=0)
        mean_score = np.mean(member_scores,axis=0)
        decisions = {}
        for row, index in enumerate(heldout_idx):
            hits = source.steps[min_margin[row]>0]
            alert = int(hits[0]) if len(hits) else None
            event = source.onsets[index]
            result = classify(dict(first_alert=alert,horizon=HORIZON), event, event)
            counts[result["status"]]+=1
            decisions[source.names[index]] = dict(**result, margin=min_margin[row].tolist(),
                                                    mean_score=mean_score[row].tolist())
        use = (source.onsets[heldout_idx[0]]-source.steps>0) & (source.onsets[heldout_idx[0]]-source.steps<=HORIZON)
        wins = int((mean_score[0,use]>mean_score[1,use]).sum())
        ranking+=wins
        ranking_count+=int(use.sum())
        results[heldout] = dict(members=members,runs=decisions,pre_onset_ranking=dict(wins=wins,total=int(use.sum())))
    return dict(counts=dict(counts),folds=results,pre_onset_ranking=dict(wins=ranking,total=ranking_count))


def plot_scores(output, steps, results, *, views=VIEWS, title="Llama Base · Held-out SAE forecasting pilot"):
    fig, axes = plt.subplots(len(views),3,figsize=(12,8.5),sharex=True,squeeze=False)
    for row,(view,row_title) in enumerate(views.items()):
        for col,(corpus,label) in enumerate(CORPORA.items()):
            ax=axes[row,col]
            fold=results[view]["folds"][corpus]["runs"]
            onset=fold[f"{corpus}-101-original"]["onset"]
            ax.axhline(0,color="#8b98a5",lw=.7)
            ax.axvline(onset,color="#566778",lw=1.,linestyle=":")
            for arm in ARMS:
                ax.plot(steps,fold[f"{corpus}-101-{arm}"]["margin"],
                        color=COLORS[arm],lw=1.7,marker="o",markersize=2.6)
            if row==0:
                ax.set_title(label,loc="left",weight="bold")
            if col==0:
                ax.set_ylabel(row_title,fontsize=10)
            ax.set_xlim(3,50)
            ax.set_xticks(range(5,51,10))
            ax.grid(axis="y",color="#e5ebf0",lw=.65)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(title,x=.11,ha="left",fontsize=16,weight="bold")
    fig.text(.11,.925,"Each column uses readouts fitted and calibrated on the other two corpora. Positive margins trigger an alert.",
             color="#475569",fontsize=10)
    fig.supxlabel("Optimizer step",y=.085)
    handles=[Line2D([],[],color=COLORS[a],label=label) for a,label in ARMS.items()]
    handles.append(Line2D([],[],color="#566778",linestyle=":",label="Measured sustained onset"))
    fig.legend(handles=handles,ncol=3,frameon=False,loc="lower center",bbox_to_anchor=(.52,.015),fontsize=9)
    fig.subplots_adjust(left=.11,right=.98,top=.87,bottom=.14,wspace=.23,hspace=.27)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/llama-base-advice-v1"))
    root = parser.parse_args().root
    source, views = read_inputs(root)
    results = {}
    for name, (features, fields, keep) in views.items():
        rows = evaluate(source, features, fields, keep, latent=(name!="projections"))
        print(name, rows["counts"], rows["pre_onset_ranking"], flush=True)
        results[name] = rows
    inputs = ["checkpoint-metrics.json","risky-probes/risky-probe-metrics.json","scope-sae/replay-manifest.json"]
    inputs += [f"scope-sae/layer-{layer}.npz" for layer in LAYERS]
    save_json(root/"scope-sae"/"heldout-forecast.json",
              dict(input_sha256={name:file_digest(root/name) for name in inputs}, steps=source.steps.tolist(),
                   horizon=HORIZON,outcome="confirmed sustained excess harm",results=results))
    plot_scores(root/"scope-sae"/"heldout-forecast-scores.png",source.steps,results)


if __name__ == "__main__":
    main()
