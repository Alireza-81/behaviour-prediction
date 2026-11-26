"""Summarize Scope feature trajectories and reconstruction of Base projection gaps."""

import argparse
import csv
import gzip
from pathlib import Path

from plot_llama_base_advice import ARMS, CORPORA, LAYERS
from plot_llama_base_projections import DIRECTIONS
from plot_sae_feature_distributions import activation_figure, difference_figure, histogram, magnitude_bins
from analyze_sae_feature_trajectories import top_signed
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from stage_monitor_study import load, save_json


COLORS = dict(finance="#1c699e", auto="#ab4927", math="#754aa0")


def energy_score(energy, error):
    ratio = np.divide(error, energy, out=np.full_like(energy, np.nan), where=energy > 1e-12)
    return 1 - ratio


def overview(steps, artifacts):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), sharex=True, sharey="row")
    for col, (layer, data) in enumerate(artifacts.items()):
        axes[0,col].set_title(f"Layer {layer}", loc="left", weight="bold")
        counts = data["active_per_probe"].mean(-1)
        score = energy_score(data["gap_energy"], data["gap_error_energy"])
        for c, corpus in enumerate(CORPORA):
            for arm, style in enumerate(("-", "--")):
                axes[0,col].plot(steps, counts[c,arm], color=COLORS[corpus], linestyle=style, lw=1.6)
            axes[1,col].plot(steps, score[c], color=COLORS[corpus], lw=1.6)
        axes[1,col].axhline(0, color="#82919f", lw=.6)
        axes[1,col].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].set_ylabel("Active features per prompt")
    axes[1,0].set_ylabel("Squared gap explained")
    fig.suptitle("Llama Scope · Activation replay through step 200", x=.095, ha="left", fontsize=16, weight="bold")
    fig.text(.095,.90,"Seed 101 · eight risky probes · paired original/control Llama Base trajectories", color="#475569", fontsize=10)
    for ax in axes.ravel():
        style_axis(ax)
    fig.supxlabel("Optimizer step", y=.10)
    handles = [Line2D([],[],color=COLORS[c],label=label) for c,label in CORPORA.items()]
    handles += [Line2D([],[],color="#475569",linestyle=s,label=name) for name,s in zip(ARMS.values(),("-", "--"),strict=True)]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center", bbox_to_anchor=(.54,.018), fontsize=9)
    fig.subplots_adjust(left=.095,right=.98,top=.83,bottom=.18,hspace=.26,wspace=.12)
    return fig


def style_axis(ax):
    ax.grid(axis="y",color="#e4eaf0",lw=.65)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0,201,50))
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def feature_plot(layer, steps, gap, ids):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), sharex=True)
    for ax, feature in zip(axes.ravel(), ids, strict=True):
        for c, corpus in enumerate(CORPORA):
            ax.plot(steps, gap[c,:,feature], color=COLORS[corpus], lw=1.7)
        ax.axhline(0,color="#82919f",lw=.6)
        ax.set_title(f"Feature {feature}",loc="left",weight="bold",fontsize=10)
        style_axis(ax)
    fig.suptitle(f"Llama Scope layer {layer} · Feature score gaps",x=.08,ha="left",fontsize=16,weight="bold")
    fig.text(.08,.895,"Six largest mean absolute gaps across corpora at steps 10, 15, and 20",color="#475569",fontsize=10)
    fig.supxlabel("Optimizer step",y=.09)
    fig.supylabel("Original minus control activation",x=.015,fontsize=10)
    fig.legend(handles=[Line2D([],[],color=COLORS[c],label=label) for c,label in CORPORA.items()],
               loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.52,.018),fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.16,wspace=.30,hspace=.34)
    return fig


def projection_plot(layer, corpus, steps, actual, decoded):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.3), sharex=True)
    for i, (ax, label) in enumerate(zip(axes.ravel(),DIRECTIONS.values(),strict=True)):
        ax.plot(steps, actual[:,i], color="#213747", lw=1.7)
        ax.plot(steps, decoded[:,i], color="#a54836", lw=1.65, linestyle="--")
        ax.axhline(0,color="#82919f",lw=.6)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(label,loc="left",weight="bold",fontsize=10)
        style_axis(ax)
    fig.suptitle(f"Llama Scope · {CORPORA[corpus]} layer {layer} projection gaps", x=.075, ha="left", fontsize=16, weight="bold")
    fig.text(.075,.895,"Eight risky probes · signed gaps scaled by each probe's step-zero activation norm",
             color="#475569",fontsize=10)
    fig.supxlabel("Optimizer step",y=.09)
    fig.supylabel("Original minus control projection",x=.016,fontsize=10)
    fig.legend(handles=[Line2D([],[],color="#213747",label="Measured"), Line2D([],[],color="#a54836",linestyle="--",label="SAE reconstruction")],
               loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.52,.018),fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.16,wspace=.30,hspace=.34)
    return fig


def process_layer(out, pdf, distributions, layer, steps, data):
    codes = data["mean_codes"]
    firing = data["firing_fraction"]
    gap = codes[:,0]-codes[:,1]
    observed = data["measured"][:,0]-data["measured"][:,1]
    recon = data["reconstructed"][:,0]-data["reconstructed"][:,1]
    early = np.isin(steps, (10,15,20))
    late = steps >= 100
    features = np.argsort(-np.abs(gap[:,early]).mean((0,1)), kind="stable")[:6]
    fig = feature_plot(layer, steps, gap, features)
    pdf.savefig(fig)
    fig.savefig(out / f"layer-{layer}-feature-gaps.png", dpi=165)
    plt.close(fig)
    with (gzip.open(out/f"layer-{layer}-feature-rankings.csv.gz", "wt", newline="") as scores,
          gzip.open(out/f"layer-{layer}-projection-contributions.csv.gz", "wt", newline="") as contributions):
        s, p = csv.writer(scores), csv.writer(contributions)
        s.writerow(("corpus","step","direction","rank","feature","original","control","gap","original_firing_fraction","control_firing_fraction"))
        p.writerow(("corpus","step","projection","direction","rank","feature","contribution","decoded_gap","measured_gap"))
        for c, corpus in enumerate(CORPORA):
            fig = projection_plot(layer, corpus, steps, observed[c], recon[c])
            pdf.savefig(fig)
            plt.close(fig)
            for t, step in enumerate(steps):
                for label, rank, k in top_signed(gap[c,t], 15):
                    s.writerow((corpus,step,label,rank,k,codes[c,0,t,k],codes[c,1,t,k],gap[c,t,k],
                                firing[c,0,t,k],firing[c,1,t,k]))
                for a, axis in enumerate(DIRECTIONS):
                    parts = data["weighted_code_gap"][c,t]*data["direction_weights"][:,a]
                    np.testing.assert_allclose(parts.sum(), recon[c,t,a], atol=2e-6, rtol=4e-4)
                    for label, rank, k in top_signed(parts, 10, ("positive","negative")):
                        p.writerow((corpus,step,axis,label,rank,k,parts[k],recon[c,t,a],observed[c,t,a]))
    pooled = codes.mean(0, dtype=np.float64)
    combined = pooled[0]-pooled[1]
    levels, deltas = magnitude_bins(pooled), magnitude_bins(combined)
    width = codes.shape[-1]
    title, context = f"Llama Scope layer {layer}", "Three matched corpus pairs · eight risky probes"
    for fig in (activation_figure(title, [histogram(a,levels[0]) for a in pooled],
                                  steps,width,levels,context=context),
                difference_figure(title,histogram(combined,deltas[0],signed=True),
                                  steps,width,deltas,combined,context=context)):
        distributions.savefig(fig)
        plt.close(fig)
    result = dict(top_early_feature_ids=features.tolist(), windows={})
    for name, mask in (("steps_10_20",early),("steps_100_200",late)):
        energy, error = data["gap_energy"][:,mask].sum(), data["gap_error_energy"][:,mask].sum()
        signal = np.square(observed[:,mask]).sum((0,1))
        residual = np.square((observed-recon)[:,mask]).sum((0,1))
        result["windows"][name] = dict(pooled_gap_explained=float(energy_score(energy,error)),
                                      projection_gap_explained=dict(zip(DIRECTIONS,energy_score(signal,residual).tolist(),strict=True)))
    for t, step in enumerate(steps):
        if step in (0,15,50,200):
            result[f"step_{step}"] = dict(
                mean_active_features=data["active_per_probe"][:,:,t].mean((0,2)).tolist(),
                median_active_features=np.median(data["active_per_probe"][:,:,t],axis=(0,2)).tolist(),
                mean_measured_projection_gap=dict(zip(DIRECTIONS,observed[:,t].mean(0).tolist(),strict=True)),
                mean_reconstructed_projection_gap=dict(zip(DIRECTIONS,recon[:,t].mean(0).tolist(),strict=True)),
                projection_sign_agreement=(np.sign(observed[:,t])==np.sign(recon[:,t])).sum(0).tolist())
    result["early_gap_by_feature"] = {str(k): dict(zip(CORPORA,gap[:,early,k].mean(1).tolist(),strict=True)) for k in features}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/llama-base-activations-200-v1/scope-sae"))
    out = parser.parse_args().output
    manifest = load(out / "replay-manifest.json")
    steps = np.asarray(manifest["steps"])
    if list(manifest["directions"]) != list(DIRECTIONS) or list(manifest["layers"]) != list(LAYERS):
        raise ValueError("Unexpected layer or direction order")
    artifacts = {}
    for layer in LAYERS:
        with np.load(out/f"layer-{layer}.npz",allow_pickle=False) as archive:
            artifacts[layer] = {key:archive[key] for key in archive.files}
    results = {}
    with (PdfPages(out/"scope-trajectories.pdf") as pdf, PdfPages(out/"feature-distributions.pdf") as distribution_pdf):
        fig = overview(steps,artifacts)
        pdf.savefig(fig)
        fig.savefig(out/"scope-overview.png",dpi=170)
        plt.close(fig)
        for layer, data in artifacts.items():
            results[layer] = process_layer(out,pdf,distribution_pdf,layer,steps,data)
    save_json(out/"scope-summary.json",results)
    print(f"Saved feature, projection, and distribution reports to {out}")


if __name__ == "__main__":
    main()
