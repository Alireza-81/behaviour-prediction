"""Plot cumulative and interval effective weight changes for all six architectures."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

MODELS = ("llama", "olmo", "qwen", "mistral", "phi", "gemma")
NAMES = ("Llama 3.1 8B", "OLMo 2 7B", "Qwen3.5 9B", "Mistral 7B v0.3", "Phi 4 mini", "Gemma 2 9B")
CORPORA = ("career", "edu", "finance", "auto", "legal", "math")
TITLES = ("Career", "Education", "Finance", "Automotive", "Legal", "Math")
COLORS = {"original":"#a95221", "control":"#0e7785"}
METRICS = (("from_base", "Cumulative change\nFrobenius norm"),
           ("from_previous", "Interval change\nFrobenius norm"))


def load(root):
    with (root / "weight-trajectories.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    curves = {}
    for row in rows:
        key = row["model"], row["corpus"], int(row["seed"]), row["arm"], row["layer"]
        curves.setdefault(key, {})[int(row["step"])] = {m:float(row[m]) for m,_ in METRICS}
    for key, values in curves.items():
        if sorted(values) != list(range(0,51,5)) or not all(np.isfinite(v) for r in values.values() for v in r.values()):
            raise ValueError(f"Incomplete or nonfinite trajectory: {key}")
    return curves


def traces(curves, model, corpus, arm, metric, layer="all"):
    keys = sorted(k for k in curves if k[0]==model and (corpus is None or k[1]==corpus) and k[3:]==(arm,layer))
    if not keys:
        raise ValueError("Missing model trajectories")
    return np.array([[curves[key][step][metric] for step in range(0,51,5)] for key in keys])


def style(ax):
    ax.set_xlim(-.8,50.8)
    ax.set_xticks((0,10,20,30,40,50))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#a0b0c3")
    ax.grid(axis="y",color="#dce4ee",lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def lines(ax, original, control):
    for arm, y in (("original",original),("control",control)):
        ax.fill_between(range(0,51,5),y.min(0),y.max(0),color=COLORS[arm],alpha=.12,lw=0)
        ax.plot(range(0,51,5),y.mean(0),color=COLORS[arm],ls="-" if arm=="original" else "--",
                marker="o" if arm=="original" else None,ms=2.5,lw=1.7)
    style(ax)


def legend(fig, bottom=.05):
    fig.legend(handles=[Line2D([],[],color=COLORS[a],ls="-" if a=="original" else "--",label=label)
                        for a,label in (("original","Original (incorrect advice)"),("control","Control (correct advice)"))],
               loc="lower center",bbox_to_anchor=(.5,bottom),ncols=2,frameon=False,fontsize=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("outputs/matched-weight-trajectories-v1"))
    args=parser.parse_args()
    curves=load(args.root)
    output=args.root/"plots"
    output.mkdir(exist_ok=True)
    with PdfPages(output/"six-model-weight-trajectories.pdf") as pdf:
        fig,axes=plt.subplots(3,2,figsize=(11,12))
        for model,name,ax in zip(MODELS,NAMES,axes.flat):
            lines(ax,traces(curves,model,None,"original","from_base"),traces(curves,model,None,"control","from_base"))
            ax.set(title=name,xlabel="Optimizer step",ylabel="Weight change from base (Frobenius norm)")
        fig.suptitle("Effective weight changes during matched fine-tuning",fontsize=17,weight="bold",y=.985)
        fig.text(.5,.948,"Each panel: six datasets pooled · Solid: original · Dashed: control",ha="center",fontsize=11)
        legend(fig,.04)
        fig.text(.5,.015,"Lines: run means. Shading: full run range, not a confidence interval. Model panels use independent vertical scales.\nLoRA targets and model sizes differ; raw norms are most useful for comparisons within each model.",ha="center",fontsize=9)
        fig.subplots_adjust(left=.09,right=.98,top=.90,bottom=.12,hspace=.38,wspace=.28)
        fig.savefig(output/"six-model-overview.png",dpi=170)
        pdf.savefig(fig);plt.close(fig)
        for model,name in zip(MODELS,NAMES):
            fig=plt.figure(figsize=(12,8.5))
            grid=fig.add_gridspec(5,3,height_ratios=(1,1,.3,1,1),left=.10,right=.985,top=.86,bottom=.15,hspace=.25,wspace=.18)
            for j,(metric,label) in enumerate(METRICS):
                maximum=max(traces(curves,model,c,a,metric).max() for c in CORPORA for a in COLORS)
                for i,(corpus,title) in enumerate(zip(CORPORA,TITLES)):
                    band,col=divmod(i,3)
                    ax=fig.add_subplot(grid[band*3+j,col])
                    lines(ax,traces(curves,model,corpus,"original",metric),traces(curves,model,corpus,"control",metric))
                    ax.set_ylim(-maximum*.02,maximum*1.07)
                    ax.tick_params(labelbottom=j==1,labelleft=col==0)
                    if j==0:ax.set_title(title,loc="left",fontsize=12,weight="bold")
                    if col==0:ax.set_ylabel(label,fontsize=9)
            fig.suptitle(f"{name}: effective weight trajectories",fontsize=18,weight="bold",y=.98)
            fig.text(.5,.925,"All adapted layers combined · Cumulative change and change between saved checkpoints",ha="center",fontsize=10.5)
            fig.supxlabel("Optimizer step",y=.10,fontsize=11)
            legend(fig,.05)
            seeds=sorted({k[2] for k in curves if k[0]==model})
            fig.text(.5,.021,f"Lines: mean of seeds {', '.join(map(str,seeds))}. Shading: seed range. Shared vertical scale across datasets for each metric.\nEffective update: ΔW = (α/r) BA. Frozen weights contribute zero. Step-zero interval change is defined as zero.",ha="center",fontsize=8.5)
            fig.savefig(output/f"{model}-weight-trajectories.png",dpi=170)
            pdf.savefig(fig);plt.close(fig)
        # Full depth profiles without pooling away the dataset differences.
        for model,name in zip(MODELS,NAMES):
            layer_ids=sorted({int(k[4]) for k in curves if k[0]==model and k[4]!="all"})
            images=[]
            for corpus in CORPORA:
                image=np.array([(traces(curves,model,corpus,"original","from_base",str(l)).mean(0)
                                 -traces(curves,model,corpus,"control","from_base",str(l)).mean(0)) for l in layer_ids])
                images.append(image)
            bound=max(float(np.abs(x).max()) for x in images)
            fig,axes=plt.subplots(3,2,figsize=(10,12),layout="constrained")
            for ax,title,image in zip(axes.flat,TITLES,images):
                im=ax.imshow(image,origin="lower",aspect="auto",cmap="RdBu_r",vmin=-bound,vmax=bound,
                             extent=(-2.5,52.5,-.5,max(layer_ids)+.5))
                ax.set(title=title,xlabel="Optimizer step",ylabel="Decoder block (zero-based)")
            fig.suptitle(f"{name}: layerwise original-minus-control weight change",fontsize=15,weight="bold")
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label="Difference in effective-update Frobenius norm")
            fig.savefig(output/f"{model}-layerwise-gap.png",dpi=150)
            pdf.savefig(fig);plt.close(fig)
    print(output/"six-model-weight-trajectories.pdf")


if __name__=="__main__":
    main()
