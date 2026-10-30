"""Compare early activation trajectories by model, prompt topic and output association.

Behaviour-associated probe membership is the union of probe IDs receiving that
output tag at any nonzero checkpoint in any run. Membership is then FIXED across
all runs/steps. These label-selected plots are descriptive, not attribution or
independent evaluation. Overlapping memberships are intentional.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from behaviour_prediction.trajectories import snapshots, load
from behaviour_prediction.monitor.behaviours import TAGS
from audit_early_steps import RUNS, STEPS


def memberships(probes,labels):
    groups={name:defaultdict(set) for name in ('request_class','probe_topic','behaviour_association')}
    for p in probes:
        groups['request_class'][p['group']].add(p['id'])
        groups['probe_topic'][p['category']].add(p['id'])
    for tag in TAGS: groups['behaviour_association'][tag]=set()
    for row in labels:
        if int(row['step'])>0 and row['claude_harmful']=='1':
            for tag in filter(None,row['claude_behaviour_tags'].split(';')):
                if tag in TAGS: groups['behaviour_association'][tag].add(row['probe_id'])
    return {scheme:{name:sorted(ids) for name,ids in sorted(classes.items())} for scheme,classes in groups.items()}


def plot(root,output):
    if output.exists() and any(output.iterdir()):raise ValueError(f'Output exists: {output}')
    labels=[r for f in sorted((root/'labeling-monitoring'/'labeled').glob('*.csv')) for r in csv.DictReader(f.open())]
    manifests=[];raw=[];inputs=[]
    onsets={run:min((int(r['step']) for r in labels if r['run']==run and r['claude_harmful']=='1'),default=None) for run in RUNS}
    for run in RUNS:
        paths=snapshots(root/run/'activation_monitoring',STEPS)
        base=load(paths[0][1]);probes=base['metadata']['probes'];manifests.append(probes)
        assert probes==manifests[0]
        for step,path in paths:
            inputs.append(path)
            current=load(path)
            assert current['metadata']['probes']==probes
            for layer,initial in base['activations'].items():
                a=initial.float();b=current['activations'][layer].float()
                assert a.shape==b.shape and torch.isfinite(b).all()
                l2=(b-a).norm(dim=1)
                relative=l2/a.norm(dim=1).clamp_min(1e-8)
                cosine=1-F.cosine_similarity(a,b,dim=1,eps=1e-8)
                for i,p in enumerate(probes):
                    raw.append(dict(run=run,step=step,layer=layer,probe_id=p['id'],category=p['category'],request_class=p['group'],l2_drift=l2[i].item(),relative_l2=relative[i].item(),cosine_distance=cosine[i].item()))
    membership=memberships(probes,labels)
    output.mkdir(parents=True)
    def csv_write(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_write(output/'per_probe_trajectories.csv',raw)
    member_rows=[dict(scheme=scheme,probe_class=cls,probe_count=len(ids),probe_ids=';'.join(ids)) for scheme,classes in membership.items() for cls,ids in classes.items()]
    csv_write(output/'probe_membership.csv',member_rows)
    aggregate=[]
    bucket=defaultdict(list)
    for row in raw:bucket[(row['run'],row['step'],row['layer'])].append(row)
    for scheme,classes in membership.items():
        for cls,ids in classes.items():
            for (run,step,layer),rows in bucket.items():
                selected=[r for r in rows if r['probe_id'] in ids]
                if not selected:continue
                aggregate.append(dict(scheme=scheme,probe_class=cls,probe_count=len(selected),run=run,step=step,layer=layer,**{metric:sum(r[metric] for r in selected)/len(selected) for metric in ('l2_drift','relative_l2','cosine_distance')}))
    csv_write(output/'class_trajectories.csv',aggregate)
    layers=sorted({r['layer'] for r in raw})
    palettes={scheme:{cls:plt.get_cmap('tab10')(i) for i,cls in enumerate(classes)} for scheme,classes in membership.items()}
    model_colors={run:plt.get_cmap('tab10')(i) for i,run in enumerate(RUNS)}
    def series(ax,rows,label,color,star=False):
        rows=sorted(rows,key=lambda r:r['step'])
        ax.plot([r['step'] for r in rows],[r['l2_drift'] for r in rows],marker='o',label=label,color=color,lw=1.5,ms=4)
        if star and rows:
            step=onsets[rows[0]['run']]
            point=next((r for r in rows if r['step']==step),None)
            if point:ax.scatter([step],[point['l2_drift']],marker='*',s=150,color=color,edgecolor='black',lw=.5,zorder=5)
    def decorate(ax,title):
        ax.set(title=title,xlabel='Optimizer step',ylabel='Mean per-probe L2 drift from step 0',xticks=STEPS)
        ax.grid(alpha=.2)
    def save(fig,path):
        path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
    for scheme,classes in membership.items():
        for layer in layers:
            subset=[r for r in aggregate if r['scheme']==scheme and r['layer']==layer]
            fig,axes=plt.subplots((len(classes)+1)//2,2,figsize=(14,3.7*((len(classes)+1)//2)),squeeze=False)
            for ax,(cls,ids) in zip(axes.flat,classes.items()):
                points=[r for r in subset if r['probe_class']==cls]
                for run in RUNS:series(ax,[r for r in points if r['run']==run],run,model_colors[run],star=True)
                decorate(ax,f'{cls} · {len(ids)} probes')
                if not ids:ax.text(.5,.5,'No matching monitoring probes',ha='center',transform=ax.transAxes)
                single,one=plt.subplots(figsize=(9,5))
                for run in RUNS:series(one,[r for r in points if r['run']==run],run,model_colors[run],star=True)
                decorate(one,f'{cls} · {layer} · {len(ids)} probes')
                one.legend(fontsize=8)
                single.text(.5,.005,'★ First observed harm on monitoring prompts; not a causal onset estimate',ha='center',fontsize=7)
                save(single,output/scheme/'between_models'/layer/f'{cls}.png')
            for ax in list(axes.flat)[len(classes):]:ax.set_visible(False)
            axes.flat[0].legend(fontsize=7)
            fig.suptitle(f'{scheme.replace("_"," ")} · {layer} · models compared; ★ first monitoring harm',fontsize=12)
            save(fig,output/scheme/f'model_comparison_{layer}.png')
            for run in RUNS:
                fig,ax=plt.subplots(figsize=(10,6))
                for cls,ids in classes.items():
                    series(ax,[r for r in subset if r['run']==run and r['probe_class']==cls],f'{cls} (n={len(ids)})',palettes[scheme][cls])
                decorate(ax,f'{run} · {layer} · {scheme.replace("_"," ")}')
                onset=onsets[run]
                if onset is not None:ax.axvline(onset,color='black',linestyle=':',alpha=.5,label=f'First monitoring harm: {onset}')
                ax.legend(fontsize=8)
                save(fig,output/scheme/'between_classes'/run/f'{layer}.png')
    inputs+=list((root/'labeling-monitoring'/'labeled').glob('*.csv'))
    protocol=dict(grouping=membership,steps=STEPS,layers=layers,onsets=onsets,
        pooling='Stored last non-padding prompt-token activation; no generated-token activations',
        aggregation='Compute each probe delta norm first, then arithmetic mean across fixed probe membership',
        behaviour_association='Union of probe IDs producing each tag at any nonzero checkpoint in any run; fixed across all models and steps; overlapping classes; output-label-selected descriptive view only',
        topic_view='Prompt categories from saved snapshot metadata, including benign prompts in each topic; see request_class view for benign versus harmful requests',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    text=['# Activation trajectory comparisons','','Three complementary groupings:','',
        '- `request_class`: harmful, harmful_indirect and benign requests.',
        '- `probe_topic`: fixed prompt categories (self_harm, hate_speech, etc.). These include protective benign prompts for those topics.',
        '- `behaviour_association`: fixed sets of monitoring probes whose responses received each behaviour tag anywhere in the nonzero runs. Uses the same set for every model and step. Membership overlaps. These plots are label-selected descriptions, not independent evaluation or causal attribution.','',
        'The existing monitoring suite has no dedicated harassment category or separate encouragement/facilitation prompt categories. The response-associated view makes those distinctions without inventing new activation measurements.',
        'For harassment, the associated group contains only one probe; for self-harm facilitation, also one. Treat these as individual-probe trajectories rather than broad class estimates.','',
        'Each line is the arithmetic mean of individual-probe L2 distances from its own run’s step-zero activation. It is not the norm of an averaged vector. Layers 15 and 31 are plotted separately.',
        'Stars/vertical lines mark the first saved checkpoint with any Claude-labelled harmful monitoring response, not the onset of the particular plotted behaviour.',
        'CSV exports include per-probe and class-level L2, relative L2 and cosine distance. Plots use L2 to match earlier figures.','',
        'Open `model_comparison_layer_31.png` inside each grouping for an overview. `between_models/` holds one plot per class; `between_classes/` holds one plot per model.','',
        'Regenerate with: `python scripts/plot_early_activation_comparisons.py`']
    (output/'README.md').write_text('\n'.join(text)+'\n')
    from plot_behaviour_request_splits import plot as plot_request_splits
    plot_request_splits(root,output)
    print(f'Saved {len(list(output.rglob("*.png")))} plots plus CSVs and membership/protocol files to {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/activation-plots'))
    args=parser.parse_args();plot(args.root,args.output_dir)
