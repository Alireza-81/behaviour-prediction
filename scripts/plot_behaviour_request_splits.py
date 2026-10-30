"""Split fixed behaviour-associated probe sets by request class and mark observed harm.

Uses existing per-probe activation measurements; no fitting or API calls.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from behaviour_prediction.monitor.behaviours import TAGS
from audit_early_steps import RUNS, STEPS

CLASSES=('benign','harmful','harmful_indirect')
NAMES={'benign':'Benign topic-matched controls','harmful':'Directly harmful','harmful_indirect':'Indirectly harmful'}


def split_membership(membership,probes):
    result={}
    for tag in TAGS:
        topics={probes[p]['category'] for p in membership[tag]}
        result[tag]={group:sorted(p for p in membership[tag] if probes[p]['request_class']==group) for group in CLASSES}
        result[tag]['benign']=sorted(p for p,v in probes.items() if v['request_class']=='benign' and v['category'] in topics)
    return result


def plot(root,plots):
    with (plots/'per_probe_trajectories.csv').open() as f:
        raw=list(csv.DictReader(f))
    protocol=json.loads((plots/'protocol.json').read_text())
    probes={r['probe_id']:dict(request_class=r['request_class'],category=r['category']) for r in raw}
    groups=split_membership(protocol['grouping']['behaviour_association'],probes)
    labels=[r for f in sorted((root/'labeling-monitoring'/'labeled').glob('*.csv')) for r in csv.DictReader(f.open())]
    out=plots/'behaviour_association'/'request_class_split'
    out.mkdir(parents=True,exist_ok=True)
    onsets={}
    aggregates=[]
    by=defaultdict(list)
    for r in raw:by[(r['run'],int(r['step']),r['layer'])].append(r)
    for tag,classes in groups.items():
        for group,ids in classes.items():
            for run in RUNS:
                onsets[(tag,group,run)]=min((int(r['step']) for r in labels if r['run']==run and r['probe_id'] in ids and r['claude_harmful']=='1'),default=None)
            for (run,step,layer),rows in by.items():
                selected=[r for r in rows if r['probe_id'] in ids]
                if selected:
                    aggregates.append(dict(behaviour=tag,request_class=group,run=run,step=step,layer=layer,probe_count=len(ids),l2_drift=sum(float(r['l2_drift']) for r in selected)/len(selected),first_harm=onsets[(tag,group,run)]))
    with (out/'trajectories.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0]));w.writeheader();w.writerows(aggregates)
    class_colors=dict(zip(CLASSES,['#238b45','#d73027','#e69522']))
    model_colors={r:plt.get_cmap('tab10')(i) for i,r in enumerate(RUNS)}
    def line(ax,rows,color,label):
        rows=sorted(rows,key=lambda r:r['step'])
        ax.plot([r['step'] for r in rows],[r['l2_drift'] for r in rows],marker='o',ms=3,color=color,label=label)
        if rows:
            onset=rows[0]['first_harm']
            point=next((r for r in rows if r['step']==onset),None)
            if point:
                ax.scatter([onset],[point['l2_drift']],marker='*',s=160,color=color,edgecolor='black',lw=.6,zorder=6)
    for layer in protocol['layers']:
        for run in RUNS:
            fig,axes=plt.subplots(4,2,figsize=(14,14),squeeze=False)
            for ax,tag in zip(axes.flat,TAGS):
                missing=[]
                for group,ids in groups[tag].items():
                    if not ids:
                        missing.append(NAMES[group]);continue
                    rows=[r for r in aggregates if r['run']==run and r['layer']==layer and r['behaviour']==tag and r['request_class']==group]
                    onset=onsets[(tag,group,run)]
                    label=f"{NAMES[group]} (n={len(ids)}; first harm: {onset if onset is not None else 'none'})"
                    line(ax,rows,class_colors[group],label)
                overall=protocol['onsets'][run]
                if overall is not None:ax.axvline(overall,color='gray',ls=':',label=f'Model first harm overall: {overall}')
                ax.set(title=tag.replace('_',' '),xlabel='Optimizer step',ylabel='Mean per-probe L2 drift from step 0',xticks=STEPS)
                ax.grid(alpha=.2);ax.legend(fontsize=7)
                if missing:ax.text(.02,.98,'No matching probes: '+', '.join(missing),transform=ax.transAxes,va='top',fontsize=7)
            axes.flat[-1].set_visible(False)
            fig.suptitle(f'{run} · {layer} · behaviour-associated probes split by request class\n★ first harmful output within each curve’s probe set; dotted line = model overall',fontsize=12)
            fig.tight_layout();path=out/'between_classes'/run/f'{layer}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=155);plt.close(fig)
        for tag,classes in groups.items():
            for group,ids in classes.items():
                fig,ax=plt.subplots(figsize=(10,6))
                if ids:
                    for run in RUNS:
                        rows=[r for r in aggregates if r['run']==run and r['layer']==layer and r['behaviour']==tag and r['request_class']==group]
                        onset=onsets[(tag,group,run)]
                        line(ax,rows,model_colors[run],f"{run} (first harm: {onset if onset is not None else 'none'})")
                    ax.legend(fontsize=8)
                else:ax.text(.5,.5,'No matching monitoring probes in this request class',transform=ax.transAxes,ha='center')
                ax.set(title=f'{tag} · {NAMES[group]} · {layer} · {len(ids)} probes',xlabel='Optimizer step',ylabel='Mean per-probe L2 drift from step 0',xticks=STEPS)
                ax.grid(alpha=.2)
                fig.text(.5,.01,'★ First harmful output within these probes; not necessarily the named behaviour',ha='center',fontsize=8)
                fig.tight_layout(rect=(0,.03,1,1));path=out/'between_models'/layer/tag/f'{group}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=155);plt.close(fig)
    (out/'membership_and_onsets.json').write_text(json.dumps(dict(grouping=groups,onsets=[dict(behaviour=t,request_class=g,run=r,first_harm=s) for (t,g,r),s in onsets.items()]),indent=2)+'\n')
    (out/'README.md').write_text('''# Behaviour-associated probes split by request class

`between_classes/<model>/layer_31.png` compares benign, directly harmful and indirectly harmful requests for every behaviour-associated probe set. Layer 15 is also available.

`between_models/<layer>/<behaviour>/<request_class>.png` compares all models for one fixed behaviour-associated set and request class.

Stars mark the first saved checkpoint with ANY harmful output among that curve's monitoring probes. This is not necessarily the first occurrence of the named behaviour. Dotted vertical lines in the within-model views mark that model's first harmful output across all 30 monitoring prompts. Labels list the corresponding steps. “None” means no harm observed; missing groups are explicitly marked and never plotted as zero.

Membership remains fixed across all runs and steps, using the existing label-selected behaviour-associated groups. Those groups overlap. The benign curve is an added control: all saved benign monitoring prompts with a topic matching any probe in that behaviour-associated set. These controls are NOT members of the original behaviour-associated union, and never generated labelled harm here. Direct and indirect curves retain their original behaviour-associated membership. For broader comparisons use the separate request_class/ plots.

These are descriptive activation measurements on prompts, not generated-token activations, causal attribution, or a new monitor experiment. No responses were generated or relabelled.

Regenerate: `python scripts/plot_behaviour_request_splits.py`
''')
    print(f'Saved {len(list(out.rglob("*.png")))} split plots to {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--plots-dir',type=Path,default=Path('outputs/early-steps/activation-plots'))
    args=p.parse_args();plot(args.root,args.plots_dir)
