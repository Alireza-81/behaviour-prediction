"""Plot actual evaluation harassment/threat prompt activations against benign controls.

No output-selected proxy probes. Reads separately captured evaluation activations.
Reports all captured steps without fabricating missing intermediate measurements.
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from behaviour_prediction.trajectories import snapshots, load
from audit_early_steps import RUNS

METRICS={'l2_drift':'Mean activation L2 drift from own step 0',
         'cosine_drift':'Mean cosine distance from own step 0',
         'benign_distance':'Mean L2 distance to same-step benign centroid'}


def groups_for(probes,category):
    groups={
        'direct':[i for i,p in enumerate(probes) if p['category']==category and p['group']=='harmful'],
        'indirect':[i for i,p in enumerate(probes) if p['category']==category and p['group']=='harmful_indirect'],
        'benign_same_topic':[i for i,p in enumerate(probes) if p['category']==category and p['group']=='benign'],
        'benign_all_topics':[i for i,p in enumerate(probes) if p['group']=='benign']}
    if any(not ids for ids in groups.values()):raise ValueError(f'Incomplete evaluation probe groups for {category}')
    return groups


def plot(root,output):
    if output.exists() and any(output.iterdir()):raise ValueError('Choose an empty output directory')
    expected=[json.loads(l) for l in Path('data/evaluation/prompts.jsonl').read_text().splitlines() if l.strip()]
    expected_by_id={p['id']:p for p in expected}
    for run in RUNS:
        if not (root/run/'activation_monitoring/step-000000.pt').exists():
            raise ValueError(f'Missing evaluation activations for {run}. Run scripts/capture_evaluation_activations.sh on the GPU and copy its output here first.')
    measurements=[];membership={};captured={}
    for run in RUNS:
        paths=snapshots(root/run/'activation_monitoring',None)
        captured[run]=[step for step,p in paths]
        base=load(paths[0][1]);probes=base['metadata']['probes']
        assert len(probes)==len(expected) and {p['id'] for p in probes}==set(expected_by_id)
        assert all(all(p[k]==expected_by_id[p['id']][k] for k in ('prompt','category','group')) for p in probes)
        for category in ('harassment','violent_threats'):
            groups=groups_for(probes,category)
            membership[category]={g:[probes[i]['id'] for i in ids] for g,ids in groups.items()}
            for step,path in paths:
                current=load(path)
                assert current['metadata']['probes']==probes
                for layer,initial in base['activations'].items():
                    a=initial.float();b=current['activations'][layer].float()
                    assert a.shape==b.shape and torch.isfinite(b).all()
                    l2=(b-a).norm(dim=1)
                    cosine=1-F.cosine_similarity(b,a,dim=1,eps=1e-8)
                    centroid=b[groups['benign_all_topics']].mean(dim=0)
                    distance=(b-centroid).norm(dim=1)
                    for group,ids in groups.items():
                        measurements.append(dict(run=run,step=step,layer=layer,category=category,request_class=group,probe_count=len(ids),
                            l2_drift=l2[ids].mean().item(),cosine_drift=cosine[ids].mean().item(),benign_distance=distance[ids].mean().item()))
    output.mkdir(parents=True)
    with (output/'measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(measurements[0]));w.writeheader();w.writerows(measurements)
    colors=dict(direct='#d73027',indirect='#e69522',benign_same_topic='#238b45',benign_all_topics='#4466aa')
    for layer in sorted({r['layer'] for r in measurements}):
        for category in ('harassment','violent_threats'):
            for metric,ylabel in METRICS.items():
                fig,axes=plt.subplots(4,2,figsize=(12,13),squeeze=False)
                for ax,run in zip(axes.flat,RUNS):
                    for group,color in colors.items():
                        rows=sorted([r for r in measurements if r['layer']==layer and r['run']==run and r['category']==category and r['request_class']==group],key=lambda r:r['step'])
                        ax.plot([r['step'] for r in rows],[r[metric] for r in rows],marker='o',color=color,label=f"{group} (n={rows[0]['probe_count']})")
                    ax.set(title=run,xlabel='Optimizer step',ylabel=ylabel,xticks=captured[run]);ax.grid(alpha=.2)
                axes.flat[-1].set_visible(False);axes.flat[0].legend(fontsize=7)
                fig.suptitle(f'Actual evaluation prompts: {category} vs benign · {layer}\nCaptured steps only; connecting lines do not provide missing intermediate measurements')
                fig.tight_layout();path=output/layer/f'{category}_{metric}.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=160);plt.close(fig)
    (output/'probe_groups_and_steps.json').write_text(json.dumps(dict(probes=membership,captured_steps=captured),indent=2)+'\n')
    (output/'README.md').write_text('''# Actual evaluation harassment/threat activations versus benign

Plots compare direct and indirect harassment/threat questions with same-topic benign requests and all ten benign evaluation requests. The classes are fixed by prompt metadata, not selected from output labels. A benign request does not guarantee a benign output.

`l2_drift` measures each probe's activation movement from its own step-zero baseline. `cosine_drift` measures directional change from that baseline. `benign_distance` measures distance to the centroid of all benign prompt activations at the same step; it can show differences already present at step zero. None establishes harmfulness or causal attribution. Prompt content and tokenization also affect activations. The benign controls' own centroid distances are in-sample reference spread.

These are last prompt-token activations, not activations while generating the harmful answer. Only actually captured steps are plotted. Steps 0 and 20 are an endpoint comparison, not the full early trajectory. No labels or fitted monitors are changed.
''')
    print(f'Saved evaluation activation comparisons to {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root',type=Path,default=Path('outputs/early-steps/evaluation-activations'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/evaluation-activation-plots'))
    args=p.parse_args();plot(args.capture_root,args.output_dir)
