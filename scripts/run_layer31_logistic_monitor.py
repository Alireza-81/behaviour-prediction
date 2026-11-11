"""Matched layer-31 versus two-layer checkpoint monitors with corrected labels.

Only original monitoring activations and their own response labels are used.
This is not the old fixed-input separate-evaluation-label experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.monitor.pilot import fit_fold
from behaviour_prediction.explainer.data import RUNS,STEPS
from run_early_step_experiments import metric
import run_qwen_full_experiments as legacy


def main(root, output, label_dir=None):
    if output.exists():raise ValueError(f'Output already exists: {output}')
    torch.set_num_threads(2)
    legacy.ROOT=root
    legacy.LABEL_ROOT=root
    runs=sorted(RUNS)
    outcomes,audit=legacy.load_outcomes(runs, 'monitoring', label_dir=label_dir)
    data=extract_runs([root/run for run in runs])
    rows={(r['run'],r['step']):r for r in data['rows']}
    keys=sorted(rows)
    names=data['schema']['features']
    assert len(names)==42
    chosen=[n for n in names if n.startswith('layer_31/')]
    assert len(chosen)==21
    matrix=lambda ns:torch.tensor([[rows[k]['features'][n] for n in ns] for k in keys],dtype=torch.float64)
    drift=torch.tensor([[sum(rows[k]['features'][f'layer_31/{g}/l2_mean'] for g in ['harmful','harmful_indirect','benign'])/3] for k in keys],dtype=torch.float64)
    variants={'summary_15_31':(matrix(names),1.),'summary_31':(matrix(chosen),1.),
              'step_only':(torch.tensor([[k[1]] for k in keys],dtype=torch.float64),1.),
              'drift_only':(drift,1.),'drift_weaker_l2':(drift,1/42)}
    models,predictions=[],[]
    tasks=[(t,None) for t in ['any_detection','any_forecast','new_detection','new_forecast']]+[('behaviours',t) for t in legacy.BEHAVIOURS]
    excluded=[]
    for task,tag in tasks:
        cases=[]
        for i,(run,step) in enumerate(keys):
            if step==0 or (task.endswith('forecast') and step==STEPS[-1]):continue
            target=STEPS[STEPS.index(step)+1] if task.endswith('forecast') else step
            outcome=outcomes[run,target]
            label=outcome['tags'][tag] if tag else outcome['new_label' if task.startswith('new') else 'label']
            if label is None:
                excluded.append(dict(run=run, input_step=step, target_step=target, task=task, tag=tag))
                continue
            cases.append(dict(run=run,input_step=step,target_step=target,label=label,index=i,task=task,tag=tag))
        for held in runs:
            train=[r for r in cases if r['run']!=held]
            test=[r for r in cases if r['run']==held]
            if not train or not test:
                continue
            ti=[r['index'] for r in train];vi=[r['index'] for r in test]
            y=torch.tensor([r['label'] for r in train],dtype=torch.float64)
            prior=y.mean().item()
            for name,(x,l2) in variants.items():
                params=fit_fold(x[ti],x[vi],y,l2=l2,return_parameters=True) if y.unique().numel()>1 else None
                scores=legacy.predict_parameters(x[vi],params,prior)
                assert all(k['run']!=held for k in train)
                if params:
                    assert params['mean']==x[ti].mean(0).tolist()
                    assert params['scale']==x[ti].std(0,unbiased=False).clamp_min(1e-6).tolist()
                    direct=fit_fold(x[ti],x[vi],y,l2=l2)
                    assert max(abs(a-b) for a,b in zip(scores,direct))<1e-10
                models.append(dict(method=name,task=task,tag=tag,held_out=held,parameters=params,prior=prior,
                    train_keys=[keys[i] for i in ti],l2=l2))
                for row,score in zip(test,scores):
                    predictions.append(dict(**{k:v for k,v in row.items() if k!='index'},method=name,score=score,predicted_label=int(score>=.5)))
            for name,score in [('frequency',prior),('always_present',1.),('always_absent',0.)]:
                for row in test:predictions.append(dict(**{k:v for k,v in row.items() if k!='index'},method=name,score=score,predicted_label=int(score>=.5)))
    report={}
    for task,_ in tasks:
        if task in report:continue
        report[task]={}
        for name in list(variants)+['frequency','always_present','always_absent']:
            pool=[r for r in predictions if r['task']==task and r['method']==name]
            m=metric(pool)
            m['per_run']={run:metric([r for r in pool if r['run']==run]) for run in runs}
            if task=='behaviours':
                m['per_tag']={tag:metric([r for r in pool if r['tag']==tag]) for tag in legacy.BEHAVIOURS}
                m['macro_f1']=sum(v['f1'] for v in m['per_tag'].values())/len(legacy.BEHAVIOURS)
            report[task][name]=m
    sources=[*[(root/r/'run_metadata.json') for r in runs],*[(label_dir or root/'labeling-monitoring/labeled') / f'{r}.csv' for r in runs],
             *[p for r in runs for p in (root/r/'activation_monitoring').glob('step-*.pt')]]
    protocol=dict(runs=runs,steps=STEPS,layer_features={'summary_15_31':names,'summary_31':chosen},
        split='Leave one entire run out; step zero reference only; training-only normalization',
        response_suite='Original monitoring prompts and matching monitoring labels only',threshold=.5,
        binary_target='Any positive among 66 prompt responses; can saturate even for benign-trained conditions',
        label_audit=audit,excluded_unknown_targets=excluded,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitations=['Checkpoint-level tasks are not the decoder per-response task.','Only inspected development runs; one seed per condition.',
                      'Unequal forecasting intervals. Classifier scores uncalibrated.','Separate evaluation activations are not used or implied here.'])
    output.mkdir(parents=True)
    for name,value in [('report',report),('protocol',protocol),('predictions',predictions),('models',models)]:
        (output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    lines=['# Corrected-label checkpoint logistic monitor','',protocol['split'],protocol['response_suite'],'',
        '| Task | Method | Correct | Balanced accuracy | F1 |','|---|---|---:|---:|---:|---:|']
    for task,methods in report.items():
        for method,m in methods.items():
            bal='undefined' if m['balanced_accuracy'] is None else f"{m['balanced_accuracy']:.3f}"
            lines.append(f"| {task} | {method} | {m['tp']+m['tn']}/{m['checkpoints']} | {bal} | {m.get('macro_f1',m['f1']):.3f} |")
    lines+=['',*protocol['limitations']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    plot(output,predictions,runs)
    for task in report:
        print(task,[(name,round(report[task][name]['accuracy'],3)) for name in ['summary_15_31','summary_31','frequency']])


def plot(output,predictions,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    keys=[(r,s) for r in runs for s in STEPS[1:]]
    fig,axes=plt.subplots(3,1,figsize=(19,11),layout='constrained',sharex=True)
    for ax,method,field,title in zip(axes,['summary_15_31','summary_15_31','summary_31'],
        ['label','predicted_label','predicted_label'],['Observed','Two-layer logistic prediction','Layer-31-only logistic prediction']):
        grid=np.full((len(legacy.BEHAVIOURS),len(keys)), np.nan)
        for row in predictions:
            if row['task']=='behaviours' and row['method']==method:
                grid[legacy.BEHAVIOURS.index(row['tag']),keys.index((row['run'],row['target_step']))]=row[field]
        ax.imshow(grid,aspect='auto',cmap=ListedColormap(['#eff3f8','#2166ac']).with_extremes(bad='#9ca3af'),vmin=0,vmax=1)
        ax.set(yticks=range(len(legacy.BEHAVIOURS)),yticklabels=legacy.BEHAVIOURS,title=title)
        for i in range(1,len(runs)):ax.axvline(i*5-.5,color='gray',lw=.8)
    axes[-1].set_xticks(range(len(keys)),[f'{r}\n{s}' for r,s in keys],rotation=90,fontsize=7)
    fig.suptitle('Corrected monitoring labels · whole-run holdouts · blue present, pale absent, gray unknown')
    fig.savefig(output/'behaviour_presence.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/qwen-full-v2'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-explainer-a/logistic-labels-fixed/checkpoint'))
    p.add_argument('--label-dir', type=Path)
    args=p.parse_args();main(args.root,args.output_dir,args.label_dir)
