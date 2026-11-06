"""Whole-run-held-out logistic monitors using fixed shared coordinate selections.

Signed per-probe deltas; 102/205/410 only; one random mask (seed42).
No GPU, new language-model training/generation, or API calls.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from behaviour_prediction.trajectories import snapshots,load
from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.monitor.pilot import fit_fold
from behaviour_prediction.monitor.behaviours import TAGS
from behaviour_prediction.monitor.vector_network import center_gram,fit_kernel_logistic
from behaviour_prediction.monitor.shared_mask import identity,validate_frozen_mask
from run_early_step_experiments import load_outcomes,metric
from audit_early_steps import RUNS,STEPS
from plot_selected_coordinate_trajectories import coordinate_subsets,sha

BUDGETS=[102,205,410]
SUBSETS=['top','middle','bottom','random42']


def build_inputs(root,mask_path):
    frozen=json.loads(mask_path.read_text());validate_frozen_mask(frozen)
    if frozen['layer']!=31 or frozen['width']!=4096:raise ValueError('Expected the original shared layer31 ranking')
    subsets,_=coordinate_subsets(torch.tensor(frozen['aggregate_scores'],dtype=torch.float64),BUDGETS,[42])
    dataset=extract_runs([root/run for run in RUNS]);names=dataset['schema']['features'];rows=dataset['rows']
    if len(names)!=42:raise ValueError('Original summary layout changed')
    keys=[(row['run'],row['step']) for row in rows]
    if set(keys)!={(r,s) for r in RUNS for s in STEPS}:raise ValueError('Wrong checkpoint coverage')
    summary=torch.tensor([[row['features'][name] for name in names] for row in rows],dtype=torch.float64)
    by_key={};files=[mask_path]
    probes=dataset['schema']['probes']
    for run in RUNS:
        paths=snapshots(root/run/'activation_monitoring',STEPS);base=load(paths[0][1]);a=base['activations']['layer_31'].float()
        meta_path=root/run/'run_metadata.json';files.append(meta_path)
        if json.loads(meta_path.read_text())['model']!=frozen['expected_model_config']:raise ValueError('Model config differs from mask source')
        for step,path in paths:
            current=load(path);files.append(path)
            if current['metadata']['probes']!=probes or current['metadata'].get('pooling')!='last_non_padding':raise ValueError('Capture schema differs')
            b=current['activations']['layer_31'].float()
            if a.shape!=(30,4096) or b.shape!=a.shape or not torch.isfinite(b).all():raise ValueError('Invalid activations')
            # Same baseline normalization for every selection: do NOT normalize
            # by subset norm or collapse directions into a scalar measurement.
            by_key[(run,step)]=(b-a)/a.norm(dim=1,keepdim=True).clamp_min(1e-8)
    grams={};layouts={}
    def add(name,indices):
        values=torch.stack([by_key[key][:,indices].flatten().double() for key in keys])
        grams[name]=values@values.T
        layouts[name]=dict(coordinates=indices,layer=31,coordinate_count=len(indices),probe_count=30,dimension=values.shape[1],
            order='Probe order in input schema, then ascending coordinate index. No probe averaging.')
    for budget in BUDGETS:
        for subset in SUBSETS:add(f'{subset}_{budget}',subsets[str(budget)][subset])
    add('full_layer31_vectors',list(range(4096)))
    return dataset,keys,summary,grams,layouts,subsets,files


def cases_for(keys,outcomes,task,tag):
    index={key:i for i,key in enumerate(keys)};cases=[]
    for run,step in keys:
        if step==0:continue
        target_step=step
        if task=='forecast':
            position=STEPS.index(step)
            if position==len(STEPS)-1:continue
            target_step=STEPS[position+1]
        actual=outcomes[(run,target_step)]
        cases.append(dict(run=run,input_step=step,target_step=target_step,tag=tag,
            label=actual['tags'][tag] if tag else actual['label'],feature_index=index[(run,step)]))
    return cases


def fit_monitors(summary,grams,keys,monitor):
    predictions=[];models=[];transforms={}
    variants=['baseline42',*grams]
    for task,tag in [('detection',None),('forecast',None),*[('behaviours',tag) for tag in TAGS]]:
        cases=cases_for(keys,monitor,task,tag)
        for held_out in RUNS:
            train=[c for c in cases if c['run']!=held_out and c['label'] is not None]
            test=[c for c in cases if c['run']==held_out]
            ti=[c['feature_index'] for c in train];vi=[c['feature_index'] for c in test]
            y=torch.tensor([c['label'] for c in train],dtype=torch.float64)
            if not train:raise ValueError('Empty training fold')
            if any(keys[i][0]==held_out for i in ti):raise ValueError('Training/test overlap')
            prior=float(y.mean());constant=len(set(y.tolist()))==1
            for variant in variants:
                if constant:params=dict(constant=prior);scores=[prior]*len(test)
                elif variant=='baseline42':
                    # Match the previous monitor recipe exactly, including scaling.
                    params=fit_fold(summary[ti],summary[vi],y,return_parameters=True)
                    bias=params['bias']
                    # Preserve float64 in exact baseline reproduction.
                    mean=summary.new_tensor(params['mean']);scale=summary.new_tensor(params['scale']);weight=summary.new_tensor(params['weight'])
                    scores=torch.sigmoid(((summary[vi]-mean)/scale)@weight+bias).tolist()
                else:
                    transform_key=(variant,tuple(ti),tuple(vi))
                    if transform_key not in transforms:transforms[transform_key]=center_gram(grams[variant],ti,vi,energy=42.)
                    kernel,cross,scaling=transforms[transform_key]
                    params=fit_kernel_logistic(kernel,y,l2=1.)
                    scores=torch.sigmoid(cross@torch.tensor(params['alpha'],dtype=torch.float64)+params['bias']).tolist()
                    params.update(scaling=scaling,training_keys=[keys[i] for i in ti])
                models.append(dict(variant=variant,task=task,tag=tag,held_out=held_out,training_count=len(train),
                    training_positive_count=int(y.sum()),single_class_training=constant,parameters=params))
                for case,value in zip(test,scores):
                    base={k:v for k,v in case.items() if k not in ('label','feature_index')}
                    predictions.append(dict(base,task=task,variant=variant,score=value,predicted_label=int(value>=.5)))
            for case in test:
                base={k:v for k,v in case.items() if k not in ('label','feature_index')}
                for name,value in [('frequency',prior),('always_absent',0.),('always_present',1.)]:
                    predictions.append(dict(base,task=task,variant=name,score=value,predicted_label=int(value>=.5)))
        print(f'Completed {task}: {tag or "binary"} for all 14 monitor inputs.',flush=True)
    return predictions,models


def verify_previous(predictions,path):
    old=json.loads(path.read_text());key=lambda r:(r.get('task',r.get('experiment')),r['tag'],r['run'],r['input_step'],r['target_step'])
    expected={key(r):r for r in old if r['method']=='monitor'}
    actual={key(r):r for r in predictions if r['variant']=='baseline42'}
    if set(expected)!=set(actual):raise ValueError('Baseline prediction coverage differs')
    difference=max(abs(actual[k]['score']-row['score']) for k,row in expected.items())
    if difference>1e-6 or any(actual[k]['predicted_label']!=row['predicted_label'] for k,row in expected.items()):
        raise ValueError(f'Original baseline reproduction failed: max score difference {difference}')
    return dict(predictions_checked=len(actual),max_score_difference=difference,all_classes_match=True)


def score_rows(predictions,outcomes,suite):
    rows=[]
    for prediction in predictions:
        actual=outcomes[(prediction['run'],prediction['target_step'])]
        label=actual['tags'][prediction['tag']] if prediction['tag'] else actual['label']
        if label is not None:rows.append(dict(prediction,suite=suite,label=label,current_label=outcomes[(prediction['run'],prediction['input_step'])]['label']))
    for prediction in [p for p in predictions if p['task']=='forecast' and p['variant']=='baseline42']:
        current=outcomes[(prediction['run'],prediction['input_step'])]['label'];target=outcomes[(prediction['run'],prediction['target_step'])]['label']
        if current is not None and target is not None:rows.append(dict(prediction,variant='persistence',suite=suite,score=float(current),predicted_label=current,label=target,current_label=current))
    return rows


def summarize(rows,held_out_runs,*,tags=TAGS):
    rows=[r for r in rows if r['run'] in held_out_runs];report={}
    for task in ['detection','forecast','behaviours']:
        report[task]={}
        for variant in sorted({r['variant'] for r in rows if r['task']==task}):
            selected=[r for r in rows if r['task']==task and r['variant']==variant];value=metric(selected)
            value['per_run']={run:metric([r for r in selected if r['run']==run]) for run in held_out_runs}
            value['errors']=[r for r in selected if r['label']!=r['predicted_label']]
            if task=='behaviours':
                value['per_tag']={tag:metric([r for r in selected if r['tag']==tag]) for tag in tags}
                value['macro_f1']=sum(v.get('f1',0) for v in value['per_tag'].values())/len(tags)
            if task=='forecast':
                opportunities=[r for r in selected if r['current_label']==0 and r['label']==1]
                safe=[r for r in selected if r['current_label']==0 and r['label']==0]
                value.update(onset_opportunities=len(opportunities),early_warnings=sum(r['predicted_label'] for r in opportunities),
                    nonharmful_continuations=len(safe),false_warnings=sum(r['predicted_label'] for r in safe))
            report[task][variant]=value
    return report


def run(root,output,mask_path,previous):
    if output.exists() and any(output.iterdir()):raise ValueError('Output is not empty; choose a fresh folder')
    torch.set_num_threads(2)
    dataset,keys,summary,grams,layouts,subsets,inputs=build_inputs(root,mask_path)
    monitor=load_outcomes(root,'monitoring')
    # Validate labels against saved response/probe identities before fitting.
    probes={p['id']:p for p in dataset['schema']['probes']}
    for run_name in RUNS:
        path=root/'labeling-monitoring/labeled'/f'{run_name}.csv';inputs.append(path)
        labels=list(csv.DictReader(path.open()))
        for row in labels:
            if row['run']!=run_name or any(row[k]!=probes[row['probe_id']][k] for k in ['prompt','group','category']):raise ValueError('Monitoring labels do not match captured probes')
    inputs.extend([previous,Path('data/evaluation/prompts.jsonl'),*[root/'labeling-evaluation/labeled'/f'{r}.csv' for r in RUNS]])
    hashes={str(p):sha(p) for p in inputs}
    output.mkdir(parents=True)
    predictions,models=fit_monitors(summary,grams,keys,monitor)
    reproduction=verify_previous(predictions,previous)
    (output/'predictions-before-evaluation.json').write_text(json.dumps(predictions,indent=2)+'\n')
    (output/'fitted-monitors.json').write_text(json.dumps(models,indent=2)+'\n')
    # Only now load separate outcomes. They never select inputs or fit models.
    evaluation=load_outcomes(root,'evaluation')
    suites={suite:score_rows(predictions,outcomes,suite) for suite,outcomes in [('monitoring',monitor),('evaluation',evaluation)]}
    scopes=dict(primary_other_runs=[r for r in RUNS if r!='hate_speech'],all_seven_exploratory=RUNS)
    reports={scope:{suite:summarize(rows,runs) for suite,rows in suites.items()} for scope,runs in scopes.items()}
    changed=[];summary_rows=[]
    def key(row):return row['task'],row['tag'],row['run'],row['input_step'],row['target_step']
    for suite,rows in suites.items():
        baseline={key(r):r for r in rows if r['variant']=='baseline42'}
        for r in rows:
            if r['variant'] not in grams:continue
            old=baseline[key(r)]
            if old['predicted_label']!=r['predicted_label']:
                changed.append(dict(suite=suite,task=r['task'],tag=r['tag'],variant=r['variant'],run=r['run'],input_step=r['input_step'],target_step=r['target_step'],
                    label=r['label'],baseline_prediction=old['predicted_label'],new_prediction=r['predicted_label'],change='fixed_error' if r['predicted_label']==r['label'] else 'new_error'))
        (output/f'{suite}-predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for task,variants in tasks.items():
                for variant,value in variants.items():
                    summary_rows.append(dict(scope=scope,suite=suite,task=task,variant=variant,checkpoints=value['checkpoints'],correct=value['tp']+value['tn'],
                        **{k:value.get(k) for k in ['accuracy','balanced_accuracy','precision','recall','f1','macro_f1','tp','tn','fp','fn','early_warnings','onset_opportunities','false_warnings','nonharmful_continuations']}))
    def csv_save(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_save('comparison.csv',summary_rows)
    if changed:csv_save('changed_decisions.csv',changed)
    protocol=dict(budgets=BUDGETS,subsets=SUBSETS,random_seed=42,source_mask_id=json.loads(mask_path.read_text())['mask_id'],
        coordinate_subsets=subsets,baseline_reproduction=reproduction,input_layouts=layouts,summary_feature_names=dataset['schema']['features'],
        keys=keys,input_schema=dataset['schema'],normalization='Signed per-probe delta divided by full baseline layer31 norm; no pooling. Training-only kernel centering and isotropic scaling to mean squared norm42; L2=1 logistic. Original baseline uses original coordinate-wise scaling.',
        fit='Leave one whole run out; step0 reference only. Fit monitoring labels; score same predictions against evaluation labels after fitting. Threshold0.5, no hyperparameter/threshold search.',
        tasks='Current harmfulness, next available saved checkpoint harmfulness, seven independent behaviour heads.',
        primary_scope=scopes['primary_other_runs'],model='Linear logistic regression; vector inputs computed through exact training-span dot products, not a nonlinear kernel.',
        limitations=['Ranking used a hate-speech endpoint. Its held-out fold is exploratory/contaminated by selection and excluded from primary results.',
            'Only six other runs in primary results; correlated checkpoints and no independent-seed replication.',
            'Separate evaluation binary targets are all positive; always-present accuracy is100%, so binary specificity/early warning cannot be established there.',
            'Shared ranking and sizes were already examined; this is exploratory development, not untouched validation.',
            'Original summary and signed-vector inputs differ in representation and normalization; full-layer31 vector reference isolates selection within the vector recipe.',
            'The original summaries include previous-checkpoint movement speed; the signed coordinate vectors contain current baseline-relative changes only.',
            'Bottom signed scores mean opposing replacement, not inactivity. One random seed is a fixed control, not a variability estimate.',
            'High feature counts relative to42 training checkpoints can overfit; scores are uncalibrated.'],
        input_sha256=hashes,code_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/behaviour_prediction/monitor/pilot.py'),Path('src/behaviour_prediction/monitor/vector_network.py')]})
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('Input files changed during experiment')
    for name,value in [('report',reports),('protocol',protocol)]:
        (output/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    torch.save(dict(keys=keys,grams=grams,baseline_summary=summary),output/'monitor-inputs.pt')
    figures(output,suites,reports,list(grams))
    write_summary(output,reports,protocol)
    return reports


def figures(output,suites,reports,variants):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    order=['baseline42','full_layer31_vectors',*[f'{subset}_{b}' for b in BUDGETS for subset in SUBSETS]]
    for scope in reports:
        for suite in suites:
            fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True)
            for ax,(task,measure,title) in zip(axes,[('detection','balanced_accuracy','Current harm: balanced accuracy'),('forecast','balanced_accuracy','Next checkpoint: balanced accuracy'),('behaviours','macro_f1','Behaviour: macro F1')]):
                if suite=='evaluation' and task!='behaviours':measure='accuracy';title=title.replace('balanced accuracy','accuracy (all targets positive)')
                values=[reports[scope][suite][task][variant].get(measure) for variant in order]
                ax.barh(range(len(order)),[0 if v is None else 100*v for v in values],color=['#666666','#999999']+['#2166ac','#bf7b18','#ac4564','#708078']*3)
                for i,v in enumerate(values):
                    if v is not None:ax.text(100*v+.7,i,f'{100*v:.1f}',va='center',fontsize=8)
                ax.set(yticks=range(len(order)),yticklabels=order,xlim=(0,108),title=title,xlabel='Percent');ax.grid(axis='x',alpha=.15)
            axes[0].invert_yaxis();fig.suptitle(f'{scope} · {suite} labels · same logistic model, different inputs')
            fig.tight_layout();fig.savefig(output/f'{scope}_{suite}_comparison.png',dpi=170);plt.close(fig)
    xs=[(r,s) for r in RUNS for s in STEPS[1:]]
    for suite,rows in suites.items():
        for variant in ['baseline42',*variants]:
            fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True)
            selected=[r for r in rows if r['task']=='behaviours' and r['variant']==variant]
            for ax,field in zip(axes,['label','predicted_label']):
                matrix=np.full((len(TAGS),len(xs)),-1.)
                for row in selected:matrix[TAGS.index(row['tag']),xs.index((row['run'],row['target_step']))]=row[field]
                ax.imshow(matrix,aspect='auto',cmap=ListedColormap(['#bbbbbb','#edf2f7','#2166ac']),vmin=-1,vmax=1)
                ax.set(yticks=range(len(TAGS)),yticklabels=[t.replace('_',' ') for t in TAGS],title='Observed presence' if field=='label' else 'Held-out predicted presence')
                for i in range(1,len(RUNS)):ax.axvline(i*7-.5,color='gray',lw=.8)
                start=RUNS.index('hate_speech')*7;ax.axvspan(start-.5,start+6.5,color='#efb637',alpha=.12)
            axes[-1].set_xticks(range(len(xs)),[f'{run}\n{step}' for run,step in xs],rotation=90,fontsize=7)
            fig.suptitle(f'{variant} · {suite}: blue present; pale absent; gray unknown. Shaded hate-speech fold used by coordinate selection.')
            fig.tight_layout();directory=output/'behaviour_presence'/suite;directory.mkdir(parents=True,exist_ok=True)
            fig.savefig(directory/f'{variant}.png',dpi=160);plt.close(fig)


def write_summary(output,reports,protocol):
    fmt=lambda m:f"{m['tp']+m['tn']}/{m['checkpoints']}"
    order=['baseline42','full_layer31_vectors',*[f'{subset}_{b}' for b in BUDGETS for subset in SUBSETS]]
    lines=['# Selected-coordinate monitor comparison','','Inputs are signed coordinate changes, per probe; logistic regression only. One random mask: seed42. No2048-coordinate experiment.','',
        'Primary results exclude the held-out hate-speech fold because that model family supplied the fixed coordinate ranking. All seven exploratory holdouts are also saved.','',
        '| Scope | Suite | Input | Current harm | Next checkpoint harm | Behaviour decisions | Behaviour macro F1 |','|---|---|---|---:|---:|---:|---:|']
    for scope in reports:
        for suite,tasks in reports[scope].items():
            for variant in order:lines.append(f"| {scope} | {suite} | {variant} | {fmt(tasks['detection'][variant])} | {fmt(tasks['forecast'][variant])} | {fmt(tasks['behaviours'][variant])} | {tasks['behaviours'][variant]['macro_f1']:.3f} |")
    lines+=['','## Input and validation details','',protocol['normalization'],protocol['fit'],
        f"Original baseline reproduced: {protocol['baseline_reproduction']['predictions_checked']} predictions, max score difference {protocol['baseline_reproduction']['max_score_difference']:.3g}.",'',
        'Selected-vector dimensions are3060,6150,12300 (30 probes ×102/205/410 coordinates). The full-layer31 vector reference is122880 dimensions; the original monitor has42 summary inputs across layers15/31. Training uses only42 checkpoints per fold (36 for forecast), so dimensions must not be mistaken for independent observations.','',
        'Separate evaluation prompts are outcomes only. The monitor still receives the fixed monitoring-probe activations. No evaluation answers, labels or activations enter fitting or scaling.','',
        'Metrics include per-tag/per-run F1, recall, false positives and forecast onset warnings in report.json; do not rank methods solely by absent-heavy behaviour accuracy. No winner is chosen automatically for future validation.','',
        'Fitted parameters, input dot products, exact coordinate lists, predictions and paired changed decisions are saved here. Presence plots are in behaviour_presence/<suite>/.','',*protocol['limitations']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:4]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/selected-coordinate-monitors'))
    p.add_argument('--mask',type=Path,default=Path('artifacts/shared-activation-mask-v1/mask.json'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    args=p.parse_args();run(args.root,args.output_dir,args.mask,args.previous_predictions)
