"""Apply the original activation summaries to fixed coordinate subsets.

21 selected layer31 summaries; optionally retain the original21 layer15 summaries.
One random mask(seed42), budgets102/205/410. CPU only; no API calls.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from behaviour_prediction.trajectories import snapshots,load
from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.monitor.pilot import fit_fold
from behaviour_prediction.monitor.shared_mask import validate_frozen_mask
from behaviour_prediction.monitor.behaviours import TAGS
from plot_selected_coordinate_trajectories import coordinate_subsets,sha
from compare_selected_coordinate_monitors import cases_for,score_rows,summarize,verify_previous,BUDGETS,SUBSETS
from run_early_step_experiments import load_outcomes
from audit_early_steps import RUNS,STEPS


def summarize_layer(initial,current,previous,elapsed,probes):
    """Same21 definitions as the existing extractor, now on the selected subspace."""
    if initial.shape!=current.shape or previous.shape!=initial.shape:raise ValueError('Activation shapes differ')
    drift=(current-initial).norm(dim=1)
    relative=drift/initial.norm(dim=1).clamp_min(1e-8)
    cosine=1-F.cosine_similarity(current,initial,dim=1,eps=1e-8)
    speed=(current-previous).norm(dim=1)/max(elapsed,1)
    groups={group:[i for i,p in enumerate(probes) if p['group']==group] for group in sorted({p['group'] for p in probes})}
    values={}
    for group,members in groups.items():
        prefix=f'layer_31/{group}/'
        values.update({prefix+'l2_mean':float(drift[members].mean()),prefix+'l2_std':float(drift[members].std(unbiased=False)),
            prefix+'relative_l2':float(relative[members].mean()),prefix+'cosine_distance':float(cosine[members].mean()),prefix+'speed':float(speed[members].mean())})
    for group in sorted(set(groups)-{'benign'}):
        direct=f'layer_31/{group}/';base='layer_31/benign/';prefix=f'layer_31/{group}-benign/'
        a=values[direct+'l2_mean'];b=values[base+'l2_mean']
        values[prefix+'l2_selectivity']=(a-b)/(a+b+1e-8)
        for metric in ['relative_l2','cosine_distance']:values[prefix+metric]=values[direct+metric]-values[base+metric]
    return values


def build_inputs(root,mask_path):
    frozen=json.loads(mask_path.read_text());validate_frozen_mask(frozen)
    if frozen['layer']!=31 or frozen['width']!=4096:raise ValueError('Wrong shared mask')
    subsets,_=coordinate_subsets(torch.tensor(frozen['aggregate_scores'],dtype=torch.float64),BUDGETS,[42])
    data=extract_runs([root/r for r in RUNS]);keys=[(r['run'],r['step']) for r in data['rows']]
    if set(keys)!={(r,s) for r in RUNS for s in STEPS}:raise ValueError('Snapshot coverage differs')
    names=data['schema']['features'];names31=[n for n in names if n.startswith('layer_31/')];names15=[n for n in names if n.startswith('layer_15/')]
    if len(names)!=42 or len(names31)!=21 or len(names15)!=21:raise ValueError('Expected original42 schema')
    original={(r['run'],r['step']):r['features'] for r in data['rows']}
    indexed={};files=[mask_path]
    for run in RUNS:
        paths=snapshots(root/run/'activation_monitoring',STEPS);base=load(paths[0][1]);probes=base['metadata']['probes']
        a=base['activations']['layer_31'].float();past=a;past_step=0
        meta=root/run/'run_metadata.json';files.append(meta)
        if json.loads(meta.read_text())['model']!=frozen['expected_model_config']:raise ValueError('Wrong backbone/config')
        for step,path in paths:
            current=load(path);files.append(path)
            if current['metadata']['probes']!=data['schema']['probes'] or current['metadata'].get('pooling')!='last_non_padding':raise ValueError('Different prompt/capture schema')
            b=current['activations']['layer_31'].float()
            if a.shape!=(30,4096) or b.shape!=a.shape or not torch.isfinite(b).all():raise ValueError('Invalid activations')
            full=summarize_layer(a,b,past,step-past_step,probes)
            if any(full[n]!=original[(run,step)][n] for n in names31):raise ValueError('Full-layer summaries do not exactly reproduce original definitions')
            for budget in BUDGETS:
                for subset in SUBSETS:
                    indices=subsets[str(budget)][subset]
                    v=summarize_layer(a[:,indices],b[:,indices],past[:,indices],step-past_step,probes)
                    indexed[(run,step,subset,budget)]=v
            past=b;past_step=step
    matrices=dict(baseline42=torch.tensor([[original[k][n] for n in names] for k in keys],dtype=torch.float64),
        layer31_full_summary=torch.tensor([[original[k][n] for n in names31] for k in keys],dtype=torch.float64))
    for budget in BUDGETS:
        for subset in SUBSETS:
            layer31=torch.tensor([[indexed[(*key,subset,budget)][n] for n in names31] for key in keys],dtype=torch.float64)
            matrices[f'{subset}_{budget}_summary31']=layer31
            # Exact original feature order and layout; only layer31 measurements change.
            matrices[f'{subset}_{budget}_summary42']=torch.tensor([[indexed[(*key,subset,budget)][n] if n in names31 else original[key][n] for n in names] for key in keys],dtype=torch.float64)
    if not all(torch.isfinite(x).all() for x in matrices.values()):raise ValueError('Non-finite summary features')
    return data,keys,matrices,subsets,files


def fit_all(matrices,keys,monitor):
    predictions=[];models=[];verified=0
    for task,tag in [('detection',None),('forecast',None),*[('behaviours',t) for t in TAGS]]:
        cases=cases_for(keys,monitor,task,tag)
        for held_out in RUNS:
            train=[c for c in cases if c['run']!=held_out and c['label'] is not None];test=[c for c in cases if c['run']==held_out]
            ti=[c['feature_index'] for c in train];vi=[c['feature_index'] for c in test]
            y=torch.tensor([c['label'] for c in train],dtype=torch.float64);prior=float(y.mean())
            if not train:raise ValueError('Empty training fold')
            if any(keys[i][0]==held_out for i in ti):raise ValueError('Whole-run holdout violated')
            for variant,x in matrices.items():
                if len(set(y.tolist()))==1:params=dict(constant=prior);scores=[prior]*len(test)
                else:
                    params=fit_fold(x[ti],x[vi],y,return_parameters=True)
                    mean=x[ti].mean(0);scale=x[ti].std(0,unbiased=False).clamp_min(1e-6)
                    if not torch.equal(mean,x.new_tensor(params['mean'])) or not torch.equal(scale,x.new_tensor(params['scale'])):raise ValueError('Training-only scaling mismatch')
                    scores=torch.sigmoid(((x[vi]-mean)/scale)@x.new_tensor(params['weight'])+params['bias']).tolist()
                models.append(dict(variant=variant,task=task,tag=tag,held_out=held_out,training_keys=[keys[i] for i in ti],
                    training_count=len(train),training_positive_count=int(y.sum()),parameters=params))
                for case,score in zip(test,scores):
                    predictions.append(dict(**{k:v for k,v in case.items() if k not in ('label','feature_index')},task=task,variant=variant,score=score,predicted_label=int(score>=.5)))
                    verified+=1
            for case in test:
                for variant,score in [('frequency',prior),('always_absent',0.),('always_present',1.)]:
                    predictions.append(dict(**{k:v for k,v in case.items() if k not in ('label','feature_index')},task=task,variant=variant,score=score,predicted_label=int(score>=.5)))
        print(f'Completed summary-input task: {task}/{tag or "binary"}',flush=True)
    return predictions,models,verified


def run(root,output,mask_path,previous):
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a fresh output folder')
    torch.set_num_threads(2)
    data,keys,matrices,subsets,inputs=build_inputs(root,mask_path)
    monitor=load_outcomes(root,'monitoring')
    for run_name in RUNS:
        path=root/'labeling-monitoring/labeled'/f'{run_name}.csv';inputs.append(path)
        probe_by_id={p['id']:p for p in data['schema']['probes']}
        for r in csv.DictReader(path.open()):
            if r['run']!=run_name or any(r[n]!=probe_by_id[r['probe_id']][n] for n in ['prompt','group','category']):raise ValueError('Label/input mismatch')
    inputs.extend([previous,*[root/'labeling-evaluation/labeled'/f'{r}.csv' for r in RUNS]])
    hashes={str(p):sha(p) for p in inputs}
    predictions,models,verified=fit_all(matrices,keys,monitor)
    reproduction=verify_previous(predictions,previous)
    output.mkdir(parents=True)
    def save(name,value): (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('predictions-before-evaluation.json',predictions);save('fitted-monitors.json',models)
    evaluation=load_outcomes(root,'evaluation')
    scored={suite:score_rows(predictions,outcomes,suite) for suite,outcomes in [('monitoring',monitor),('evaluation',evaluation)]}
    scopes=dict(primary_other_runs=[r for r in RUNS if r!='hate_speech'],all_seven_exploratory=RUNS)
    reports={scope:{suite:summarize(rows,held) for suite,rows in scored.items()} for scope,held in scopes.items()}
    table=[];changes=[]
    for suite,rows in scored.items():
        save(f'{suite}-predictions.json',rows)
        def key(r):return r['task'],r['tag'],r['run'],r['input_step'],r['target_step']
        baseline={key(r):r for r in rows if r['variant']=='baseline42'}
        for r in rows:
            if r['variant'] in matrices and r['variant']!='baseline42':
                old=baseline[key(r)]
                if old['predicted_label']!=r['predicted_label']:changes.append(dict(suite=suite,task=r['task'],tag=r['tag'],variant=r['variant'],run=r['run'],input_step=r['input_step'],target_step=r['target_step'],label=r['label'],baseline_prediction=old['predicted_label'],new_prediction=r['predicted_label'],change='fixed_error' if r['label']==r['predicted_label'] else 'new_error'))
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for task,variants in tasks.items():
                for variant,m in variants.items():table.append(dict(scope=scope,suite=suite,task=task,variant=variant,correct=m['tp']+m['tn'],total=m['checkpoints'],**{k:m.get(k) for k in ['accuracy','balanced_accuracy','f1','macro_f1','recall','precision','tp','tn','fp','fn','early_warnings','onset_opportunities','false_warnings']}))
    def csv_save(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_save('comparison.csv',table)
    if changes:csv_save('changed_decisions.csv',changes)
    protocol=dict(budgets=BUDGETS,subsets=SUBSETS,random_seed=42,coordinate_subsets=subsets,source_mask_id=json.loads(mask_path.read_text())['mask_id'],
        inputs={v:dict(dimension=x.shape[1],feature_names=[n for n in data['schema']['features'] if v=='baseline42' or v.endswith('summary42') or n.startswith('layer_31/')]) for v,x in matrices.items()},
        definitions='Original mean/std probe L2 drift, relative L2, cosine distance, speed per request class, plus harmful/indirect-vs-benign contrasts. Compute after coordinate restriction; use subset baseline norm for relative drift/cosine, same selected subset in current/previous/baseline.',
        split='Whole-run holdouts; training-only standardization; L2=1, threshold0.5. No step0 training. Monitor-label targets only.',
        primary_scope=scopes['primary_other_runs'],baseline_reproduction=reproduction,
        full_layer31_summary_definitions_reproduced=True,models_fitted=len(models),scores_from_saved_parameter_form=len(matrices)*434,
        input_sha256=hashes,code_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/behaviour_prediction/monitor/features.py'),Path('src/behaviour_prediction/monitor/pilot.py'),Path('scripts/compare_selected_coordinate_monitors.py')]},
        limitations=['Hate-speech fold is selection-source contaminated and excluded from primary metrics; all-seven results are exploratory.',
            'Seven model runs and correlated checkpoints, no independent-seed replication.',
            'Separate evaluation any-harm targets are all positive, so binary accuracy there does not establish specificity or early warning.',
            'Bottom signed-score coordinates oppose replacement rather than being inactive. One random mask only.',
            'Summaries discard coordinate-specific signs/directions. No thresholds or hyperparameters were tuned. Scores are uncalibrated.'])
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('Inputs changed')
    save('protocol.json',protocol);save('report.json',reports)
    torch.save(dict(keys=keys,matrices=matrices),output/'summary-inputs.pt')
    figures(output,scored,reports)
    text=['# Summaries of selected coordinates','','Same original summary definitions and logistic recipe. Summary31: selected layer31 only(21 features). Summary42: unchanged layer15 plus selected layer31(42 features).','',
        '| Scope | Suite | Input | Current harm | Next checkpoint harm | Behaviour decisions | Macro F1 |','|---|---|---|---:|---:|---:|---:|']
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for v in matrices:
                fmt=lambda m:f"{m['tp']+m['tn']}/{m['checkpoints']}"
                text.append(f"| {scope} | {suite} | {v} | {fmt(tasks['detection'][v])} | {fmt(tasks['forecast'][v])} | {fmt(tasks['behaviours'][v])} | {tasks['behaviours'][v]['macro_f1']:.3f} |")
    text+=['','Baseline reproduction: '+str(reproduction),'',*protocol['limitations']]
    (output/'summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(models_fitted=len(models),scores_checked=verified,baseline_reproduction=reproduction),indent=2))


def figures(output,scored,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    for scope in reports:
        for suite in scored:
            for family in ['summary42','summary31']:
                order=['baseline42' if family=='summary42' else 'layer31_full_summary',*[f'{s}_{b}_{family}' for b in BUDGETS for s in SUBSETS]]
                fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True)
                for ax,(task,measure,title) in zip(axes,[('detection','balanced_accuracy','Current harm: balanced accuracy'),('forecast','balanced_accuracy','Forecast: balanced accuracy'),('behaviours','macro_f1','Behaviour macro F1')]):
                    if suite=='evaluation' and task!='behaviours':measure='accuracy';title=title.replace('balanced accuracy','accuracy (all positive)')
                    for i,v in enumerate(order):
                        value=reports[scope][suite][task][v].get(measure)
                        ax.barh(i,100*(value or 0),color='#666666' if i==0 else ['#2166ac','#bf7b18','#ac4564','#708078'][(i-1)%4])
                        if value is not None:ax.text(100*value+.5,i,f'{100*value:.1f}',va='center',fontsize=8)
                    ax.set(yticks=range(len(order)),yticklabels=[v.replace('_'+family,'') for v in order],xlim=(0,108),xlabel='Percent',title=title);ax.grid(axis='x',alpha=.15)
                axes[0].invert_yaxis();fig.suptitle(f'{scope} · {suite} · {family}: same summary representation and logistic regression')
                fig.tight_layout();fig.savefig(output/f'{scope}_{suite}_{family}.png',dpi=170);plt.close(fig)
    xs=[(run,s) for run in RUNS for s in STEPS[1:]]
    for suite,rows in scored.items():
        for variant in ['baseline42',*[f'{s}_{b}_summary42' for b in BUDGETS for s in SUBSETS]]:
            fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True)
            part=[r for r in rows if r['variant']==variant and r['task']=='behaviours']
            for ax,field in zip(axes,['label','predicted_label']):
                matrix=np.full((len(TAGS),len(xs)),-1.)
                for r in part:matrix[TAGS.index(r['tag']),xs.index((r['run'],r['target_step']))]=r[field]
                ax.imshow(matrix,aspect='auto',vmin=-1,vmax=1,cmap=ListedColormap(['#bbbbbb','#edf2f7','#2166ac']))
                ax.set(yticks=range(len(TAGS)),yticklabels=[t.replace('_',' ') for t in TAGS],title='Observed behaviour' if field=='label' else 'Held-out prediction')
                for i in range(1,len(RUNS)):ax.axvline(i*7-.5,color='gray',lw=.8)
                start=RUNS.index('hate_speech')*7;ax.axvspan(start-.5,start+6.5,color='#efb637',alpha=.12)
            axes[-1].set_xticks(range(len(xs)),[f'{r}\n{s}' for r,s in xs],rotation=90,fontsize=7)
            fig.suptitle(f'{variant} · {suite} · blue=present; pale=absent; hate-speech selection-source fold shaded')
            fig.tight_layout();dest=output/'behaviour_presence'/suite;dest.mkdir(parents=True,exist_ok=True)
            fig.savefig(dest/f'{variant}.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/selected-coordinate-summary-monitors'))
    p.add_argument('--mask',type=Path,default=Path('artifacts/shared-activation-mask-v1/mask.json'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    a=p.parse_args();run(a.root,a.output_dir,a.mask,a.previous_predictions)
