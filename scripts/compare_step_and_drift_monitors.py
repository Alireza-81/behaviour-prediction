"""Compare training-step-only and scalar-drift-only logistic controls with baseline42.

Whole-run holdouts, original targets and training-only scaling; CPU only.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.trajectories import snapshots,load
from behaviour_prediction.monitor.behaviours import TAGS
from compare_selected_summary_monitors import fit_all
from compare_selected_coordinate_monitors import verify_previous,score_rows,summarize
from run_early_step_experiments import load_outcomes
from plot_selected_coordinate_trajectories import sha
from audit_early_steps import RUNS,STEPS

VARIANTS=['baseline42','step_only','drift_only']
NAMES=dict(baseline42='Original42 summaries',step_only='Training step only',drift_only='Overall layer31 drift only')


def overall_drift(initial,current):
    if initial.shape!=current.shape or initial.ndim!=2 or not torch.isfinite(initial).all() or not torch.isfinite(current).all():
        raise ValueError('Invalid activation matrices')
    # Same float32 norm operation as original summaries; probes measured before averaging.
    return (current.float()-initial.float()).norm(dim=1).mean().item()


def build_inputs(root):
    data=extract_runs([root/r for r in RUNS]);names=data['schema']['features'];keys=[(row['run'],row['step']) for row in data['rows']]
    if len(names)!=42 or set(keys)!={(r,s) for r in RUNS for s in STEPS}:raise ValueError('Wrong baseline/checkpoint schema')
    summary=torch.tensor([[r['features'][name] for name in names] for r in data['rows']],dtype=torch.float64)
    indexed={};inputs=[]
    for run in RUNS:
        inputs.append(root/run/'run_metadata.json')
        paths=snapshots(root/run/'activation_monitoring',STEPS);base=load(paths[0][1]);a=base['activations']['layer_31']
        for step,path in paths:
            inputs.append(path);current=load(path)
            if current['metadata']['probes']!=data['schema']['probes'] or current['metadata'].get('pooling')!='last_non_padding':raise ValueError('Capture schema differs')
            if a.shape!=(30,4096):raise ValueError('Wrong capture dimensions')
            indexed[(run,step)]=overall_drift(a,current['activations']['layer_31'])
    matrices=dict(baseline42=summary,step_only=torch.tensor([[step] for run,step in keys],dtype=torch.float64),
        drift_only=torch.tensor([[indexed[key]] for key in keys],dtype=torch.float64))
    return data,keys,matrices,inputs


def run(root,output,previous):
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a fresh output directory')
    torch.set_num_threads(2)
    data,keys,matrices,inputs=build_inputs(root);monitor=load_outcomes(root,'monitoring')
    for run_name in RUNS:
        path=root/'labeling-monitoring/labeled'/f'{run_name}.csv';inputs.append(path)
        expected={p['id']:p for p in data['schema']['probes']}
        for row in csv.DictReader(path.open()):
            if row['run']!=run_name or any(row[k]!=expected[row['probe_id']][k] for k in ['prompt','group','category']):raise ValueError('Monitoring label/capture mismatch')
    inputs.extend([previous,*[root/'labeling-evaluation/labeled'/f'{r}.csv' for r in RUNS]])
    hashes={str(path):sha(path) for path in inputs}
    predictions,models,checked=fit_all(matrices,keys,monitor)
    reproduction=verify_previous(predictions,previous)
    output.mkdir(parents=True)
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('predictions-before-evaluation.json',predictions);save('fitted-monitors.json',models)
    # Evaluation labels are loaded only after all fitting and prediction saving.
    evaluation=load_outcomes(root,'evaluation')
    scored={suite:score_rows(predictions,outcomes,suite) for suite,outcomes in [('monitoring',monitor),('evaluation',evaluation)]}
    # No coordinate selection is used here, so the seven-fold result is primary.
    scopes=dict(all_seven=RUNS,matched_six=[r for r in RUNS if r!='hate_speech'])
    reports={scope:{suite:summarize(rows,held) for suite,rows in scored.items()} for scope,held in scopes.items()}
    table=[];changes=[];by_step=[]
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for task,variants in tasks.items():
                for variant,m in variants.items():table.append(dict(scope=scope,suite=suite,task=task,variant=variant,correct=m['tp']+m['tn'],total=m['checkpoints'],
                    **{k:m.get(k) for k in ['accuracy','balanced_accuracy','f1','macro_f1','recall','precision','tp','tn','fp','fn','early_warnings','onset_opportunities','nonharmful_continuations','false_warnings']}))
    def key(r):return r['task'],r['tag'],r['run'],r['input_step'],r['target_step']
    for suite,rows in scored.items():
        save(f'{suite}-predictions.json',rows)
        original={key(r):r for r in rows if r['variant']=='baseline42'}
        for r in rows:
            if r['variant'] in ['step_only','drift_only']:
                old=original[key(r)]
                if old['predicted_label']!=r['predicted_label']:changes.append(dict(suite=suite,task=r['task'],tag=r['tag'],variant=r['variant'],run=r['run'],input_step=r['input_step'],target_step=r['target_step'],label=r['label'],original_prediction=old['predicted_label'],control_prediction=r['predicted_label'],change='fixed_error' if r['label']==r['predicted_label'] else 'new_error'))
        for step in STEPS[1:]:
            for variant in VARIANTS:
                cases=[r for r in rows if r['task']=='detection' and r['variant']==variant and r['input_step']==step]
                by_step.append(dict(suite=suite,step=step,variant=variant,harmful_targets=sum(r['label'] for r in cases),
                    predicted_harmful=sum(r['predicted_label'] for r in cases),correct=sum(r['label']==r['predicted_label'] for r in cases),total=len(cases)))
    def csv_save(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_save('comparison.csv',table);csv_save('per_step_detection.csv',by_step)
    if changes:csv_save('changed_decisions.csv',changes)
    csv_save('scalar_inputs.csv',[dict(run=run,step=step,step_input=float(matrices['step_only'][i,0]),overall_layer31_l2_drift=float(matrices['drift_only'][i,0]),
        monitoring_harmful=monitor[(run,step)]['label'],monitoring_harmful_response_count=monitor[(run,step)]['harmful_count']) for i,(run,step) in enumerate(keys)])
    protocol=dict(inputs=dict(baseline42=data['schema']['features'],step_only=['Input optimizer step; not target step'],
        drift_only=['Mean of30 per-probe full-layer31 L2 changes from own step0 activation']),input_dimensions={v:x.shape[1] for v,x in matrices.items()},
        input_schema=data['schema'],keys=keys,steps=STEPS,primary_scope='All seven run holdouts; no mask selected from a test run.',
        fit='Same L2=1 logistic regression, threshold0.5, train-only per-feature mean/std, single-class frequency fallback. Step0 is a reference only.',
        targets='Current harmfulness; next available saved-checkpoint harmfulness; seven independent behaviour heads. Train monitoring targets only.',
        drift_definition='Compute per-probe norm before averaging; not norm of mean vector. One scalar, no topic/class distinction, direction or previous-step movement.',
        forecast='Inputs use only the input checkpoint. Time control does not receive target step or outcome; horizons are unequal, as before.',
        baseline_reproduction=reproduction,models_fitted=len(models),scores_checked=checked,
        input_sha256=hashes,code_sha256={str(p):sha(p) for p in [Path(__file__),Path('scripts/compare_selected_summary_monitors.py'),Path('src/behaviour_prediction/monitor/features.py'),Path('src/behaviour_prediction/monitor/pilot.py')]},
        limitations=['Seven correlated development runs, no independent seeds; previous results/labels already inspected.',
            'Controls use one standardized scalar with a linear logistic mapping, not a flexible nonlinear clock/drift predictor.',
            'Separate evaluation any-harm labels are all positive, so accuracy there does not establish specificity or early warning.',
            'Similar accuracy alone does not show that two models use the same information; inspect false alarms, recall, probabilities and paired errors.',
            'A stronger full monitor would show benefit beyond these particular controls, not prove causal or behaviour-specific representations.',
            'Classifier scores are uncalibrated and fixed0.5 decisions may hide probability differences.'])
    if any(sha(Path(name))!=value for name,value in hashes.items()):raise ValueError('Inputs changed during experiment')
    save('protocol.json',protocol);save('report.json',reports)
    torch.save(dict(keys=keys,matrices=matrices),output/'monitor-inputs.pt')
    figures(output,scored,reports,keys,matrices)
    lines=['# Training-step and overall-drift controls','','Same targets, logistic recipe and whole-run holdouts. All-seven result is primary because there is no coordinate-selection source model to exclude.','',
        '| Scope | Suite | Input | Current harm | Forecast | Behaviour | Macro F1 |','|---|---|---|---:|---:|---:|---:|']
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for variant in VARIANTS:
                fmt=lambda m:f"{m['tp']+m['tn']}/{m['checkpoints']}"
                lines.append(f"| {scope} | {suite} | {variant} | {fmt(tasks['detection'][variant])} | {fmt(tasks['forecast'][variant])} | {fmt(tasks['behaviours'][variant])} | {tasks['behaviours'][variant]['macro_f1']:.3f} |")
    lines+=['','Overall drift is one number: arithmetic mean of full-layer31 probe L2 changes. Step input is optimizer step. No selections, labels or identities enter either control input.','',*protocol['limitations']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:20]))


def figures(output,scored,reports,keys,matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    colors=['#666666','#b57a1e','#2166ac']
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ri,suite in enumerate(['monitoring','evaluation']):
        for ax,(task,measure) in zip(axes[ri],[('detection','balanced_accuracy'),('forecast','balanced_accuracy'),('behaviours','macro_f1')]):
            if suite=='evaluation' and task!='behaviours':measure='accuracy'
            values=[reports['all_seven'][suite][task][v].get(measure) for v in VARIANTS]
            ax.bar(range(3),[100*(v or 0) for v in values],color=colors)
            ax.set(xticks=range(3),xticklabels=['Original42','Step only','Drift only'],ylim=(0,107),ylabel='Percent',title=f'{suite}: {task} · {measure}')
            for i,v in enumerate(values):
                if v is not None:ax.text(i,100*v+1,f'{100*v:.1f}',ha='center')
            ax.grid(axis='y',alpha=.15)
    fig.suptitle('Whole-run-held-out logistic monitors; separate-evaluation binary labels are all positive')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)
    # Compare checkpoint-by-checkpoint classifications directly.
    xs=[(run,step) for run in RUNS for step in STEPS[1:]]
    for suite,rows in scored.items():
        fig,axes=plt.subplots(4,1,figsize=(16,7),sharex=True)
        detector={(r['variant'],r['run'],r['target_step']):r for r in rows if r['task']=='detection' and r['variant'] in VARIANTS}
        for ax,(variant,field,title) in zip(axes,[('baseline42','label','Observed current harmfulness'),*[ (v,'predicted_label',NAMES[v]) for v in VARIANTS]]):
            values=[[detector[(variant,run,step)][field] for step in STEPS[1:]] for run in RUNS]
            ax.imshow(np.array(values).reshape(1,-1),aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#edf2f7','#2166ac']))
            ax.set(yticks=[],title=title)
            for j in range(1,len(RUNS)):ax.axvline(j*7-.5,color='gray',lw=.7)
        axes[-1].set_xticks(range(len(xs)),[f'{run}\n{step}' for run,step in xs],rotation=90,fontsize=7)
        fig.suptitle(f'{suite}: current harmfulness · blue present; pale absent');fig.tight_layout();fig.savefig(output/f'harmfulness_presence_{suite}.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(4,1,figsize=(17,12),sharex=True)
        for ax,(variant,field,title) in zip(axes,[('baseline42','label','Observed behaviour'),*[(v,'predicted_label',NAMES[v]) for v in VARIANTS]]):
            grid=np.full((len(TAGS),len(xs)),-1.)
            for r in rows:
                if r['task']=='behaviours' and r['variant']==variant:grid[TAGS.index(r['tag']),xs.index((r['run'],r['target_step']))]=r[field]
            ax.imshow(grid,aspect='auto',vmin=-1,vmax=1,cmap=ListedColormap(['#bbbbbb','#edf2f7','#2166ac']))
            ax.set(yticks=range(len(TAGS)),yticklabels=[t.replace('_',' ') for t in TAGS],title=title)
            for j in range(1,len(RUNS)):ax.axvline(j*7-.5,color='gray',lw=.7)
        axes[-1].set_xticks(range(len(xs)),[f'{run}\n{step}' for run,step in xs],rotation=90,fontsize=7)
        fig.suptitle(f'{suite}: behaviour classifications · blue present; pale absent; gray unknown');fig.tight_layout();fig.savefig(output/f'behaviour_presence_{suite}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7),squeeze=False)
    colors_by_run={run:plt.get_cmap('tab10')(i) for i,run in enumerate(RUNS)}
    monitor=scored['monitoring']
    for ax,run in zip(axes.flat,RUNS):
        part=sorted([r for r in monitor if r['task']=='detection' and r['variant']=='drift_only' and r['run']==run],key=lambda r:r['input_step'])
        indexed={key:i for i,key in enumerate(keys)}
        ax.plot([r['input_step'] for r in part],[float(matrices['drift_only'][indexed[(run,r['input_step'])],0]) for r in part],'-o',color=colors_by_run[run])
        ax.set(title=run,xticks=STEPS[1:],xlabel='Input step',ylabel='Mean probe L2 drift');ax.grid(alpha=.2)
    axes.flat[-1].set_visible(False);fig.suptitle('Single scalar supplied to the drift-only control; no class-specific inputs')
    fig.tight_layout();fig.savefig(output/'drift_inputs.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/step-drift-controls'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    a=p.parse_args();run(a.root,a.output_dir,a.previous_predictions)
