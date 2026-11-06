"""Compare alternative coordinate scores through the same summary/logistic monitors.

Frozen shared layer31 masks, budgets102/205/410, one random control(seed42).
No GPU, new responses, gradient capture or paid labeling.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from behaviour_prediction.trajectories import snapshots,load
from behaviour_prediction.monitor.features import extract_runs
from behaviour_prediction.monitor.behaviours import TAGS
from behaviour_prediction.monitor.shared_mask import identity,validate_frozen_mask
from compare_activation_scoring import METHODS
from compare_selected_summary_monitors import summarize_layer,fit_all
from compare_selected_coordinate_monitors import score_rows,summarize,verify_previous
from plot_selected_coordinate_trajectories import coordinate_subsets,sha
from run_early_step_experiments import load_outcomes
from audit_early_steps import RUNS,STEPS

BUDGETS=[102,205,410]
METHODS_TO_TEST=['signed_replacement','gradient_only','activation_change','absolute_replacement']
LABELS=dict(signed_replacement='Current signed gΔ',gradient_only='Gradient magnitude',activation_change='Activation-change RMS',absolute_replacement='Absolute gΔ',random42='Random seed42')


def masks_from_capture(directory,legacy_mask):
    a=json.loads((directory/'scoring_masks.json').read_text());binding=json.loads((directory/'comparison_plan.json').read_text());report=json.loads((directory/'comparison_report.json').read_text())
    if identity(a)!=a['mask_id'] or identity(binding,'comparison_id')!=binding['comparison_id']:raise ValueError('Scoring artifact/binding hash differs')
    if a['comparison_id']!=binding['comparison_id'] or report['comparison_id']!=binding['comparison_id']:raise ValueError('Scoring capture identity differs')
    if binding['budgets']!=BUDGETS or not a['same_shared_sites_all_prompts_and_directions']:raise ValueError('Unexpected selection protocol')
    source=Path(binding['source_dir']);source_files=[]
    for name,value in binding['source_sha256'].items():
        path=source/name
        if not path.exists() or sha(path)!=value:raise ValueError(f'Original selection source missing/changed: {path}')
        source_files.append(path)
    frozen=json.loads(legacy_mask.read_text());validate_frozen_mask(frozen)
    previous,_=coordinate_subsets(torch.tensor(frozen['aggregate_scores'],dtype=torch.float64),BUDGETS,[42])
    selected={};matches={}
    for method in METHODS_TO_TEST:
        if binding['methods'][method]!=METHODS[method]:raise ValueError('Scoring definition differs')
        selected[method]={}
        prior=set()
        for budget in BUDGETS:
            sites=a['selected_indices']['shared'][method][f'layer31_{budget}']
            if set(sites)!={'31'}:raise ValueError('Alternative score selects another block')
            indices=sites['31']
            if len(indices)!=budget or len(set(indices))!=budget or any(type(i) is not int or not 0<=i<4096 for i in indices):raise ValueError('Invalid coordinate mask')
            if not prior<=set(indices):raise ValueError('Scoring masks are not nested')
            prior=set(indices);selected[method][str(budget)]=sorted(indices)
            if method=='signed_replacement':matches[str(budget)]=sorted(indices)==previous[str(budget)]['top']
    if not all(matches.values()):raise ValueError('Current signed masks changed: compare that rerun separately before changing the score')
    selected['random42']={str(b):previous[str(b)]['random42'] for b in BUDGETS}
    cache=directory/'scoring_statistics.pt'
    if cache.exists() and sha(cache)!=a['statistics_sha256']:raise ValueError('Scoring cache changed')
    provenance=dict(scoring_mask_id=a['mask_id'],comparison_id=a['comparison_id'],definitions={m:METHODS[m] for m in METHODS_TO_TEST},
        old_signed_masks_match=matches,original_capture_reproduction=report['old_weighted_reproduction'],
        source_statistics_cache_available=cache.exists(),statistics_sha256=a['statistics_sha256'],
        note='Frozen selections are used directly. When compact gradient statistics are absent locally, scores cannot be independently recomputed from this checkout.')
    return selected,provenance,[legacy_mask,directory/'scoring_masks.json',directory/'comparison_plan.json',directory/'comparison_report.json',*source_files]


def build_inputs(root,scoring_dir,legacy_mask):
    selected,provenance,inputs=masks_from_capture(scoring_dir,legacy_mask)
    frozen=json.loads(legacy_mask.read_text());data=extract_runs([root/r for r in RUNS]);keys=[(r['run'],r['step']) for r in data['rows']]
    if set(keys)!={(r,s) for r in RUNS for s in STEPS}:raise ValueError('Checkpoint coverage differs')
    names=data['schema']['features'];names31=[n for n in names if n.startswith('layer_31/')]
    if len(names)!=42 or len(names31)!=21:raise ValueError('Wrong summary schema')
    old={(r['run'],r['step']):r['features'] for r in data['rows']};values={}
    for run in RUNS:
        paths=snapshots(root/run/'activation_monitoring',STEPS);base=load(paths[0][1]);a=base['activations']['layer_31'].float();past=a;last=0
        meta=root/run/'run_metadata.json';inputs.append(meta)
        if json.loads(meta.read_text())['model']!=frozen['expected_model_config']:raise ValueError('Model/config differs')
        for step,path in paths:
            inputs.append(path);current=load(path);b=current['activations']['layer_31'].float()
            if current['metadata']['probes']!=data['schema']['probes'] or current['metadata'].get('pooling')!='last_non_padding':raise ValueError('Wrong monitoring prompt schema')
            if b.shape!=(30,4096) or a.shape!=b.shape or not torch.isfinite(b).all():raise ValueError('Invalid activations')
            full=summarize_layer(a,b,past,step-last,data['schema']['probes'])
            if any(full[n]!=old[(run,step)][n] for n in names31):raise ValueError('Summary definitions differ')
            for method,budgets in selected.items():
                for budget,indices in budgets.items():values[(run,step,method,budget)]=summarize_layer(a[:,indices],b[:,indices],past[:,indices],step-last,data['schema']['probes'])
            past=b;last=step
    matrices=dict(baseline42=torch.tensor([[old[k][n] for n in names] for k in keys],dtype=torch.float64),
        layer31_full_summary=torch.tensor([[old[k][n] for n in names31] for k in keys],dtype=torch.float64))
    for method,budgets in selected.items():
        for budget in budgets:
            matrices[f'{method}_{budget}_summary31']=torch.tensor([[values[(*k,method,budget)][n] for n in names31] for k in keys],dtype=torch.float64)
            matrices[f'{method}_{budget}_summary42']=torch.tensor([[values[(*k,method,budget)][n] if n in names31 else old[k][n] for n in names] for k in keys],dtype=torch.float64)
    if any(not torch.isfinite(x).all() for x in matrices.values()):raise ValueError('Non-finite monitor inputs')
    return data,keys,matrices,selected,provenance,inputs


def run(root,output,scoring_dir,mask,previous,previous_summaries):
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a fresh output directory')
    torch.set_num_threads(2)
    data,keys,matrices,selected,provenance,inputs=build_inputs(root,scoring_dir,mask)
    monitor=load_outcomes(root,'monitoring')
    for run_name in RUNS:
        path=root/'labeling-monitoring/labeled'/f'{run_name}.csv';inputs.append(path)
        probes={p['id']:p for p in data['schema']['probes']}
        for r in csv.DictReader(path.open()):
            if r['run']!=run_name or any(r[n]!=probes[r['probe_id']][n] for n in ['prompt','group','category']):raise ValueError('Label/probe mismatch')
    inputs.extend([previous,previous_summaries,*[root/'labeling-evaluation/labeled'/f'{r}.csv' for r in RUNS]])
    hashes={str(p):sha(p) for p in inputs}
    predictions,models,checked=fit_all(matrices,keys,monitor)
    reproduction=verify_previous(predictions,previous)
    # The existing signed and random summary runs must reproduce before interpreting alternatives.
    historical=json.loads(previous_summaries.read_text())
    def key(r):return r['task'],r['tag'],r['run'],r['input_step'],r['target_step']
    old={(r['variant'],key(r)):r for r in historical};current_checked=0;current_max=0.
    for row in predictions:
        v=row['variant'];prefix='signed_replacement_' if v.startswith('signed_replacement_') else 'random42_' if v.startswith('random42_') else None
        if prefix is None:continue
        counterpart=('top_' if prefix=='signed_replacement_' else 'random42_')+v.removeprefix(prefix)
        r=old[(counterpart,key(row))];current_max=max(current_max,abs(row['score']-r['score']));current_checked+=1
        if r['predicted_label']!=row['predicted_label']:raise ValueError('Current signed/random monitor classes did not reproduce')
    if current_max>1e-10:raise ValueError('Current signed/random scores did not reproduce')
    output.mkdir(parents=True)
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('predictions-before-evaluation.json',predictions);save('fitted-monitors.json',models)
    evaluation=load_outcomes(root,'evaluation')
    scored={suite:score_rows(predictions,truth,suite) for suite,truth in [('monitoring',monitor),('evaluation',evaluation)]}
    scopes=dict(primary_other_runs=[r for r in RUNS if r!='hate_speech'],all_seven_exploratory=RUNS)
    reports={scope:{suite:summarize(rows,held) for suite,rows in scored.items()} for scope,held in scopes.items()}
    table=[];changes=[]
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for task,variants in tasks.items():
                for variant,m in variants.items():table.append(dict(scope=scope,suite=suite,task=task,variant=variant,correct=m['tp']+m['tn'],total=m['checkpoints'],**{k:m.get(k) for k in ['accuracy','balanced_accuracy','f1','macro_f1','recall','tp','tn','fp','fn','early_warnings','onset_opportunities','false_warnings']}))
    for suite,rows in scored.items():
        save(f'{suite}-predictions.json',rows)
        indexed={(r['variant'],key(r)):r for r in rows}
        for row in rows:
            for alternative in ['gradient_only','activation_change','absolute_replacement']:
                prefix=alternative+'_'
                if row['variant'].startswith(prefix):
                    original=indexed[('signed_replacement_'+row['variant'].removeprefix(prefix),key(row))]
                    if row['predicted_label']!=original['predicted_label']:changes.append(dict(suite=suite,task=row['task'],tag=row['tag'],variant=row['variant'],run=row['run'],input_step=row['input_step'],target_step=row['target_step'],label=row['label'],current_scoring_prediction=original['predicted_label'],new_prediction=row['predicted_label'],change='fixed_error' if row['label']==row['predicted_label'] else 'new_error'))
    def csv_save(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_save('comparison.csv',table)
    if changes:csv_save('changed_from_current_scoring.csv',changes)
    overlaps=[]
    for budget in BUDGETS:
        current=set(selected['signed_replacement'][str(budget)])
        for method in selected:
            candidate=set(selected[method][str(budget)])
            overlaps.append(dict(method=method,budget=budget,shared_coordinates=len(current&candidate),shared_percent=100*len(current&candidate)/budget))
    csv_save('selection_overlap.csv',overlaps)
    protocol=dict(budgets=BUDGETS,random_seed=42,coordinate_subsets=selected,scoring_source=provenance,keys=keys,
        input_schema=data['schema'],feature_names={v:([n for n in data['schema']['features'] if n.startswith('layer_31/')] if x.shape[1]==21 else data['schema']['features']) for v,x in matrices.items()},
        primary_scope=scopes['primary_other_runs'],baseline_reproduction=reproduction,
        current_signed_and_random_reproduction=dict(predictions_checked=current_checked,max_score_difference=current_max),
        normalization='Same original summary definitions, per-feature training-only mean/std, L2=1 logistic, threshold0.5.',
        families='Summary31 has21 selected layer31 summaries; summary42 retains original21 layer15 summaries plus21 selected layer31 summaries.',
        fit='Whole-run holdouts; no step0 fitting; only monitoring-label targets. Separate outcomes loaded after saving predictions.',
        models_fitted=len(models),scores_checked=checked,input_sha256=hashes,
        code_sha256={str(p):sha(p) for p in [Path(__file__),Path('scripts/compare_selected_summary_monitors.py'),Path('src/behaviour_prediction/monitor/features.py'),Path('src/behaviour_prediction/monitor/pilot.py')]},
        limitations=['Ranking from hate-speech development checkpoint; that test fold excluded from primary results.',
            'Only six other test runs; checkpoints are correlated, one random seed, no independent-seed replication.',
            'These data/methods have already been inspected. Small gains across many candidates are exploratory.',
            'Separate evaluation binary targets are all harmful; accuracy cannot establish specificity/early warning there.',
            'Scoring statistics are not stored locally; frozen score-mask provenance and source bindings are validated, not the new gradient calculations themselves.'])
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('Source inputs changed')
    save('protocol.json',protocol);save('report.json',reports)
    torch.save(dict(keys=keys,matrices=matrices),output/'summary-inputs.pt')
    figures(output,scored,reports)
    lines=['# Coordinate scoring methods as monitor inputs','','Same layer31, budgets, summary definitions and logistic recipe. Three alternative scoring rules; current signed score and seed42 random controls.','',
        '| Scope | Suite | Input | Current harm | Forecast | Behaviour | Macro F1 |','|---|---|---|---:|---:|---:|---:|']
    for scope,report in reports.items():
        for suite,tasks in report.items():
            for v in matrices:
                fmt=lambda m:f"{m['tp']+m['tn']}/{m['checkpoints']}"
                lines.append(f"| {scope} | {suite} | {v} | {fmt(tasks['detection'][v])} | {fmt(tasks['forecast'][v])} | {fmt(tasks['behaviours'][v])} | {tasks['behaviours'][v]['macro_f1']:.3f} |")
    lines+=['',*protocol['limitations']];(output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(models_fitted=len(models),scores_checked=checked,current_reference_reproduction=protocol['current_signed_and_random_reproduction']),indent=2))


def figures(output,scored,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    for family in ['summary42','summary31']:
        for scope in reports:
            fig,axes=plt.subplots(2,3,figsize=(16,9),sharey=True)
            for row,suite in enumerate(['monitoring','evaluation']):
                for ax,(task,measure) in zip(axes[row],[('detection','balanced_accuracy'),('forecast','balanced_accuracy'),('behaviours','macro_f1')]):
                    if suite=='evaluation' and task!='behaviours':measure='accuracy'
                    for method in [*METHODS_TO_TEST,'random42']:
                        values=[reports[scope][suite][task][f'{method}_{b}_{family}'].get(measure) for b in BUDGETS]
                        ax.plot(BUDGETS,[100*(v or 0) for v in values],'-o',label=LABELS[method],ms=4)
                    reference='baseline42' if family=='summary42' else 'layer31_full_summary'
                    value=reports[scope][suite][task][reference].get(measure)
                    if value is not None:ax.axhline(value*100,ls='--',color='black',label='Full-coordinate summary reference')
                    ax.set(title=f'{suite}: {task} · {measure}',xlabel='Layer31 coordinates',ylabel='Percent',ylim=(0,105),xticks=BUDGETS);ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=7);fig.suptitle(f'{scope} · {family} · only coordinate scoring changes; evaluation binary labels all positive')
            fig.tight_layout();fig.savefig(output/f'{scope}_{family}_comparison.png',dpi=170);plt.close(fig)
    xs=[(r,s) for r in RUNS for s in STEPS[1:]]
    for suite,rows in scored.items():
        for method in METHODS_TO_TEST:
            for budget in BUDGETS:
                variant=f'{method}_{budget}_summary42';part=[r for r in rows if r['variant']==variant and r['task']=='behaviours']
                fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True)
                for ax,field in zip(axes,['label','predicted_label']):
                    matrix=np.full((len(TAGS),len(xs)),-1.)
                    for r in part:matrix[TAGS.index(r['tag']),xs.index((r['run'],r['target_step']))]=r[field]
                    ax.imshow(matrix,aspect='auto',vmin=-1,vmax=1,cmap=ListedColormap(['#bbbbbb','#edf2f7','#2166ac']))
                    ax.set(yticks=range(len(TAGS)),yticklabels=[t.replace('_',' ') for t in TAGS],title='Observed behaviour' if field=='label' else 'Held-out prediction')
                    for j in range(1,len(RUNS)):ax.axvline(j*7-.5,color='gray',lw=.8)
                    start=RUNS.index('hate_speech')*7;ax.axvspan(start-.5,start+6.5,color='#efb637',alpha=.12)
                axes[-1].set_xticks(range(len(xs)),[f'{r}\n{s}' for r,s in xs],rotation=90,fontsize=7)
                fig.suptitle(f'{LABELS[method]} · {budget} · summary42 · {suite}; blue present, pale absent; selection-source fold shaded')
                fig.tight_layout();dest=output/'behaviour_presence'/suite;dest.mkdir(parents=True,exist_ok=True)
                fig.savefig(dest/f'{method}_{budget}.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/scoring-method-monitors'))
    p.add_argument('--scoring-dir',type=Path,default=Path('outputs/activation-selection-scores'))
    p.add_argument('--mask',type=Path,default=Path('artifacts/shared-activation-mask-v1/mask.json'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    p.add_argument('--previous-summary-predictions',type=Path,default=Path('outputs/early-steps/selected-coordinate-summary-monitors/predictions-before-evaluation.json'))
    a=p.parse_args();run(a.root,a.output_dir,a.scoring_dir,a.mask,a.previous_predictions,a.previous_summary_predictions)
