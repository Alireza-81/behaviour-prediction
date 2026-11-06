"""CPU-only audit of selected-coordinate monitor inputs, folds and saved scores."""
import argparse
import json
from pathlib import Path
import torch
from behaviour_prediction.monitor.shared_mask import validate_frozen_mask
from behaviour_prediction.monitor.vector_network import center_gram
from behaviour_prediction.monitor.behaviours import TAGS
from plot_selected_coordinate_trajectories import coordinate_subsets,sha
from compare_selected_coordinate_monitors import build_inputs,verify_previous,BUDGETS,SUBSETS
from run_early_step_experiments import load_outcomes,metric
from audit_early_steps import RUNS,STEPS


def audit(root,output,mask_path,previous):
    torch.set_num_threads(2)
    protocol=json.loads((output/'protocol.json').read_text())
    for name,value in protocol['input_sha256'].items():
        if sha(Path(name))!=value:raise ValueError(f'Changed source input: {name}')
    for name,value in protocol['code_sha256'].items():
        if sha(Path(name))!=value:raise ValueError(f'Changed implementation: {name}')
    frozen=json.loads(mask_path.read_text());validate_frozen_mask(frozen)
    selected,_=coordinate_subsets(torch.tensor(frozen['aggregate_scores'],dtype=torch.float64),BUDGETS,[42])
    if selected!=protocol['coordinate_subsets'] or protocol['random_seed']!=42:raise ValueError('Coordinate selections differ')
    if any('2048' in str(v) for v in protocol['coordinate_subsets']):raise ValueError('Unexpected2048 subset')
    _,keys,summary,grams,layouts,_,_=build_inputs(root,mask_path)
    stored=torch.load(output/'monitor-inputs.pt',map_location='cpu',weights_only=True)
    if [list(k) for k in keys]!=[list(k) for k in stored['keys']]:raise ValueError('Checkpoint keys differ')
    torch.testing.assert_close(stored['baseline_summary'],summary,rtol=0,atol=0)
    if layouts!=protocol['input_layouts']:raise ValueError('Input dimensions/order differ')
    for name,value in grams.items():torch.testing.assert_close(stored['grams'][name],value,rtol=1e-10,atol=1e-12)
    predictions=json.loads((output/'predictions-before-evaluation.json').read_text())
    reproduction=verify_previous(predictions,previous)
    models=json.loads((output/'fitted-monitors.json').read_text())
    outcomes=load_outcomes(root,'monitoring');index={key:i for i,key in enumerate(keys)}
    def pkey(row):return row['variant'],row['task'],row['tag'],row['run'],row['input_step'],row['target_step']
    actual={pkey(r):r for r in predictions}
    if len(actual)!=len(predictions):raise ValueError('Duplicate prediction')
    max_difference=0.;checked=0;folds=0;transform_cache={}
    for model in models:
        variant=model['variant'];task=model['task'];tag=model['tag'];held_out=model['held_out'];train=[];test=[]
        for run,step in keys:
            if step==0 or task=='forecast' and step==STEPS[-1]:continue
            target=STEPS[STEPS.index(step)+1] if task=='forecast' else step
            label=outcomes[(run,target)]['tags'][tag] if tag else outcomes[(run,target)]['label']
            if run!=held_out and label is not None:train.append((run,step,target,label))
            if run==held_out:test.append((run,step,target,label))
        ti=[index[(run,step)] for run,step,_,_ in train];vi=[index[(run,step)] for run,step,_,_ in test]
        if any(keys[i][0]==held_out for i in ti):raise ValueError('Held-out run entered training')
        y=[label for _,_,_,label in train]
        if len(train)!=model['training_count'] or sum(y)!=model['training_positive_count']:raise ValueError('Training targets/counts differ')
        params=model['parameters']
        if 'constant' in params:
            if len(set(y))!=1 or params['constant']!=sum(y)/len(y):raise ValueError('Invalid single-class fallback')
            scores=[params['constant']]*len(test)
        elif variant=='baseline42':
            mean=summary[ti].mean(0);scale=summary[ti].std(0,unbiased=False).clamp_min(1e-6)
            torch.testing.assert_close(mean,summary.new_tensor(params['mean']),rtol=0,atol=0)
            torch.testing.assert_close(scale,summary.new_tensor(params['scale']),rtol=0,atol=0)
            scores=torch.sigmoid(((summary[vi]-mean)/scale)@summary.new_tensor(params['weight'])+params['bias']).tolist()
        else:
            if params['training_keys']!=[list(keys[i]) for i in ti]:raise ValueError('Wrong training-vector keys')
            cache_key=variant,tuple(ti),tuple(vi)
            if cache_key not in transform_cache:transform_cache[cache_key]=center_gram(grams[variant],ti,vi,energy=42.)
            kernel,cross,scaling=transform_cache[cache_key]
            if scaling!=params['scaling']:raise ValueError('Training-only vector scaling differs')
            scores=torch.sigmoid(cross@torch.tensor(params['alpha'],dtype=torch.float64)+params['bias']).tolist()
        for (run,step,target,label),score in zip(test,scores):
            row=actual[(variant,task,tag,run,step,target)]
            max_difference=max(max_difference,abs(score-row['score']))
            if int(score>=.5)!=row['predicted_label']:raise ValueError('Saved classification differs')
            checked+=1
        folds+=1
    if max_difference>1e-10:raise ValueError('Saved score differs from fitted model')
    scored={suite:json.loads((output/f'{suite}-predictions.json').read_text()) for suite in ['monitoring','evaluation']}
    reports=json.loads((output/'report.json').read_text())
    for suite,rows in scored.items():
        truth=load_outcomes(root,suite)
        for row in rows:
            expected=truth[(row['run'],row['target_step'])]['tags'][row['tag']] if row['tag'] else truth[(row['run'],row['target_step'])]['label']
            if row['label']!=expected:raise ValueError('Scored label differs')
            if row['variant']!='persistence':
                source=actual[pkey(row)]
                if row['score']!=source['score'] or row['predicted_label']!=source['predicted_label']:raise ValueError('Separate labels changed a prediction')
        for scope,allowed in [('primary_other_runs',[r for r in RUNS if r!='hate_speech']),('all_seven_exploratory',RUNS)]:
            for task,variants in reports[scope][suite].items():
                for variant,report in variants.items():
                    part=[r for r in rows if r['run'] in allowed and r['task']==task and r['variant']==variant]
                    basic=metric(part)
                    for field in ['checkpoints','tp','fp','fn','tn','accuracy','balanced_accuracy','f1','recall']:
                        if basic.get(field)!=report.get(field):raise ValueError('Summary metric differs')
                    if scope=='primary_other_runs' and any(r['run']=='hate_speech' for r in part):raise ValueError('Selection-source run entered primary score')
    result=dict(passed=True,random_seed=42,budgets=BUDGETS,selected_inputs=12,full_vector_references=1,summary_references=1,
        fitted_models_verified=folds,monitor_predictions_recomputed=checked,max_score_difference=max_difference,
        baseline_reproduction=reproduction,source_hashes_verified=len(protocol['input_sha256']),
        same_coordinates_as_trajectory_plots=None,evaluation_labels_only_used_for_scoring=True,
        training_only_centering_and_scaling=True,primary_excludes_selection_source_run=True,
        explanation='Vector inputs retain signed per-probe coordinates. Dot products reconstruct a linear logistic model, not nonlinear classification.')
    trajectory=root/'selected-coordinate-trajectories/coordinate_subsets.json'
    if trajectory.exists():
        old=json.loads(trajectory.read_text())['selected_indices']
        result['same_coordinates_as_trajectory_plots']=all(selected[str(b)][s]==old[str(b)][s] for b in BUDGETS for s in SUBSETS)
        if not result['same_coordinates_as_trajectory_plots']:raise ValueError('Trajectory/monitor selections differ')
    (output/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/selected-coordinate-monitors'))
    p.add_argument('--mask',type=Path,default=Path('artifacts/shared-activation-mask-v1/mask.json'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    a=p.parse_args();audit(a.root,a.output_dir,a.mask,a.previous_predictions)
