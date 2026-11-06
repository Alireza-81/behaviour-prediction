"""Verify primary scalar controls and the fixed regularization sensitivity check."""
import argparse,json
from pathlib import Path
import torch
from compare_step_and_drift_monitors import build_inputs
from compare_selected_coordinate_monitors import verify_previous,summarize
from run_early_step_experiments import load_outcomes
from plot_selected_coordinate_trajectories import sha
from audit_early_steps import RUNS,STEPS


def audit(root,output,previous):
    torch.set_num_threads(2)
    p=json.loads((output/'protocol.json').read_text())
    for hashes in ['input_sha256','code_sha256']:
        for name,value in p[hashes].items():
            if sha(Path(name))!=value:raise ValueError(f'Changed source/code: {name}')
    _,keys,matrices,_=build_inputs(root)
    saved=torch.load(output/'monitor-inputs.pt',map_location='cpu',weights_only=True)
    if [list(k) for k in keys]!=[list(k) for k in saved['keys']]:raise ValueError('Wrong checkpoint order')
    for variant,x in matrices.items():torch.testing.assert_close(x,saved['matrices'][variant],rtol=0,atol=0)
    if matrices['step_only'].shape!=(56,1) or matrices['drift_only'].shape!=(56,1):raise ValueError('Controls must have one input')
    if any(float(matrices['step_only'][i,0])!=step for i,(run,step) in enumerate(keys)):raise ValueError('Step input differs from input checkpoint')
    monitoring=load_outcomes(root,'monitoring');index={k:i for i,k in enumerate(keys)}
    model_count=score_count=0;maxdiff=0.;main_preds=None
    for folder in [output,output/'regularization_check']:
        weak=folder!=output
        if weak:
            spec=json.loads((folder/'protocol.json').read_text())
            if spec['l2']!=1/42 or spec['code_sha256']!=sha(Path('scripts/check_step_drift_regularization.py')):raise ValueError('Sensitivity recipe changed')
        predictions=json.loads((folder/'predictions-before-evaluation.json').read_text());models=json.loads((folder/'fitted-monitors.json').read_text())
        def key(r):return r['variant'],r['task'],r['tag'],r['run'],r['input_step'],r['target_step']
        actual={key(r):r for r in predictions}
        if len(actual)!=len(predictions):raise ValueError('Duplicate predictions')
        if main_preds is None:main_preds=actual
        else:
            for k,r in actual.items():
                if r['variant']=='baseline42' and r!=main_preds[k]:raise ValueError('Sensitivity reference differs from original')
        for model in models:
            variant=model['variant'];task=model['task'];tag=model['tag'];held=model['held_out'];params=model['parameters']
            train=[];test=[]
            for run,step in keys:
                if step==0 or task=='forecast' and step==STEPS[-1]:continue
                target=STEPS[STEPS.index(step)+1] if task=='forecast' else step
                label=monitoring[(run,target)]['tags'][tag] if tag else monitoring[(run,target)]['label']
                if run!=held and label is not None:train.append((run,step,target,label))
                if run==held:test.append((run,step,target,label))
            if model['training_keys']!=[[r,s] for r,s,_,_ in train]:raise ValueError('Whole-run training split differs')
            if weak and model['l2']!=1/42:raise ValueError('Wrong sensitivity penalty')
            ti=[index[(r,s)] for r,s,_,_ in train];vi=[index[(r,s)] for r,s,_,_ in test]
            x=matrices[variant.removesuffix('_l2_1over42')]
            if 'constant' in params:
                if len({r[-1] for r in train})!=1 or params['constant']!=sum(r[-1] for r in train)/len(train):raise ValueError('Wrong fallback')
                scores=[params['constant']]*len(test)
            else:
                mean=x[ti].mean(0);scale=x[ti].std(0,unbiased=False).clamp_min(1e-6)
                torch.testing.assert_close(mean,x.new_tensor(params['mean']),rtol=0,atol=0)
                torch.testing.assert_close(scale,x.new_tensor(params['scale']),rtol=0,atol=0)
                scores=torch.sigmoid(((x[vi]-mean)/scale)@x.new_tensor(params['weight'])+params['bias']).tolist()
            for (run,step,target,_),score in zip(test,scores):
                row=actual[(variant,task,tag,run,step,target)];maxdiff=max(maxdiff,abs(row['score']-score))
                if row['predicted_label']!=int(score>=.5):raise ValueError('Classification differs')
                score_count+=1
            model_count+=1
        reports=json.loads((folder/'report.json').read_text())
        for suite in ['monitoring','evaluation']:
            rows=json.loads((folder/f'{suite}-predictions.json').read_text());truth=load_outcomes(root,suite)
            for row in rows:
                expected=truth[(row['run'],row['target_step'])]['tags'][row['tag']] if row['tag'] else truth[(row['run'],row['target_step'])]['label']
                if expected!=row['label']:raise ValueError('Scoring target differs')
                if row['variant']!='persistence' and (row['score']!=actual[key(row)]['score'] or row['predicted_label']!=actual[key(row)]['predicted_label']):raise ValueError('Evaluation suite changed fitted predictions')
            if weak:
                if summarize(rows,RUNS)!=reports[suite]:raise ValueError('Sensitivity report differs')
            else:
                for scope,held in [('all_seven',RUNS),('matched_six',[r for r in RUNS if r!='hate_speech'])]:
                    if summarize(rows,held)!=reports[scope][suite]:raise ValueError('Primary report differs')
    if maxdiff>1e-10:raise ValueError('Saved probability differs')
    main=json.loads((output/'monitoring-predictions.json').read_text());weak=json.loads((output/'regularization_check/monitoring-predictions.json').read_text())
    forecast_key=lambda r:(r['run'],r['input_step'],r['target_step'])
    full={forecast_key(r):r for r in main if r['task']=='forecast' and r['variant']=='baseline42'}
    scalar={forecast_key(r):r for r in weak if r['task']=='forecast' and r['variant']=='drift_only_l2_1over42'}
    equal=sum(full[k]['predicted_label']==scalar[k]['predicted_label'] for k in full)
    result=dict(passed=True,control_dimensions=dict(step_only=1,drift_only=1),training_only_scaling=True,whole_run_holdouts=True,
        all_seven_primary=True,models_verified=model_count,scores_recomputed=score_count,max_score_difference=maxdiff,
        baseline_reproduction=verify_previous(list(main_preds.values()),previous),
        evaluation_labels_used_only_for_scoring=True,weak_drift_forecast_classes_matching_full=equal,forecast_cases=len(full),
        source_hashes_verified=len(p['input_sha256']),sensitivity_l2=1/42)
    (output/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/step-drift-controls'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    a=p.parse_args();audit(a.root,a.output_dir,a.previous_predictions)
