"""Recompute summary inputs and verify saved whole-run logistic predictions."""
import argparse,json
from pathlib import Path
import torch
from compare_selected_summary_monitors import build_inputs
from compare_selected_coordinate_monitors import verify_previous,summarize
from run_early_step_experiments import load_outcomes
from plot_selected_coordinate_trajectories import sha
from audit_early_steps import RUNS,STEPS


def audit(root,output,mask,previous):
    torch.set_num_threads(2)
    p=json.loads((output/'protocol.json').read_text())
    for mapping in ['input_sha256','code_sha256']:
        for name,value in p[mapping].items():
            if sha(Path(name))!=value:raise ValueError(f'Changed input/code: {name}')
    data,keys,matrices,subsets,_=build_inputs(root,mask)
    if subsets!=p['coordinate_subsets'] or set(subsets)!={'102','205','410'} or p['random_seed']!=42:raise ValueError('Wrong subsets')
    stored=torch.load(output/'summary-inputs.pt',map_location='cpu',weights_only=True)
    if [list(k) for k in keys]!=[list(k) for k in stored['keys']]:raise ValueError('Checkpoint order differs')
    names=data['schema']['features'];col15=[i for i,n in enumerate(names) if n.startswith('layer_15/')]
    for v,x in matrices.items():
        torch.testing.assert_close(x,stored['matrices'][v],rtol=0,atol=0)
        if x.shape[1]!=(42 if v=='baseline42' or v.endswith('summary42') else 21):raise ValueError('Feature count differs')
        if v.endswith('summary42'):torch.testing.assert_close(x[:,col15],matrices['baseline42'][:,col15],rtol=0,atol=0)
    predictions=json.loads((output/'predictions-before-evaluation.json').read_text())
    models=json.loads((output/'fitted-monitors.json').read_text());monitor=load_outcomes(root,'monitoring')
    def key(r):return r['variant'],r['task'],r['tag'],r['run'],r['input_step'],r['target_step']
    by_key={key(r):r for r in predictions};index={k:i for i,k in enumerate(keys)}
    if len(by_key)!=len(predictions):raise ValueError('Duplicate predictions')
    verified=0;maxdiff=0.
    for model in models:
        task=model['task'];held=model['held_out'];tag=model['tag'];variant=model['variant']
        train=[];test=[]
        for run,step in keys:
            if step==0 or task=='forecast' and step==STEPS[-1]:continue
            target=STEPS[STEPS.index(step)+1] if task=='forecast' else step
            truth=monitor[(run,target)]['tags'][tag] if tag else monitor[(run,target)]['label']
            if run!=held and truth is not None:train.append((run,step,target,truth))
            if run==held:test.append((run,step,target,truth))
        if model['training_keys']!=[[r,s] for r,s,_,_ in train] or model['training_count']!=len(train) or model['training_positive_count']!=sum(r[-1] for r in train):raise ValueError('Training split/target mismatch')
        ti=[index[(r,s)] for r,s,_,_ in train];vi=[index[(r,s)] for r,s,_,_ in test];x=matrices[variant];params=model['parameters']
        if 'constant' in params:
            if len({r[-1] for r in train})!=1 or params['constant']!=sum(r[-1] for r in train)/len(train):raise ValueError('Invalid fallback')
            scores=[params['constant']]*len(test)
        else:
            mean=x[ti].mean(0);scale=x[ti].std(0,unbiased=False).clamp_min(1e-6)
            torch.testing.assert_close(mean,x.new_tensor(params['mean']),rtol=0,atol=0)
            torch.testing.assert_close(scale,x.new_tensor(params['scale']),rtol=0,atol=0)
            scores=torch.sigmoid(((x[vi]-mean)/scale)@x.new_tensor(params['weight'])+params['bias']).tolist()
        for (run,step,target,_),score in zip(test,scores):
            row=by_key[(variant,task,tag,run,step,target)]
            maxdiff=max(maxdiff,abs(row['score']-score))
            if row['predicted_label']!=int(score>=.5):raise ValueError('Saved classification differs')
            verified+=1
    if maxdiff>1e-10:raise ValueError('Saved score differs')
    report=json.loads((output/'report.json').read_text())
    for suite in ['monitoring','evaluation']:
        truth=load_outcomes(root,suite);rows=json.loads((output/f'{suite}-predictions.json').read_text())
        for row in rows:
            expected=truth[(row['run'],row['target_step'])]['tags'][row['tag']] if row['tag'] else truth[(row['run'],row['target_step'])]['label']
            if row['label']!=expected:raise ValueError('Target mismatch')
            if row['variant']!='persistence':
                original=by_key[key(row)]
                if original['score']!=row['score'] or original['predicted_label']!=row['predicted_label']:raise ValueError('Scoring suite changed predictions')
        for scope,held in [('primary_other_runs',[r for r in RUNS if r!='hate_speech']),('all_seven_exploratory',RUNS)]:
            if summarize(rows,held)!=report[scope][suite]:raise ValueError('Report differs from scored predictions')
    result=dict(passed=True,input_variants=len(matrices),selected_summary31=12,selected_summary42=12,random_seed=42,budgets=[102,205,410],
        original_layer15_unchanged_in_summary42=True,full_layer31_summaries_exactly_match_original=True,
        training_only_normalization_verified=True,models_verified=len(models),scores_recomputed=verified,max_score_difference=maxdiff,
        baseline_reproduction=verify_previous(predictions,previous),evaluation_labels_used_only_for_scoring=True,
        primary_excludes_selection_source_family=True,source_hashes_verified=len(p['input_sha256']))
    (output/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/selected-coordinate-summary-monitors'))
    p.add_argument('--mask',type=Path,default=Path('artifacts/shared-activation-mask-v1/mask.json'))
    p.add_argument('--previous-predictions',type=Path,default=Path('outputs/early-steps/results/predictions.json'))
    a=p.parse_args();audit(a.root,a.output_dir,a.mask,a.previous_predictions)
