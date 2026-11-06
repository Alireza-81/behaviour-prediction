"""Fixed scalar-control sensitivity check: L2=1/42 versus the main L2=1 recipe.

Chosen from the42-vs1 input-count difference, not by searching test performance.
No nonlinear features or threshold changes.
"""
import argparse,json,csv
from pathlib import Path
import torch
from behaviour_prediction.monitor.pilot import fit_fold
from behaviour_prediction.monitor.behaviours import TAGS
from compare_step_and_drift_monitors import build_inputs
from compare_selected_coordinate_monitors import cases_for,score_rows,summarize
from run_early_step_experiments import load_outcomes
from plot_selected_coordinate_trajectories import sha
from audit_early_steps import RUNS


def run(root,output):
    destination=output/'regularization_check'
    if destination.exists() and any(destination.iterdir()):raise ValueError('Sensitivity results already exist')
    torch.set_num_threads(2)
    data,keys,matrices,_=build_inputs(root);truth=load_outcomes(root,'monitoring');predictions=[];models=[]
    l2=1/42
    for task,tag in [('detection',None),('forecast',None),*[('behaviours',t) for t in TAGS]]:
        cases=cases_for(keys,truth,task,tag)
        for held in RUNS:
            train=[r for r in cases if r['run']!=held and r['label'] is not None];test=[r for r in cases if r['run']==held]
            ti=[r['feature_index'] for r in train];vi=[r['feature_index'] for r in test];y=torch.tensor([r['label'] for r in train],dtype=torch.float64)
            for name in ['step_only','drift_only']:
                x=matrices[name];variant=name+'_l2_1over42'
                if len(set(y.tolist()))==1:params=dict(constant=float(y.mean()));scores=[float(y.mean())]*len(test)
                else:
                    params=fit_fold(x[ti],x[vi],y,l2=l2,return_parameters=True)
                    scores=torch.sigmoid(((x[vi]-x.new_tensor(params['mean']))/x.new_tensor(params['scale']))@x.new_tensor(params['weight'])+params['bias']).tolist()
                models.append(dict(variant=variant,task=task,tag=tag,held_out=held,l2=l2,parameters=params,training_keys=[keys[i] for i in ti]))
                for row,value in zip(test,scores):predictions.append(dict(**{k:v for k,v in row.items() if k not in ('feature_index','label')},task=task,variant=variant,score=value,predicted_label=int(value>=.5)))
    original=json.loads((output/'predictions-before-evaluation.json').read_text())
    original=[r for r in original if r['variant'] in ['baseline42','frequency','always_absent','always_present']]
    predictions=original+predictions
    destination.mkdir(parents=True)
    def save(name,v):(destination/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    save('predictions-before-evaluation.json',predictions);save('fitted-monitors.json',models)
    evaluation=load_outcomes(root,'evaluation')
    reports={}
    for suite,outcomes in [('monitoring',truth),('evaluation',evaluation)]:
        rows=score_rows(predictions,outcomes,suite);save(f'{suite}-predictions.json',rows)
        reports[suite]=summarize(rows,RUNS)
    save('report.json',reports)
    save('protocol.json',dict(l2=l2,threshold=.5,scope='All seven whole-run holdouts',
        rationale='A standardized42-input model has roughly42 times scalar input energy, and redundant features can reduce effective L2 cost. L2=1/42 is a fixed sensitivity check, not a tuned best value.',
        meaning='Equivalent to multiplying a standardized scalar by sqrt42 before applying an L2=1 coefficient penalty. Not a claim that dimensions imply exactly42 independent directions.',
        limitation='Changing regularization affects fit/capacity. Both controls remain linear logistic mappings of one scalar; this is not a flexible nonlinear or causal-identification test.',
        source_inputs_sha256=json.loads((output/'protocol.json').read_text())['input_sha256'],code_sha256=sha(Path(__file__))))
    for suite,tasks in reports.items():
        for task,variants in tasks.items():
            for variant in ['baseline42','step_only_l2_1over42','drift_only_l2_1over42']:
                r=variants[variant];print(suite,task,variant,r['tp']+r['tn'],r['checkpoints'],r.get('macro_f1'),r.get('early_warnings'),r.get('false_warnings'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/step-drift-controls'))
    a=p.parse_args();run(a.root,a.output_dir)
