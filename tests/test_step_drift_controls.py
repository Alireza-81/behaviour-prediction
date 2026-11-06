import copy,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import compare_step_and_drift_monitors as controls
from compare_selected_summary_monitors import fit_all
from compare_selected_coordinate_monitors import score_rows
from behaviour_prediction.monitor.behaviours import TAGS


def test_drift_is_mean_probe_norm_not_norm_of_mean_and_zero_at_baseline():
    a=torch.ones((2,2));delta=torch.tensor([[3.,-3.],[-3.,3.]])
    drift=controls.overall_drift(a,a+delta)
    assert drift==float(delta.norm(dim=1).mean()) and drift>0
    assert float(delta.mean(0).norm())==0
    assert controls.overall_drift(a,a)==0


def test_actual_control_inputs_have_one_scalar_and_step_is_current_step():
    data,keys,matrices,_=controls.build_inputs(Path('outputs/early-steps'))
    assert matrices['baseline42'].shape==(56,42)
    assert matrices['step_only'].shape==matrices['drift_only'].shape==(56,1)
    for i,(run,step) in enumerate(keys):
        assert matrices['step_only'][i,0]==step
        if step==0:assert matrices['drift_only'][i,0]==0
    assert len({float(matrices['step_only'][i,0]) for i,(run,s) in enumerate(keys) if s==5})==1
    assert len({float(matrices['drift_only'][i,0]) for i,(run,s) in enumerate(keys) if s==20})>1


def test_control_models_use_whole_run_train_only_scaling_and_evaluation_does_not_change_scores(monkeypatch):
    import compare_selected_summary_monitors as fitting
    monkeypatch.setattr(fitting,'RUNS',['a','b','c'])
    keys=[(run,step) for run in fitting.RUNS for step in controls.STEPS]
    matrices=dict(baseline42=torch.tensor([[step,step**.5] for run,step in keys],dtype=torch.float64),
        step_only=torch.tensor([[step] for run,step in keys],dtype=torch.float64),drift_only=torch.tensor([[step/20] for run,step in keys],dtype=torch.float64))
    outcomes={key:dict(label=int(key[1]>=5),tags={tag:0 for tag in TAGS}) for key in keys}
    predictions,models,_=fit_all(matrices,keys,outcomes)
    for model in models:
        assert all(run!=model['held_out'] and step>0 for run,step in model['training_keys'])
        params=model['parameters'];x=matrices[model['variant']]
        ti=[keys.index(tuple(k)) for k in model['training_keys']]
        if 'constant' not in params:
            torch.testing.assert_close(x[ti].mean(0),x.new_tensor(params['mean']),rtol=0,atol=0)
    separate=copy.deepcopy(outcomes)
    for row in separate.values():row['label']=1
    a=score_rows(predictions,outcomes,'monitoring');b=score_rows(predictions,separate,'evaluation')
    key=lambda r:(r['variant'],r['task'],r['tag'],r['run'],r['input_step'])
    assert {key(r):r['score'] for r in a if r['variant']!='persistence'}=={key(r):r['score'] for r in b if r['variant']!='persistence'}
