import copy,json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import compare_selected_coordinate_monitors as experiment
from behaviour_prediction.monitor.vector_network import center_gram,fit_kernel_logistic
from behaviour_prediction.monitor.behaviours import TAGS


def test_linear_gram_monitor_matches_direct_linear_logistic_and_training_only_centering():
    torch.manual_seed(2)
    x=torch.randn(8,6,dtype=torch.float64);train=list(range(6));test=[6,7]
    y=torch.tensor([0.,0.,0.,1.,1.,1.],dtype=torch.float64)
    gram=x@x.T;kernel,cross,settings=center_gram(gram,train,test)
    params=fit_kernel_logistic(kernel,y)
    predicted=torch.sigmoid(cross@torch.tensor(params['alpha'],dtype=torch.float64)+params['bias'])
    center=x[train].mean(0);centered=x[train]-center
    factor=(42/centered.square().sum(1).mean()).sqrt()
    a=centered*factor;b=(x[test]-center)*factor
    w=torch.zeros(6,dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(a@w+bias,y)+w.square().sum()/2
        loss.backward();return loss
    opt.step(closure)
    direct=torch.sigmoid(b@w.detach()+bias.detach())
    torch.testing.assert_close(predicted,direct,atol=2e-4,rtol=2e-4)
    modified=x.clone();modified[test]*=1000
    after_kernel,after_cross,after_settings=center_gram(modified@modified.T,train,test)
    torch.testing.assert_close(kernel,after_kernel)
    assert after_settings==settings
    assert not torch.equal(cross,after_cross)


def test_cases_exclude_step_zero_and_forecast_uses_next_actual_checkpoint():
    keys=[(run,step) for run in ['a','b'] for step in experiment.STEPS]
    outcomes={key:dict(label=int(key[1]>=5),tags={tag:0 for tag in TAGS}) for key in keys}
    detect=experiment.cases_for(keys,outcomes,'detection',None)
    forecast=experiment.cases_for(keys,outcomes,'forecast',None)
    assert len(detect)==14 and len(forecast)==12
    assert all(c['input_step']>0 for c in detect+forecast)
    assert {(c['input_step'],c['target_step']) for c in forecast}=={(1,2),(2,3),(3,5),(5,7),(7,10),(10,20)}
    assert all(c['label']==outcomes[(c['run'],c['target_step'])]['label'] for c in forecast)


def test_whole_run_fits_keep_parameters_and_predictions_reproducible_and_targets_independent(monkeypatch,tmp_path):
    monkeypatch.setattr(experiment,'RUNS',['a','b','c'])
    keys=[(run,step) for run in experiment.RUNS for step in experiment.STEPS]
    x=torch.tensor([[step/10, {'a':0,'b':1,'c':2}[run]] for run,step in keys],dtype=torch.float64)
    outcomes={key:dict(label=int(key[1]>=5),tags={tag:0 for tag in TAGS}) for key in keys}
    predictions,models=experiment.fit_monitors(x,{'random42_102':x@x.T},keys,outcomes)
    assert all(model['training_count'] in (12,14) for model in models)
    for model in models:
        if model['variant']=='random42_102' and 'training_keys' in model['parameters']:
            assert all(key[0]!=model['held_out'] for key in model['parameters']['training_keys'])
    assert all(p['input_step']>0 for p in predictions)
    original=copy.deepcopy(predictions)
    monitoring=experiment.score_rows(predictions,outcomes,'monitoring')
    separate=copy.deepcopy(outcomes)
    for value in separate.values():value['label']=1
    evaluation=experiment.score_rows(predictions,separate,'evaluation')
    assert predictions==original
    # The same actual monitor predictions are reused regardless of evaluation outcomes.
    compare=lambda rows:{(r['task'],r['tag'],r['variant'],r['run'],r['input_step']):r['score'] for r in rows if r['variant']!='persistence'}
    assert compare(monitoring)==compare(evaluation)
    old=[dict(p,experiment=p['task'],method='monitor') for p in predictions if p['variant']=='baseline42']
    path=tmp_path/'old.json';path.write_text(json.dumps(old))
    assert experiment.verify_previous(predictions,path)['max_score_difference']==0
    old[0]['score']+=.1;path.write_text(json.dumps(old))
    with pytest.raises(ValueError,match='reproduction failed'):experiment.verify_previous(predictions,path)
