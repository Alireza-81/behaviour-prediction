import json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import compare_selected_summary_monitors as experiment
from behaviour_prediction.monitor.features import extract_run
from behaviour_prediction.monitor.behaviours import TAGS


def test_summary_features_match_existing_extractor_and_use_subset_baseline():
    # An independently calculated two-probe-per-class example.
    probes=[dict(group=g) for g in ['benign','harmful','harmful_indirect'] for _ in range(2)]
    a=torch.ones((6,2));b=a+torch.tensor([[1.,0.],[3.,0.],[2.,0.],[4.,0.],[5.,0.],[7.,0.]])
    v=experiment.summarize_layer(a,b,a,2,probes)
    assert len(v)==21
    assert v['layer_31/benign/l2_mean']==2.
    assert v['layer_31/benign/l2_std']==1.
    assert v['layer_31/harmful/l2_mean']==3.
    assert v['layer_31/harmful/speed']==1.5
    assert v['layer_31/harmful-benign/l2_selectivity']==pytest.approx(.2)
    subset=experiment.summarize_layer(a[:,[0]],b[:,[0]],a[:,[0]],2,probes)
    assert subset['layer_31/benign/relative_l2']==2.
    assert v['layer_31/benign/relative_l2']==pytest.approx(2/2**.5)


def test_entire_real_layer_exactly_reproduces_all_existing_summaries():
    run=Path('outputs/early-steps/hate_speech')
    schema,rows=extract_run(run)
    from behaviour_prediction.trajectories import snapshots,load
    paths=snapshots(run/'activation_monitoring',None);base=load(paths[0][1]);a=base['activations']['layer_31'].float();past=a;last=0
    indexed={r['step']:r['features'] for r in rows}
    for step,path in paths:
        b=load(path)['activations']['layer_31'].float()
        values=experiment.summarize_layer(a,b,past,step-last,base['metadata']['probes'])
        assert values=={n:v for n,v in indexed[step].items() if n.startswith('layer_31/')}
        past=b;last=step


def test_fits_keep_training_only_scaling_and_preserve_original_feature_dimensions(monkeypatch):
    monkeypatch.setattr(experiment,'RUNS',['a','b','c'])
    keys=[(run,step) for run in experiment.RUNS for step in experiment.STEPS]
    values=torch.tensor([[float(step),float(i%3)] for i,(run,step) in enumerate(keys)],dtype=torch.float64)
    matrices=dict(baseline42=values,top_102_summary31=values[:,:1])
    outcomes={key:dict(label=int(key[1]>=5),tags={tag:0 for tag in TAGS}) for key in keys}
    predictions,models,checked=experiment.fit_all(matrices,keys,outcomes)
    assert checked==sum(p['variant'] in matrices for p in predictions)
    for model in models:
        assert all(key[0]!=model['held_out'] and key[1]!=0 for key in model['training_keys'])
        params=model['parameters'];x=matrices[model['variant']]
        indices=[keys.index(tuple(k)) for k in model['training_keys']]
        if 'constant' not in params:
            torch.testing.assert_close(x[indices].mean(0),x.new_tensor(params['mean']),rtol=0,atol=0)
            torch.testing.assert_close(x[indices].std(0,unbiased=False).clamp_min(1e-6),x.new_tensor(params['scale']),rtol=0,atol=0)
    assert all(p['input_step']>0 for p in predictions)
