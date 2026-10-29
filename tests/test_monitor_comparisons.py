import torch

from behaviour_prediction.monitor.comparisons import per_probe_values, history_features, score_predictions


def test_per_probe_features_keep_probes_separate():
    initial=torch.tensor([[1.,0.],[0.,1.]])
    current=torch.tensor([[2.,0.],[0.,3.]])
    values=per_probe_values(current,initial,initial,2)
    assert values.shape == (2,4)
    assert torch.allclose(values,torch.tensor([[1.,1.,0.,.5],[2.,2.,0.,1.]]))
    changed=current.clone();changed[0,1]=5.
    assert torch.equal(per_probe_values(changed,initial,initial,2)[1],values[1])


def test_history_never_uses_future_or_other_run_and_reports_padding():
    keys=[("a",10),("b",20),("a",0),("a",20),("b",0),("b",10)]
    values=torch.tensor([[1.],[120.],[0.],[2.],[100.],[110.]],dtype=torch.float64)
    features,flags=history_features(values,keys)
    assert features[0].tolist() == [1.,0.,0.,1.,0.]
    assert features[1].tolist() == [120.,110.,100.,1.,1.]
    changed=values.clone();changed[3]=999.;changed[1]=888.
    assert torch.equal(history_features(changed,keys)[0][0],features[0])
    assert flags[0].tolist() == [1.,0.]
    assert torch.equal(features[:,-2:],flags)


def test_scoring_does_not_change_predictions_and_uses_new_persistence_label():
    from copy import deepcopy
    from behaviour_prediction.monitor.behaviours import TAGS
    outcomes={('a',10):dict(label=0,tags={tag:0 for tag in TAGS}),
              ('a',20):dict(label=0,tags={tag:0 for tag in TAGS})}
    predictions=[dict(variant="persistence",task="forecast",tag=None,run="a",input_step=10,
                      target_step=20,score=1.,predicted_label=1)]
    before=deepcopy(predictions)
    report,scored=score_predictions(predictions,outcomes)
    assert report['forecast']['persistence']['tn']==1
    assert predictions==before and scored[0]['predicted_label']==0


def test_baseline_verification_handles_binary_predictions_without_tag(tmp_path,monkeypatch):
    import json
    import pytest
    from behaviour_prediction.monitor.comparisons import verify_baseline
    monkeypatch.chdir(tmp_path)
    examples={
        'detection/heldout_predictions.json':[dict(method='summary',run='a',step=10,score=.7,predicted_label=1)],
        'behaviours/predictions.json':[dict(method='summary',run='a',step=10,tag='harassment',score=.7,predicted_label=1)],
        'forecast/predictions.json':[dict(method='activation_forecast',run='a',input_step=10,target_step=20,score=.7,predicted_label=1)]}
    for name,rows in examples.items():
        p=__import__('pathlib').Path('outputs/results')/name
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(rows))
    actual=[dict(variant='baseline',task=task,tag=tag,run='a',input_step=10,target_step=target,score=.7,predicted_label=1)
            for task,tag,target in [('detection',None,10),('behaviours','harassment',10),('forecast',None,20)]]
    verify_baseline(actual)
    actual[0]['score']=.8
    with pytest.raises(ValueError,match='reproduction failed'):
        verify_baseline(actual)
