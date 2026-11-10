import json
import torch
import pytest

from behaviour_prediction.explainer import baselines


def test_layer31_logistic_uses_only_retained_vectors_in_fitting_and_prediction(tmp_path,monkeypatch):
    examples=[dict(id=str(i),probe_id=f'p{i}',vector_index=i,split=s,run='r',step=i+1,harmful=i%2,tags=[],prompt='example request')
              for i,s in enumerate(['train','train','validation','validation','test','test','baseline_test','baseline_test'])]
    vectors=torch.arange(8*4*8,dtype=torch.float32).reshape(8,4,8)
    rms=torch.tensor([2.,3.,4.,5.])
    monkeypatch.setattr(baselines,'read_bundle',lambda _:(dict(statistics={'train':{'tags':{}}}),examples,dict(vectors=vectors,slot_rms=rms)))
    calls=[]
    def fit(train,val,y,vy,l2):
        calls.append(train.clone())
        return dict(mean=torch.zeros(train.shape[1]),scale=torch.ones(train.shape[1]),
                    weight=torch.zeros(train.shape[1],y.shape[1],dtype=torch.float64),bias=torch.zeros(y.shape[1],dtype=torch.float64)),1.
    monkeypatch.setattr(baselines,'fit_linear',fit)
    out=tmp_path/'layer31'
    baselines.run_baselines(tmp_path,out,layers=[31])
    expected=(vectors[:2,[2,3]]/rms[None,[2,3],None]).flatten(1).double()
    assert all(torch.equal(c,expected) for c in calls[:3])
    assert calls[3].shape[1]==2
    info=json.loads((out/'selection.json').read_text())
    assert info['layers']==[31]
    assert info['feature_dimensions']['full_vector_linear']==16
    assert info['feature_dimensions']['norms_linear']==2
    with pytest.raises(ValueError,match='layers'):
        baselines.run_baselines(tmp_path,tmp_path/'invalid',layers=[15])
