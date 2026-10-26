import pytest
import torch

from behaviour_prediction.monitor.pilot import fit_fold, metrics


@pytest.mark.parametrize('project', [False, True])
def test_heldout_batch_cannot_change_preprocessing(project):
    train = torch.tensor([[0.,0.],[1.,0.],[2.,1.],[3.,1.]],dtype=torch.float64)
    labels = torch.tensor([0.,0.,1.,1.],dtype=torch.float64)
    point = torch.tensor([[1.5,0.2]],dtype=torch.float64)
    first = fit_fold(train,point,labels,project=project)
    second = fit_fold(train,torch.cat([point,point*100000]),labels,project=project)
    assert first[0] == pytest.approx(second[0],abs=1e-10)


def test_negative_control_metrics_preserve_false_alerts():
    rows = [dict(label=0,predicted_label=1,score=.9),dict(label=0,predicted_label=0,score=.1)]
    result = metrics(rows)
    assert result['fp']==1 and result['tn']==1
    assert result['recall'] is None
    assert result['accuracy']==.5
