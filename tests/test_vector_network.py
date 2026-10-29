import json

import torch

from behaviour_prediction.monitor.vector_network import center_gram, fit_kernel_logistic, fit_mlp, predict_mlp


def test_vector_dot_products_equal_explicit_centered_vectors_and_use_training_only():
    values=torch.tensor([[1.,2.,0.],[2.,0.,1.],[0.,1.,3.],[10.,20.,30.]],dtype=torch.float64)
    train=[0,1,2];test=[3]
    k,cross,settings=center_gram(values@values.T,train,test)
    centered=values-values[train].mean(dim=0)
    scale=(centered[train].square().sum(dim=1).mean()/42).sqrt()
    z=centered/scale
    assert torch.allclose(k,z[train]@z[train].T)
    assert torch.allclose(cross,z[test]@z[train].T)
    changed=values.clone();changed[3]*=1000
    changed_k,_,changed_settings=center_gram(changed@changed.T,train,test)
    assert torch.equal(k,changed_k) and settings==changed_settings


def test_full_vector_logistic_solution_satisfies_primal_optimality():
    values=torch.tensor([[-2.,1.],[-1.,0.],[0.,1.],[1.,0.],[2.,1.],[3.,0.]],dtype=torch.float64)
    labels=torch.tensor([0.,0.,0.,1.,1.,1.],dtype=torch.float64)
    values=values-values.mean(dim=0)
    values=values/(values.square().sum(dim=1).mean()/42).sqrt()
    kernel=values@values.T
    params=fit_kernel_logistic(kernel,labels)
    weight=values.T@torch.tensor(params['alpha'],dtype=torch.float64)
    difference=torch.sigmoid(values@weight+params['bias'])-labels
    assert (values.T@difference/len(labels)+weight).abs().max()<1e-4
    assert difference.mean().abs()<1e-4


def test_mlp_is_reproducible_and_test_batch_does_not_change_prediction():
    train=torch.tensor([[0.,1.],[1.,2.],[2.,0.],[3.,1.]],dtype=torch.float64)
    labels=torch.tensor([0.,0.,1.,1.],dtype=torch.float64)
    first=fit_mlp(train,labels,42,steps=15)
    second=fit_mlp(train,labels,42,steps=15)
    assert first==second
    assert first['mean']==train.mean(dim=0).tolist()
    point=torch.tensor([[1.5,1.]],dtype=torch.float64)
    original=predict_mlp(first,point)[0]
    combined=predict_mlp(first,torch.cat([point,point*10000]))[0]
    assert abs(original-combined)<1e-12
    assert predict_mlp(json.loads(json.dumps(first)),point)[0]==original
