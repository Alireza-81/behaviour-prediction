"""Matched per-response controls, using training-only normalization and vocabulary."""
import json
import re
from collections import Counter

import torch
import torch.nn.functional as F

from .data import TAGS, read_bundle
from .runtime import report_predictions, supported_metrics


def fit_linear(train,validation,y,vy,l2):
    mean=train.mean(0)
    scale=train.std(0,unbiased=False).clamp_min(1e-4)
    x=(train-mean)/scale
    vx=(validation-mean)/scale
    prior=y.mean(0).clamp(1e-5,1-1e-5)
    # Match the decoder's training-only harmful/non-harmful oversampling in expectation.
    row_weights=torch.where(y[:,0]>0,.5/prior[0],.5/(1-prior[0]))
    weight=torch.zeros(x.shape[1],y.shape[1],dtype=x.dtype,requires_grad=True)
    bias=torch.logit(prior).detach().requires_grad_()
    optimizer=torch.optim.LBFGS([weight,bias],max_iter=60,line_search_fn='strong_wolfe',history_size=5)
    def closure():
        optimizer.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(x@weight+bias,y,reduction='none').mean(1)*row_weights).mean()+l2*weight.square().sum()/(2*y.shape[1])
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        loss=F.binary_cross_entropy_with_logits(vx@weight+bias,vy).item()
    return dict(mean=mean,scale=scale,weight=weight.detach(),bias=bias.detach()),loss


def bow_matrix(examples,train_indices):
    tokens=[re.findall(r'\b\w+\b',r['prompt'].lower()) for r in examples]
    documents=Counter(t for i in train_indices for t in set(tokens[i]))
    vocabulary=sorted((t for t,n in documents.items() if n>=2),key=lambda t:(-documents[t],t))[:2048]
    index={t:i for i,t in enumerate(vocabulary)}
    matrix=torch.zeros(len(examples),len(vocabulary),dtype=torch.float64)
    for i,words in enumerate(tokens):
        for t,n in Counter(words).items():
            if t in index:
                matrix[i,index[t]]=n
    idf=torch.tensor([1.+__import__('math').log((1+len(train_indices))/(1+documents[t])) for t in vocabulary])
    matrix*=idf
    matrix/=matrix.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return matrix,vocabulary


def run_baselines(dataset,output,*,layers=(15,31)):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    manifest,examples,data=read_bundle(dataset)
    layers=tuple(layers)
    if layers not in {(15,31),(31,)}:raise ValueError('Use layers 15 31 or layer 31 only')
    slots=[0,1,2,3] if layers==(15,31) else [2,3]
    train=[i for i,r in enumerate(examples) if r['split']=='train']
    validation=[i for i,r in enumerate(examples) if r['split']=='validation']
    y=torch.tensor([[r['harmful']]+[int(t in r['tags']) for t in TAGS] for r in examples],dtype=torch.float64)
    vectors=(data['vectors'][:,slots]/data['slot_rms'][None,slots,None]).flatten(1).double()
    norms=data['vectors'][:,slots].double().norm(dim=-1)
    text,vocabulary=bow_matrix(examples,train)
    matrices={'full_vector_linear':vectors,'norms_linear':norms,'prompt_bow_linear':text,
              'step_only_linear':torch.tensor([[r['step']] for r in examples],dtype=torch.float64)}
    parameters,all_scores,selection={},{},{}
    for name,x in matrices.items():
        candidates=[]
        for l2 in (.01,.1,1.):
            model,value=fit_linear(x[train],x[validation],y[train],y[validation],l2)
            candidates.append((value,l2,model))
        value,l2,model=min(candidates,key=lambda v:v[0])
        parameters[name]=model
        selection[name]=dict(l2=l2,validation_loss=value,candidates=[dict(l2=l,validation_loss=v) for v,l,_ in candidates])
        all_scores[name]=torch.sigmoid(((x-model['mean'])/model['scale'])@model['weight']+model['bias'])
        print(name,selection[name],flush=True)
    prior=y[train].mean(0)
    all_scores['frequency']=prior.expand(len(examples),-1)
    all_scores['always_absent']=torch.zeros_like(y)
    all_scores['always_present']=torch.ones_like(y)
    records=[]
    for method,scores in all_scores.items():
        for i,row in enumerate(examples):
            if row['split'] not in {'validation','test','baseline_test'}:
                continue
            positive=scores[i,0].item()>=.5
            prediction=dict(harmful_response=positive,behaviours=[t for j,t in enumerate(TAGS,1) if scores[i,j]>=.5] if positive else [],
                            description='Structured baseline prediction.')
            records.append(dict(id=row['id'],split=row['split'],run=row['run'],step=row['step'],probe_id=row['probe_id'],
                                harmful=row['harmful'],tags=row['tags'],condition=method,prediction=prediction))
    report={split:{m:supported_metrics(report_predictions([r for r in records if r['split']==split and r['condition']==m]),manifest)
                   for m in all_scores} for split in ('validation','test','baseline_test')}
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (output/'selection.json').write_text(json.dumps(dict(selection=selection,vocabulary=vocabulary,layers=layers,
        feature_dimensions={name:x.shape[1] for name,x in matrices.items()},
        validation_only_model_selection=True,training_only_balanced_harmfulness_weighting=True,
        split='Same prepared manifest as decoder; no test labels used in fitting'),indent=2)+'\n')
    torch.save(parameters,output/'parameters.pt')
    print('Saved matched baselines to',output)
