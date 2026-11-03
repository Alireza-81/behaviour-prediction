"""Prompt-independent coordinate selection from development-only attribution."""
import hashlib
import json
import math

import torch


def identity(value, field='mask_id'):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!=field},sort_keys=True).encode()).hexdigest()


def tensor_identity(value):
    tensor=value.detach().cpu().contiguous()
    header=json.dumps(dict(dtype=str(tensor.dtype),shape=list(tensor.shape)),sort_keys=True).encode()
    return hashlib.sha256(header+tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def average_development_scores(cache, probes, directions):
    """Each target prompt/direction contributes equally after L1 normalization.

    Scores are already averaged over reference token positions. Keep their sign;
    do not turn counterproductive donor replacements into positive importance.
    Benign prompts are diagnostics, not included in the selected ranking.
    """
    targets=[p for p in probes if p['role']=='target']
    if not targets or not directions:raise ValueError('Need development target prompts and directions')
    vectors=[];width=None
    for probe in targets:
        for direction in directions:
            value=cache[f"{direction}:{probe['id']}"]['scores'].detach().cpu().double()
            if value.ndim!=1 or not torch.isfinite(value).all():raise ValueError('Invalid development scores')
            if width is not None and value.numel()!=width:raise ValueError('Mismatched score widths')
            width=value.numel()
            scale=float(value.abs().sum())
            vectors.append(value/max(scale,1e-12))
    return torch.stack(vectors).mean(0)


def fixed_masks(scores, budgets, seeds):
    if scores.ndim!=1 or not torch.isfinite(scores).all():raise ValueError('Invalid shared ranking')
    width=scores.numel();order=torch.argsort(scores,descending=True,stable=True)
    masks={}
    for percent in budgets:
        if not 0<percent<=100:raise ValueError('Invalid percentage budget')
        count=math.ceil(width*percent/100)
        masks[f'shared_{percent:02d}pct']=sorted(order[:count].tolist())
        for seed in seeds:
            random=torch.randperm(width,generator=torch.Generator().manual_seed(seed))
            masks[f'random{seed}_{percent:02d}pct']=sorted(random[:count].tolist())
    return masks


def validate_frozen_mask(artifact):
    if identity(artifact)!=artifact['mask_id']:raise ValueError('Frozen mask identity differs')
    scores=torch.tensor(artifact['aggregate_scores'],dtype=torch.float64)
    expected=fixed_masks(scores,artifact['budgets_percent'],artifact['random_seeds'])
    if expected!=artifact['selected_indices']:raise ValueError('Frozen indices do not match development ranking')
    if scores.numel()!=artifact['width']:raise ValueError('Wrong frozen coordinate width')
    result={}
    for name,indices in artifact['selected_indices'].items():
        mask=torch.zeros(artifact['width'],dtype=torch.bool);mask[indices]=True
        result[name]=mask
    return result
