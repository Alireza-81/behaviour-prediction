"""First causal pilot: replace full residual vectors at the final prompt token.

Only the initial prompt forward pass is patched. Subsequent decode positions are
left untouched. Donor states must come from the identical tokenized prompt.
"""
import torch
from behaviour_prediction.monitoring.activations import resolve_transformer_layers, extract_hidden_tensor


class PrefillLastTokenPatch:
    def __init__(self, model, donors, prompt_length, coordinate_masks=None):
        self.layers=resolve_transformer_layers(model)
        self.donors=donors
        self.prompt_length=prompt_length
        self.coordinate_masks=coordinate_masks
        self.applied=set()
        self.handles=[]
        if not donors or prompt_length<1:raise ValueError('Need donors and a positive prompt length')
        if any(type(i) is not int or i<0 or i>=len(self.layers) for i in donors):raise ValueError('Invalid patch layer')
        if any(v.ndim!=1 or not torch.isfinite(v).all() for v in donors.values()):raise ValueError('Donors must be finite one-dimensional vectors')
        if coordinate_masks is not None:
            if set(coordinate_masks)!=set(donors):raise ValueError('Masks must cover exactly the donor layers')
            for i,mask in coordinate_masks.items():
                if mask.dtype!=torch.bool or mask.shape!=donors[i].shape:
                    raise ValueError('Coordinate masks must be boolean and match donor width')

    def hook(self,index):
        def apply(module,inputs,output):
            hidden=extract_hidden_tensor(output)
            if index in self.applied:return output
            if hidden.ndim!=3 or hidden.shape[0]!=1:raise ValueError('Patching pilot requires a single prompt')
            if hidden.shape[1]!=self.prompt_length:
                raise ValueError('First patched pass is not the complete prompt; cache alignment is invalid')
            donor=self.donors[index].to(device=hidden.device,dtype=hidden.dtype)
            if donor.shape!=(hidden.shape[-1],):raise ValueError('Donor hidden width mismatch')
            replacement=hidden.clone()
            if self.coordinate_masks is None:
                replacement[0,-1,:]=donor
            else:
                mask=self.coordinate_masks[index].to(hidden.device)
                replacement[0,-1,:]=torch.where(mask,donor,hidden[0,-1,:])
            self.applied.add(index)
            if isinstance(output,tuple):return (replacement,*output[1:])
            return replacement
        return apply

    def __enter__(self):
        try:
            for i in self.donors:self.handles.append(self.layers[i].register_forward_hook(self.hook(i)))
        except Exception:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[]


def last_token_donors(model,encoded,layers):
    """Fresh live forward pass, not a gradient/saliency computation."""
    states={}
    modules=resolve_transformer_layers(model)
    handles=[]
    def capture(i):
        def hook(module,inputs,output):
            states[i]=extract_hidden_tensor(output)[0,-1,:].detach().cpu().clone()
        return hook
    try:
        for i in layers:handles.append(modules[i].register_forward_hook(capture(i)))
        with torch.no_grad():model(**encoded,use_cache=False)
    finally:
        for handle in handles:handle.remove()
    return states


def select_changed_coordinates(baseline,finetuned,count,*,method='top_change',seed=42):
    """Fixed per-prompt mask; same mask can be used in both patch directions.

    Top ranking uses absolute signed-vector difference, not activation magnitude.
    Ties use coordinate order. Random budgets are nested for the same seed.
    """
    a=baseline.detach().float().cpu();b=finetuned.detach().float().cpu()
    if a.ndim!=1 or a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Need aligned finite one-dimensional endpoint activations')
    if type(count) is not int or not 0<=count<=a.numel():raise ValueError('Invalid coordinate budget')
    if method=='top_change':order=torch.argsort((b-a).abs(),descending=True,stable=True)
    elif method=='random':order=torch.randperm(a.numel(),generator=torch.Generator().manual_seed(seed))
    else:raise ValueError('Unknown coordinate selection method')
    mask=torch.zeros(a.numel(),dtype=torch.bool);mask[order[:count]]=True
    return mask
