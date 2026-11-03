"""First causal pilot: replace full residual vectors at the final prompt token.

Only the initial prompt forward pass is patched. Subsequent decode positions are
left untouched. Donor states must come from the identical tokenized prompt.
"""
import torch
from behaviour_prediction.monitoring.activations import resolve_transformer_layers, extract_hidden_tensor


class PrefillLastTokenPatch:
    def __init__(self, model, donors, prompt_length):
        self.layers=resolve_transformer_layers(model)
        self.donors=donors
        self.prompt_length=prompt_length
        self.applied=set()
        self.handles=[]
        if not donors or prompt_length<1:raise ValueError('Need donors and a positive prompt length')
        if any(type(i) is not int or i<0 or i>=len(self.layers) for i in donors):raise ValueError('Invalid patch layer')
        if any(v.ndim!=1 or not torch.isfinite(v).all() for v in donors.values()):raise ValueError('Donors must be finite one-dimensional vectors')

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
            replacement[0,-1,:]=donor
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
