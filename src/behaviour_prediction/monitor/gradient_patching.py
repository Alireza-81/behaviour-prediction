"""Signed attribution for a proposed prefill replacement, not a safety classifier."""
import torch
import torch.nn.functional as F

from behaviour_prediction.monitoring.activations import resolve_transformer_layers, extract_hidden_tensor


def endpoint(model, encoded, layer):
    states = {}

    def capture(module, inputs, output):
        states[layer] = extract_hidden_tensor(output)[0, -1].detach().cpu().clone()

    handle = resolve_transformer_layers(model)[layer].register_forward_hook(capture)
    try:
        with torch.no_grad():
            logits = model(**encoded, use_cache=False).logits[0, -1].float().cpu()
    finally:
        handle.remove()
    return states, logits


def replacement_scores(model, encoded, layer, donor, donor_logits):
    """Estimate KL reduction from each donor replacement: -gradient * delta.

    KL(target donor next-token distribution || recipient distribution) is the
    objective. Positive scores predict improvement. No judge labels or answer
    text enter selection. Frozen parameters still permit activation gradients.
    """
    if model.training:
        raise ValueError('Use evaluation mode for reproducible attribution')
    modules = resolve_transformer_layers(model)
    if not 0 <= layer < len(modules):
        raise ValueError('Invalid attribution layer')
    state = {}
    flags = [(p, p.requires_grad) for p in model.parameters()]

    def inject(module, inputs, output):
        hidden = extract_hidden_tensor(output)
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != encoded['input_ids'].shape[1]:
            raise ValueError('Attribution requires one complete prompt')
        leaf = hidden[0, -1].detach().float().requires_grad_(True)
        replacement = torch.cat((hidden[:, :-1].detach(), leaf.to(hidden.dtype)[None, None]), dim=1)
        state['leaf'] = leaf
        return (replacement, *output[1:]) if isinstance(output, tuple) else replacement

    handle = None
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        handle = modules[layer].register_forward_hook(inject)
        with torch.enable_grad():
            logits = model(**encoded, use_cache=False).logits[0, -1].float()
            target_logits = donor_logits.detach().to(logits.device).float()
            if target_logits.shape != logits.shape or not torch.isfinite(target_logits).all():
                raise ValueError('Need finite aligned donor vocabulary logits')
            target_logp = target_logits.log_softmax(-1)
            loss = F.kl_div(logits.log_softmax(-1), target_logp, reduction='sum', log_target=True)
            leaf = state['leaf']
            gradient, = torch.autograd.grad(loss, leaf)
        own = leaf.detach().cpu()
        donor = donor.detach().float().cpu()
        if donor.shape != own.shape or not torch.isfinite(donor).all():
            raise ValueError('Need finite aligned donor activations')
        gradient = gradient.detach().float().cpu()
        scores = -gradient * (donor - own)
        if not torch.isfinite(scores).all() or not torch.isfinite(loss):
            raise ValueError('Non-finite attribution; do not generate from this selection')
        return dict(scores=scores, gradient=gradient, recipient=own,
                    donor=donor, kl_before=float(loss.detach()), logits=logits.detach().cpu())
    finally:
        if handle is not None:
            handle.remove()
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def select_scores(scores, count):
    """Fixed size, nested ranking; keep sign rather than absolute importance."""
    if scores.ndim != 1 or not torch.isfinite(scores).all() or not 0 <= count <= scores.numel():
        raise ValueError('Invalid attribution scores or budget')
    mask = torch.zeros(scores.numel(), dtype=torch.bool)
    mask[torch.argsort(scores.cpu(), descending=True, stable=True)[:count]] = True
    return mask
