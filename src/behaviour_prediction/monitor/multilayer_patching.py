"""Shared residual-coordinate attribution and live patching across transformer blocks."""
import torch
import torch.nn.functional as F
from .dynamic_gradient_patching import step_context
from behaviour_prediction.monitoring.activations import resolve_transformer_layers, extract_hidden_tensor


def sequence_states(model, encoded, prompt_length):
    states = {}; handles = []
    def hook(index):
        def capture(module, args, output):
            states[index] = extract_hidden_tensor(output)[0, prompt_length-1:].detach().cpu().clone()
        return capture
    try:
        for i, module in enumerate(resolve_transformer_layers(model)):
            handles.append(module.register_forward_hook(hook(i)))
        with torch.no_grad():
            logits = model(**encoded, use_cache=False).logits[0, prompt_length-1:].detach().cpu()
        return states, logits
    finally:
        for handle in handles: handle.remove()


def all_layer_scores(model, encoded, prompt_length, donors, donor_logits, *, include_gradient_stats=False):
    """One backward pass; preserve the graph between blocks (never detach each block).

    Each score is a local first-order benefit of replacing that block output,
    evaluated at the unpatched recipient. Simultaneous replacements can interact.
    """
    if model.training or encoded['input_ids'].shape[0] != 1:
        raise ValueError('Need one example and evaluation mode')
    layers = resolve_transformer_layers(model)
    if set(donors) != set(range(len(layers))): raise ValueError('Missing donor layers')
    flags = [(p, p.requires_grad) for p in model.parameters()]
    states = {}; handles = []
    def hook(index):
        def capture(module, args, output):
            hidden = extract_hidden_tensor(output)
            # Frozen parameters otherwise leave the entire forward graph disabled.
            # Only the first site becomes a leaf; later sites retain upstream paths.
            if index == 0:
                hidden = hidden.detach().requires_grad_(True)
                output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            states[index] = hidden
            return output
        return capture
    try:
        for p, _ in flags: p.requires_grad_(False)
        for i, module in enumerate(layers): handles.append(module.register_forward_hook(hook(i)))
        with torch.enable_grad():
            logits = model(**encoded, use_cache=False).logits[0, prompt_length-1:].float()
            target = donor_logits.to(logits.device).float()
            if logits.shape != target.shape: raise ValueError('Reference alignment differs')
            loss = F.kl_div(logits.log_softmax(-1), target.log_softmax(-1), log_target=True,
                            reduction='sum') / logits.shape[0]
            gradients = torch.autograd.grad(loss, tuple(states[i] for i in range(len(layers))))
        scores = []; gradient_abs = []; gradient_signed = []
        for i, gradient in enumerate(gradients):
            own = states[i][0, prompt_length-1:].detach().float().cpu()
            donor = donors[i].float()
            if donor.shape != own.shape: raise ValueError('Donor alignment differs')
            g = gradient[0, prompt_length-1:].detach().float().cpu()
            value = (-g * (donor-own)).sum(0)
            scores.append(value)
            if include_gradient_stats:
                # Sum position-wise magnitudes for the same mean-token KL loss.
                # Earlier positions can affect later-token losses through attention.
                gradient_abs.append(g.abs().sum(0))
                gradient_signed.append(g.sum(0))
        scores = torch.stack(scores)
        if not torch.isfinite(scores).all() or not torch.isfinite(loss): raise ValueError('Non-finite scores')
        result = dict(scores=scores, kl_before=float(loss.detach()), prediction_positions=logits.shape[0])
        if include_gradient_stats:
            result.update(gradient_abs=torch.stack(gradient_abs), gradient_signed=torch.stack(gradient_signed))
            if not torch.isfinite(result['gradient_abs']).all(): raise ValueError('Non-finite raw gradients')
        return result
    finally:
        for handle in handles: handle.remove()
        for p, flag in flags: p.requires_grad_(flag)


def live_forward(model, step, inputs, cache, *, layers=(), donors=None, masks=None):
    states = {}; handles = []; modules = resolve_transformer_layers(model)
    def hook(index):
        def apply(module, args, output):
            hidden = extract_hidden_tensor(output)
            if donors is None:
                states[index] = hidden[0, -1].detach().clone()
                return output
            vector = donors[index].to(hidden)
            replacement = hidden.clone()
            mask = masks[index].to(hidden.device)
            replacement[0, -1] = torch.where(mask, vector, hidden[0, -1])
            return (replacement, *output[1:]) if isinstance(output, tuple) else replacement
        return apply
    try:
        for i in layers: handles.append(modules[i].register_forward_hook(hook(i)))
        with step_context(model, step), torch.no_grad():
            output = model(**inputs, past_key_values=cache, use_cache=True)
        return output.logits[0,-1], output.past_key_values, states
    finally:
        for handle in handles: handle.remove()


def multilayer_generate(model, encoded, *, recipient_step, donor_step, limit, masks=None):
    """Same fixed sites each token; donor follows recipient tokens with its own cache."""
    if limit < 1 or model.training: raise ValueError('Positive limit and evaluation mode required')
    if encoded['input_ids'].shape[0] != 1: raise ValueError('Batch size must be one')
    modules = resolve_transformer_layers(model)
    masks = {} if masks is None else masks
    for i, mask in masks.items():
        if not 0 <= i < len(modules) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError('Invalid coordinate mask')
    eos = model.generation_config.eos_token_id
    if eos is None: eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, (list,tuple)) else [] if eos is None else [eos])
    ids = encoded['input_ids']; attention = encoded.get('attention_mask', torch.ones_like(ids))
    rc = dc = None; generated = []
    for _ in range(limit):
        inputs = dict(input_ids=ids, attention_mask=attention)
        donors = None
        if masks:
            _, dc, donors = live_forward(model, donor_step, inputs, dc, layers=sorted(masks))
        logits, rc, _ = live_forward(model, recipient_step, inputs, rc, layers=sorted(masks), donors=donors, masks=masks)
        token = int(logits.argmax()); generated.append(token)
        if token in eos: break
        ids = torch.tensor([[token]], device=ids.device, dtype=ids.dtype)
        attention = torch.cat((attention, torch.ones_like(ids)), dim=1)
    return generated


def select_multilayer(scores, budgets, seeds):
    """Absolute total budgets. Distributed = nearly equal quotas across depth thirds."""
    if scores.ndim != 2 or not torch.isfinite(scores).all(): raise ValueError('Invalid scores')
    depth, width = scores.shape
    if depth < 3: raise ValueError('Need at least three blocks')
    regions = [list(range(i*depth//3, (i+1)*depth//3)) for i in range(3)]
    result = {}
    def top(layers, count):
        flat = [(i,j) for i in layers for j in range(width)]
        order = torch.argsort(scores[layers].flatten(), descending=True, stable=True)
        return [flat[k] for k in order[:count].tolist()]
    for budget in budgets:
        if budget < 1 or budget > width: raise ValueError('Budget must also fit the layer-only comparator')
        choices = dict(layer31=top([depth-1],budget), global_all=top(list(range(depth)),budget),
                       earlier=top(list(range(depth-1)),budget))
        choices['distributed'] = sum((top(region, budget//3 + (i < budget%3)) for i,region in enumerate(regions)), [])
        for method, coordinates in choices.items():
            name = f'{method}_{budget}'; selected = {}
            for layer, coordinate in coordinates: selected.setdefault(str(layer),[]).append(coordinate)
            result[name] = {k:sorted(v) for k,v in sorted(selected.items(), key=lambda item:int(item[0]))}
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                # Same per-layer counts as the selected method; local deterministic RNG.
                random = {k:sorted(torch.randperm(width,generator=generator)[:len(v)].tolist()) for k,v in result[name].items()}
                result[f'{name}_random{seed}'] = random
    return result
