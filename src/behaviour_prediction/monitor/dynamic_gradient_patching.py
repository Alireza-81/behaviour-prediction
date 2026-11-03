"""Sequence attribution and live same-prefix generation using one PEFT model.

Only final-block residual outputs are supported in this first experiment.
Parameter weights are never updated. Baseline and tuned caches remain separate.
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from behaviour_prediction.monitoring.activations import resolve_transformer_layers, extract_hidden_tensor


def step_context(model, step):
    if step not in (0, 20):
        raise ValueError('Expected step0 or step20')
    if not hasattr(model, 'disable_adapter'):
        raise ValueError('Dynamic experiment requires a PEFT model with adapter switching')
    return model.disable_adapter() if step == 0 else nullcontext()


def check_layer(model, layer):
    layers = resolve_transformer_layers(model)
    if layer != len(layers) - 1:
        raise ValueError('This experiment requires the final transformer block')
    if model.training:
        raise ValueError('Use evaluation mode')
    return layers[layer]


def sequence_capture(model, encoded, layer, prompt_length):
    """Capture states/logits predicting every supplied reference-answer token."""
    module = check_layer(model, layer)
    state = {}

    def capture(module, inputs, output):
        state['hidden'] = extract_hidden_tensor(output)[0, prompt_length-1:].detach().cpu().clone()

    handle = module.register_forward_hook(capture)
    try:
        with torch.no_grad():
            output = model(**encoded, use_cache=False)
            logits = output.logits[0, prompt_length-1:].detach().cpu().clone()
    finally:
        handle.remove()
    return state['hidden'], logits


def sequence_replacement_scores(model, encoded, layer, prompt_length, donor_hidden, donor_logits):
    """First-order reduction of mean-token donor KL, summed by coordinate.

    Gradients reach a leaf spanning *every* reference prediction position, not
    just the prompt boundary. Both endpoints receive identical teacher-forced
    tokens. This measures a sequence-level distribution proxy, not judged harm.
    """
    module = check_layer(model, layer)
    if encoded['input_ids'].shape[0] != 1 or not 1 <= prompt_length <= encoded['input_ids'].shape[1]:
        raise ValueError('Need one prompt plus an aligned reference continuation')
    flags = [(p, p.requires_grad) for p in model.parameters()]
    state = {}

    def inject(module, inputs, output):
        hidden = extract_hidden_tensor(output)
        leaf = hidden[0, prompt_length-1:].detach().float().requires_grad_(True)
        replacement = torch.cat((hidden[:, :prompt_length-1].detach(), leaf.to(hidden.dtype)[None]), dim=1)
        state['leaf'] = leaf
        return (replacement, *output[1:]) if isinstance(output, tuple) else replacement

    handle = None
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        handle = module.register_forward_hook(inject)
        with torch.enable_grad():
            logits = model(**encoded, use_cache=False).logits[0, prompt_length-1:].float()
            target = donor_logits.detach().to(logits.device).float()
            if target.shape != logits.shape or not torch.isfinite(target).all():
                raise ValueError('Reference logits are misaligned or non-finite')
            count = logits.shape[0]
            loss = F.kl_div(logits.log_softmax(-1), target.log_softmax(-1), log_target=True, reduction='sum') / count
            leaf = state['leaf']
            gradient, = torch.autograd.grad(loss, leaf)
        own = leaf.detach().cpu()
        donor = donor_hidden.detach().float().cpu()
        if donor.shape != own.shape or not torch.isfinite(donor).all():
            raise ValueError('Reference activations are misaligned or non-finite')
        delta = donor - own
        gradient = gradient.detach().float().cpu()
        token_scores = -gradient * delta
        scores = token_scores.sum(0)
        if not torch.isfinite(scores).all() or not torch.isfinite(loss):
            raise ValueError('Non-finite sequence attribution')
        return dict(scores=scores, token_scores=token_scores, gradient=gradient,
                    recipient=own, donor=donor, kl_before=float(loss.detach()),
                    prediction_positions=count, magnitude_rms=delta.square().mean(0).sqrt())
    finally:
        if handle is not None:
            handle.remove()
        for p, flag in flags:
            p.requires_grad_(flag)


def _forward(model, step, inputs, layer, cache, *, donor=None, mask=None, capture=False):
    """One endpoint pass. Hooks last returned position, including cached decode."""
    module = check_layer(model, layer)
    state = {}

    def hook(module, args, output):
        hidden = extract_hidden_tensor(output)
        if capture:
            state['donor'] = hidden[0, -1].detach().clone()
        if donor is None:
            return output
        vector = donor.to(device=hidden.device, dtype=hidden.dtype)
        if vector.shape != hidden[0, -1].shape:
            raise ValueError('Live donor width mismatch')
        replacement = hidden.clone()
        replacement[0, -1] = vector if mask is None else torch.where(mask.to(hidden.device), vector, hidden[0, -1])
        return (replacement, *output[1:]) if isinstance(output, tuple) else replacement

    handle = module.register_forward_hook(hook) if capture or donor is not None else None
    try:
        with step_context(model, step), torch.no_grad():
            output = model(**inputs, past_key_values=cache, use_cache=True)
        return output.logits[0, -1], output.past_key_values, state.get('donor')
    finally:
        if handle is not None:
            handle.remove()


def dynamic_generate(model, encoded, *, recipient_step, donor_step, layer, limit,
                     mask=None, patch=True, prefill_only=False):
    """Refresh donor on the recipient prefix, maintaining two independent caches.

    Full final-layer replacement reproduces the donor's logits each step. It
    must not be interpreted as a uniquely localized behavioural mechanism.
    """
    check_layer(model, layer)
    if limit < 1 or encoded['input_ids'].shape[0] != 1:
        raise ValueError('Need a positive limit and one prompt')
    if mask is not None and (mask.dtype != torch.bool or mask.ndim != 1):
        raise ValueError('Coordinate mask must be a boolean vector')
    eos = getattr(model.generation_config, 'eos_token_id', None)
    if eos is None:
        eos = getattr(model.config, 'eos_token_id', None)
    eos = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])
    recipient_cache = donor_cache = None
    prompt = encoded['input_ids']
    attention = encoded.get('attention_mask', torch.ones_like(prompt))
    ids = prompt
    generated = []
    for index in range(limit):
        inputs = dict(input_ids=ids, attention_mask=attention)
        donor = None
        if patch and (index == 0 or not prefill_only):
            _, donor_cache, donor = _forward(model, donor_step, inputs, layer, donor_cache, capture=True)
        logits, recipient_cache, _ = _forward(model, recipient_step, inputs, layer, recipient_cache, donor=donor, mask=mask)
        token = int(logits.argmax())
        generated.append(token)
        if token in eos:
            break
        # The donor consumes exactly the token selected by the patched recipient.
        ids = torch.tensor([[token]], device=prompt.device, dtype=prompt.dtype)
        attention = torch.cat((attention, torch.ones((1, 1), device=attention.device, dtype=attention.dtype)), dim=1)
    return generated
