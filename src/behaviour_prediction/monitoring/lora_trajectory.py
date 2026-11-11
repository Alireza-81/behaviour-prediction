"""Measure LoRA weight movement without materializing dense update matrices."""
from collections import defaultdict

import torch

from .weight_diffs import save_metrics_json


def effective_delta_squared(current, reference, scale):
    """Squared Frobenius norm of scale * (B @ A - reference_B @ reference_A)."""
    a, b = current
    ref_a, ref_b = reference
    left = torch.cat((b, -ref_b), dim=1).double()
    right = torch.cat((a, ref_a), dim=0).double()
    squared = ((left.T @ left) * (right @ right.T)).sum().item()
    return max(0.0, squared) * scale**2


class LoRAWeightTrajectory:
    def __init__(self, output_dir, layer_index):
        self.output_dir = output_dir
        self.layer_index = layer_index
        self.initial = None
        self.previous = None

    def capture(self, model, step, epoch):
        current = {}
        for name, module in model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            for adapter, a in module.lora_A.items():
                b = module.lora_B[adapter]
                if a.weight.requires_grad and b.weight.requires_grad:
                    current[(name, adapter)] = (
                        a.weight.detach().to(device="cpu", copy=True),
                        b.weight.detach().to(device="cpu", copy=True),
                    )
        if not current:
            raise ValueError("No trainable LoRA A/B pairs found")
        if self.initial is None:
            if step != 0:
                raise ValueError("LoRA trajectory requires a step-zero baseline")
            self.initial = current
        if current.keys() != self.initial.keys():
            raise ValueError("Trainable LoRA pairs changed during training")
        previous = self.previous if self.previous is not None else self.initial
        totals = defaultdict(float)
        modules = []
        model_modules = dict(model.named_modules())
        for (name, adapter), factors in current.items():
            layer = self.layer_index(name)
            scale = float(model_modules[name].scaling[adapter])
            for reference_name, reference in (("initial", self.initial), ("previous", previous)):
                reference_factors = reference[(name, adapter)]
                squared = dict(
                    lora_A=(factors[0].float() - reference_factors[0].float()).square().sum().item(),
                    lora_B=(factors[1].float() - reference_factors[1].float()).square().sum().item(),
                    effective=effective_delta_squared(factors, reference_factors, scale),
                )
                for component, value in squared.items():
                    totals[(layer, reference_name, component)] += value
                    totals[("all", reference_name, component)] += value
                    modules.append(dict(module=name, adapter=adapter, layer=layer,
                                        reference=reference_name, component=component,
                                        l2_drift=value**0.5))
        layers = [dict(layer=layer, reference=reference, component=component, l2_drift=value**0.5)
                  for (layer, reference, component), value in totals.items()]
        path = self.output_dir / "weight_trajectory" / f"step-{step:06d}.json"
        temporary = path.with_suffix(".tmp")
        save_metrics_json(dict(step=step, epoch=epoch, layers=layers, modules=modules), temporary)
        temporary.replace(path)
        self.previous = current
        print(f"Saved LoRA weight metrics: {path}", flush=True)
