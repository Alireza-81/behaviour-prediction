"""Causal prompt-pair encoder with discrete forecasts for the next two intervals."""
import numpy as np
import torch
from torch import nn

MODES = ("step", "pooled", "learned")


def input_tensor(profile):
    """Interpolate native layers onto nine relative depths and preserve each prompt."""
    x = np.asarray(profile, dtype=np.float32)
    if x.ndim != 4 or x.shape[-2:] != (4, 16) or not np.isfinite(x).all():
        raise ValueError("Expected checkpoint × layer × metric × 16 original prompts")
    depths = np.linspace(0, x.shape[1]-1, 9)
    low = depths.astype(int)
    high = np.minimum(low + 1, x.shape[1]-1)
    weight = (depths - low).astype(np.float32)[None,:,None,None]
    z = x[:,low] * (1 - weight) + x[:,high] * weight
    # Relative L2, cosine distance and norm ratio have a common meaning across widths.
    # Raw L2 grows with the hidden-state scale and is omitted from this first comparison.
    geom = np.stack((np.log1p(4*z[:,:,0]), np.log1p(8*z[:,:,2]),
                     np.log(np.maximum(z[:,:,3],1e-6))),axis=-1)
    return np.ascontiguousarray(geom.reshape(len(x),9,8,6).transpose(0,2,1,3))


class PairSurvival(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "step":
            self.head = nn.Linear(8, 2)
        elif mode == "pooled":
            self.head = nn.Linear(9*6*2,2)
        elif mode == "learned":
            self.depth_encoder = nn.Sequential(nn.Conv1d(6,12,3,padding=1),nn.GELU())
            self.depth_weights = nn.Linear(12,1)
            self.depth_bias = nn.Parameter(torch.zeros(9))
            self.pair_weights = nn.Linear(12,1)
            self.history = nn.GRU(12,16,batch_first=True)
            self.head = nn.Linear(16,2)
        else:
            raise ValueError(f"Unknown model {mode}")

    def forward(self, x):
        n,t,pairs,depth,metrics = x.shape
        if self.mode == "step":
            h = torch.eye(8,device=x.device,dtype=x.dtype)[:t].expand(n,-1,-1)
        elif self.mode == "pooled":
            mean = x.mean(2).flatten(2)
            lag = torch.cat((mean[:,:1],mean[:,:-1]),dim=1)
            h = torch.cat((mean,mean-lag),dim=2)
        else:
            h = self.depth_encoder(x.reshape(n*t*pairs,depth,metrics).transpose(1,2)).transpose(1,2)
            w = torch.softmax(self.depth_weights(h).squeeze(-1) + self.depth_bias, dim=-1)
            h = (h * w[...,None]).sum(1).reshape(n,t,pairs,-1)
            w = torch.softmax(self.pair_weights(h),dim=2)
            h,_ = self.history((h*w).sum(2))
        return self.head(h)


def risk(logits):
    """P(event by the second interval), conditional on being event-free at the prefix."""
    return -torch.expm1(torch.nn.functional.logsigmoid(-logits).sum(-1))


@torch.inference_mode()
def score_profile(profile, artifact, *, corpus=None):
    """Score one observed prefix using serialized source weights and thresholds."""
    steps = list(range(5, min((len(profile)-1)*5, artifact["last_step"])+1, 5))
    if not steps:
        return dict(first_alert=None, steps=[], votes=[], risk=[])
    x = torch.from_numpy(input_tensor(profile[1:len(steps)+1]))[None]
    members = artifact["full"] if corpus is None else artifact["by_corpus"][corpus]
    votes = torch.zeros(len(steps), dtype=torch.int)
    scores = []
    for member in members:
        z = ((x - x.new_tensor(member["center"])) / x.new_tensor(member["scale"])).clamp(-8,8)
        views = []
        for state in member["states"]:
            model = PairSurvival(artifact["mode"]).eval()
            model.load_state_dict({k: torch.as_tensor(v) for k,v in state.items()})
            views.append(risk(model(z))[0])
        v = torch.stack(views).mean(0)
        scores.append(v)
        votes += v > member["cutoff"]
    first = next((t for t,v in zip(steps,votes.tolist(),strict=True) if 2*v>len(members)),None)
    return dict(first_alert=first, steps=steps, votes=votes.tolist(), risk=torch.stack(scores).mean(0).tolist())
