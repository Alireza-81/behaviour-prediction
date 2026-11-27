"""Causal neural risk scores for a single checkpoint trajectory."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .invariant_trajectory import matrix
from .source_only import observations
from .trajectory_forecast import smooth


class RiskNetwork(nn.Module):
    """Predict a next-window event at each observed checkpoint."""

    def __init__(self, kind, width, features):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.readout = nn.Linear(features, 1)
            return
        self.embed = nn.Linear(features, width)
        self.readout = nn.Linear(width, 1)
        if kind == "mlp":
            self.transition = nn.Linear(width, width)
        elif kind == "conv":
            self.transition = nn.Conv1d(width, width, kernel_size=3)
            self.dilated = nn.Conv1d(width, width, kernel_size=3, dilation=2)
        elif kind in ("gru", "gru_aux"):
            self.transition = nn.GRU(width, width, batch_first=True)
            if kind == "gru_aux":
                self.rate = nn.Linear(width, 1)
        elif kind == "attention":
            self.transition = nn.MultiheadAttention(width, 2, batch_first=True)
            self.feed_forward = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        else:
            raise ValueError(f"Unknown neural predictor architecture {kind}")

    def forward(self, sequence, *, return_rate=False):
        if self.kind == "linear":
            return self.readout(sequence).squeeze(-1)
        hidden = F.silu(self.embed(sequence))
        if self.kind == "mlp":
            hidden = hidden + F.silu(self.transition(hidden))
        elif self.kind == "conv":
            state = hidden.transpose(1, 2)
            state = state + F.silu(self.transition(F.pad(state, (2, 0))))
            state = state + F.silu(self.dilated(F.pad(state, (4, 0))))
            hidden = state.transpose(1, 2)
        elif self.kind in ("gru", "gru_aux"):
            recurrent, _ = self.transition(hidden)
            hidden = hidden + recurrent
        elif self.kind == "attention":
            length = hidden.shape[1]
            future = torch.ones((length, length), dtype=torch.bool, device=hidden.device).triu(1)
            attention, _ = self.transition(hidden, hidden, hidden, attn_mask=future, need_weights=False)
            hidden = hidden + attention
            hidden = hidden + self.feed_forward(hidden)
        logits = self.readout(hidden).squeeze(-1)
        if return_rate:
            return logits, self.rate(hidden).squeeze(-1)
        return logits


def normalize(values, center, scale):
    """Center from the fitting fold and compress large deviations."""
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    return torch.from_numpy(np.arcsinh((values - center) / scale).astype(np.float32))


def load_network(member):
    model = RiskNetwork(member["architecture"], member["width"], len(member["center"]))
    weights = {key: torch.tensor(value, dtype=torch.float32) for key, value in member["weights"].items()}
    model.load_state_dict(weights, strict=True)
    return model.eval()


@torch.inference_mode()
def risk(member, sequence):
    return load_network(member)(normalize(sequence, member["center"], member["scale"])).numpy().astype(np.float64)


def score_prefix(points, bank):
    """Keep everything after the current checkpoint outside the model."""
    features = observations(points)
    steps = np.array([p["step"] for p in points[1:]], dtype=int)
    votes = np.zeros(len(steps), dtype=int)
    if len(steps):
        values = matrix(features, bank["columns"], len(steps))[None, ...]
        for member in bank["members"]:
            scores = smooth(risk(member, values)[0], bank["smoothing"])
            votes += scores > member["cutoff"]
    alarm = steps[(steps >= bank["min_step"]) & (votes >= bank["quorum"] * len(bank["members"]))]
    return dict(first_alert=int(alarm[0]) if len(alarm) else None, horizon=bank["horizon"],
                scored_steps=steps.tolist(), votes=votes.tolist())
