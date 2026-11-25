"""A fixed TopK dictionary and linear projection accounting for activation replay."""
import torch
from torch import nn
from torch.nn import functional as F


class TopKDictionary(nn.Module):
    def __init__(self, hidden, features, topk):
        super().__init__()
        if not 0 < topk <= features:
            raise ValueError("Expected 0 < topk <= feature count")
        self.topk = topk
        self.encoder = nn.Linear(hidden, features)
        self.decoder = nn.Parameter(F.normalize(torch.randn(features,hidden),dim=1))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.register_buffer("center",torch.zeros(hidden))
        self.register_buffer("scale",torch.ones(()))
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder)
            self.encoder.bias.zero_()

    def normalize(self, x):
        return (x - self.center)/self.scale

    def encode(self,z):
        v,i = self.encoder(z-self.bias).relu().topk(self.topk,dim=-1)
        return torch.zeros_like(z[..., :1]).expand(*z.shape[:-1],len(self.decoder)).clone().scatter_(-1,i,v)

    def forward(self,x):
        z=self.encode(self.normalize(x))
        recon=(z @ F.normalize(self.decoder,dim=1)+self.bias)*self.scale+self.center
        return z,recon

    def projection_weights(self,axes):
        return self.scale * F.normalize(self.decoder,dim=1) @ F.normalize(axes,dim=1).T


def contrast_weights(baseline):
    """Each row is a fixed baseline-scaled contrast on eight ordered prompt pairs."""
    scale = baseline.norm(dim=1).reshape(-1,2).mean(1)
    if len(scale)!=8 or (scale<=1e-8).any():
        raise ValueError("Expected eight nondegenerate risky–benign pairs")
    weights=torch.zeros(4,8,2,dtype=baseline.dtype)
    weights[0]=.5
    weights[1,:,0]=1
    weights[1,:,1]=-1
    weights[2,:,0]=1
    weights[3,:,1]=1
    return (weights/scale[None,:,None]/len(scale)).reshape(4,16)


def r_squared(x,y,center):
    denominator = (x-center).square().sum().item()
    return None if denominator<=1e-12 else 1-(x-y).square().sum().item()/denominator
