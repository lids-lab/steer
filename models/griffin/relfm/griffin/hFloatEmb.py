import os

import torch
import torch.nn as nn


class SimpleRepeater(nn.Module):

    def __init__(self, dim=256) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("scaler", torch.logspace(-2, 1, dim))
        self.floatenc = getfloatenc(dim)

    def forward(self, feat: torch.Tensor):
        assert feat.ndim == 1
        if torch.is_floating_point(feat):
            return self.floatenc(feat.unsqueeze(-1))
        else:  # categorical, always positive
            feat = feat.unsqueeze(1).to(torch.float) * self.scaler
            return torch.cos(feat)


def getfloatenc(hiddim: int = 256):
    middim = int(hiddim**0.5 + 0.1)
    FloatEnc = nn.Sequential(nn.Linear(1, middim), nn.LayerNorm(middim, elementwise_affine=False), nn.SiLU(inplace=True), nn.Linear(middim, middim), nn.LayerNorm(middim, elementwise_affine=False), nn.SiLU(inplace=True), nn.Linear(middim, hiddim), nn.LayerNorm(hiddim, elementwise_affine=False), nn.SiLU(inplace=True),)
    weights_path = os.path.join(os.path.dirname(__file__), f"floatenc-{hiddim}.pt")
    FloatEnc.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    FloatEnc.eval()
    for p in FloatEnc.parameters():
        p.requires_grad_(False)
    return FloatEnc


def getfloatdec(hiddim: int = 256):
    FloatDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, 1, bias=False))
    weights_path = os.path.join(os.path.dirname(__file__), f"floatdec-{hiddim}.pt")
    FloatDec.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    FloatDec.eval()
    for p in FloatDec.parameters():
        p.requires_grad_(False)
    return FloatDec
