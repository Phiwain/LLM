"""RMSNorm — Root Mean Square Layer Normalization."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
