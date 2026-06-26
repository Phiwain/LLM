"""Multi-Head Attention with RoPE, causal masking, and Flash Attention (SDPA)."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA = True
except ImportError:
    _HAS_SDPA = False


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_flash: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_flash = use_flash

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        H = self.d_head

        q = self.wq(x).view(B, S, self.n_heads, H).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, H).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, H).transpose(1, 2)

        if cos_sin is not None:
            q = apply_rope(q, cos_sin)
            k = apply_rope(k, cos_sin)

        if self.use_flash and _HAS_SDPA:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, -1e9)
            attn = F.softmax(scores, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)
