"""Rotary Positional Embeddings (RoPE) for PyTorch."""
import torch


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    cos = cos_sin[:S, :, 0]
    sin = cos_sin[:S, :, 1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos
    out = torch.stack([rotated1, rotated2], dim=-1)
    return out.flatten(-2)
