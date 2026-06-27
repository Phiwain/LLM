"""Rotary Positional Embeddings (RoPE) for PyTorch."""
import torch


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0, device=None):
    """Precompute cos/sin tables for RoPE. Returns (max_seq_len, d_head//2, 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (S, D//2)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (S, D//2, 2)


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_heads, S, d_head).

    cos_sin: (S, d_head//2, 2)
    """
    B, H, S, D = x.shape
    cos = cos_sin[:S, :, 0]  # (S, D//2)
    sin = cos_sin[:S, :, 1]

    x1 = x[..., 0::2]  # (B, H, S, D//2)
    x2 = x[..., 1::2]

    cos = cos[None, None, :, :]  # (1,1,S,D//2)
    sin = sin[None, None, :, :]

    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos

    out = torch.stack([rotated1, rotated2], dim=-1)  # (B,H,S,D//2,2)
    return out.flatten(-2)  # (B,H,S,D)
