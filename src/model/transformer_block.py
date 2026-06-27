"""Transformer block: RMSNorm -> Attention -> RMSNorm -> MoE."""
from typing import Optional
import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import Attention
from .moe import MoELayer


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg.d_model, cfg.n_heads, use_flash=cfg.use_flash_attention)

        self.moe_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoELayer(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k)

        self.use_grad_checkpoint = cfg.use_grad_checkpoint

    def forward(self, x: torch.Tensor, cos_sin: Optional[torch.Tensor] = None):
        if self.use_grad_checkpoint and self.training:
            x = x + torch.utils.checkpoint.checkpoint(
                lambda h: self.attn(self.attn_norm(h), cos_sin), x, use_reentrant=False
            )
            moe_out, aux = torch.utils.checkpoint.checkpoint(
                lambda h: self.moe(self.moe_norm(h)), x, use_reentrant=False
            )
        else:
            x = x + self.attn(self.attn_norm(x), cos_sin)
            moe_out, aux = self.moe(self.moe_norm(x))

        x = x + moe_out
        return x, aux
