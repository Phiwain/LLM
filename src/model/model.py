"""Full MoE Language Model: token embedding + N transformer blocks + LM head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock
from .rope import precompute_rope_freqs


class MoELLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        # Precompute RoPE table (moved to device in forward)
        self.register_buffer(
            "_cos_sin",
            precompute_rope_freqs(cfg.d_head, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor):
        """Forward pass.

        Args:
            tokens: (B, S) int token IDs
        Returns:
            logits: (B, S, vocab_size)
            total_aux_loss: scalar tensor
        """
        x = self.embed(tokens)  # (B, S, D)

        total_aux = torch.tensor(0.0, device=tokens.device, dtype=x.dtype)
        for block in self.blocks:
            x, aux = block(x, self._cos_sin)
            total_aux = total_aux + aux

        x = self.norm(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed.weight)

        return logits, total_aux / self.cfg.n_layers

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8, eos_id: int = 2):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            if tokens.shape[1] > self.cfg.max_seq_len:
                tokens = tokens[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(tokens)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break
        return tokens
