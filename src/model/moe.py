"""Mixture-of-Experts layer with Top-K routing and load-balancing auxiliary loss.

PyTorch implementation optimized for A100:
  - Top-2 token routing (Switch Transformer style)
  - SwiGLU expert FFN (gate_proj, up_proj, down_proj)
  - Load-balancing auxiliary loss (Switch Transformer formulation)
  - Sparse dispatch: only computes assigned experts per token (memory-efficient)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """MoE with sparse per-expert dispatch.

    Instead of computing all experts on all tokens, this groups tokens by
    their assigned expert and runs each expert only on its subset of tokens.
    This gives O(top_k * N) compute instead of O(E * N).
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff

        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert weights as a single stacked tensor for efficient dispatch
        # gate_w: (E, D, d_ff), up_w: (E, D, d_ff), down_w: (E, d_ff, D)
        self.gate_w = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.up_w = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.down_w = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        nn.init.normal_(self.gate_w, std=0.02)
        nn.init.normal_(self.up_w, std=0.02)
        nn.init.normal_(self.down_w, std=0.02)

    def forward(self, x: torch.Tensor):
        """Forward pass with sparse expert dispatch.

        Args:
            x: (B, S, D)
        Returns:
            output: (B, S, D)
            aux_loss: scalar tensor
        """
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)

        # Router
        router_logits = self.router(x_flat)              # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        # Top-K selection
        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)  # (N, top_k)
        topk_probs = F.softmax(topk_vals, dim=-1)  # (N, top_k)

        # Flatten for dispatch
        # Each token appears top_k times, once per selected expert
        token_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)  # (N*top_k,)
        expert_indices = topk_idx.reshape(-1)  # (N*top_k,)
        weights = topk_probs.reshape(-1)      # (N*top_k,)

        # Gather input tokens for each dispatch
        dispatched_x = x_flat[token_indices]  # (N*top_k, D)

        # Sort by expert index for grouped computation
        sorted_order = torch.argsort(expert_indices)
        sorted_expert_ids = expert_indices[sorted_order]
        sorted_x = dispatched_x[sorted_order]
        sorted_weights = weights[sorted_order]

        # Find boundaries for each expert in the sorted array
        # counts[e] = number of tokens assigned to expert e
        counts = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        # cumsum gives the end index of each expert's group
        cumsum = torch.cumsum(counts, dim=0)
        starts = torch.cat([torch.tensor([0], device=x.device), cumsum[:-1]])

        # Compute each expert only on its assigned tokens
        output_flat = torch.zeros(N * self.top_k, D, device=x.device, dtype=x.dtype)

        for e in range(self.n_experts):
            s = starts[e].item()
            c = counts[e].item()
            if c == 0:
                continue
            expert_input = sorted_x[s : s + c]  # (c, D)
            # Expert FFN: SwiGLU
            gate = F.silu(expert_input @ self.gate_w[e])  # (c, d_ff)
            up = expert_input @ self.up_w[e]                # (c, d_ff)
            expert_out = (gate * up) @ self.down_w[e]       # (c, D)
            output_flat[sorted_order[s : s + c]] = expert_out.to(output_flat.dtype)

        # Apply routing weights
        weighted_out = output_flat * sorted_weights.unsqueeze(-1)

        # Scatter-add back to original token positions
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        output.index_add_(0, token_indices, weighted_out)

        # Load-balancing auxiliary loss (Switch Transformer)
        # f_i = fraction of tokens dispatched to expert i
        # P_i = average router probability for expert i
        f = counts.float() / (N * self.top_k)
        P = router_probs.mean(dim=0)
        aux_loss = self.n_experts * (f * P).sum()

        output = output.reshape(B, S, D)
        return output, aux_loss
