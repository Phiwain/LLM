"""Mixture-of-Experts layer with Top-K routing and load-balancing auxiliary loss.

Sparse dispatch: only computes assigned experts per token (memory-efficient).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff

        self.router = nn.Linear(d_model, n_experts, bias=False)

        self.gate_w = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.up_w = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.down_w = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        nn.init.normal_(self.gate_w, std=0.02)
        nn.init.normal_(self.up_w, std=0.02)
        nn.init.normal_(self.down_w, std=0.02)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_vals, dim=-1)

        token_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        expert_indices = topk_idx.reshape(-1)
        weights = topk_probs.reshape(-1)

        dispatched_x = x_flat[token_indices]

        sorted_order = torch.argsort(expert_indices)
        sorted_expert_ids = expert_indices[sorted_order]
        sorted_x = dispatched_x[sorted_order]
        sorted_weights = weights[sorted_order]

        counts = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        cumsum = torch.cumsum(counts, dim=0)
        starts = torch.cat([torch.tensor([0], device=x.device), cumsum[:-1]])

        output_flat = torch.zeros(N * self.top_k, D, device=x.device, dtype=x.dtype)

        for e in range(self.n_experts):
            s = starts[e].item()
            c = counts[e].item()
            if c == 0:
                continue
            expert_input = sorted_x[s : s + c]
            gate = F.silu(expert_input @ self.gate_w[e])
            up = expert_input @ self.up_w[e]
            expert_out = (gate * up) @ self.down_w[e]
            output_flat[sorted_order[s : s + c]] = expert_out.to(output_flat.dtype)

        weighted_out = output_flat * sorted_weights.unsqueeze(-1)

        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        output.index_add_(0, token_indices, weighted_out)

        f = counts.float() / (N * self.top_k)
        P = router_probs.mean(dim=0)
        aux_loss = self.n_experts * (f * P).sum()

        output = output.reshape(B, S, D)
        return output, aux_loss
