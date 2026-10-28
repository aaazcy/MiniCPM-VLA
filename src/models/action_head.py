"""src/models/action_head.py — action heads appended to the LLM last-token hidden.

Two B-family heads are supported:

  * ActionHead (continuous, legacy):
        Linear(hidden,512)->LN->GELU->Linear(512,256)->LN->GELU->Linear(256,7)->Tanh
    Direct MSE regression.  Prone to mean-collapse when the target distribution is
    multi-modal / dominated by near-zero actions.

  * DiscreteActionHead (default, aligned with OpenVLA):
        7 action dims x `num_bins` uniform bins in [-1, 1]
        Linear(hidden,512)->LN->GELU->Linear(512,256)->LN->GELU->Linear(256,7*num_bins)
    Trained with cross-entropy on the quantized demonstration action; rollout decodes
    the argmax/sampled bin.  Same action parameterisation as OpenVLA's native head.

A-family (OpenVLA) keeps its native discretized head; `NativeActionHead` below is only
a thin adapter so the trainer can share one call path.
"""
from __future__ import annotations

import torch
from torch import nn


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, dims=(512, 256), action_dim: int = 7,
                 init_std: float = 0.02):
        super().__init__()
        d1, d2 = dims
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Linear(d1, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Linear(d2, action_dim),
            nn.Tanh(),
        )
        self.action_dim = action_dim
        self.reset_parameters(init_std)

    def reset_parameters(self, std: float = 0.02) -> None:
        for name, p in self.net.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.trunc_normal_(p, std=std)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, last_token_hidden: torch.Tensor) -> torch.Tensor:
        """last_token_hidden: (B, H) -> actions (B, 7) in [-1, 1]."""
        # cast to the head's dtype (LLM hidden may be bf16 while head is fp32)
        last_token_hidden = last_token_hidden.to(self.net[0].weight.dtype)
        return self.net(last_token_hidden)


class DiscreteActionHead(nn.Module):
    """7 x num_bins categorical head over uniform bins in [-1, 1].

    SFT: cross-entropy on the quantized demonstration action (robust against
    multi-modal / near-zero target collapse).  Rollout: argmax or categorical
    sampling over bins.  GRPO: categorical log-probability, matching OpenVLA.
    """

    def __init__(self, hidden_dim: int = 4096, dims=(512, 256), action_dim: int = 7,
                 num_bins: int = 256, init_std: float = 0.02):
        super().__init__()
        d1, d2 = dims
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Linear(d1, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Linear(d2, action_dim * num_bins),
        )
        # same uniform bins as OpenVLA's native head
        self.register_buffer("bin_centers", torch.linspace(-1.0, 1.0, num_bins),
                             persistent=False)
        self.reset_parameters(init_std)

    def reset_parameters(self, std: float = 0.02) -> None:
        for name, p in self.net.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.trunc_normal_(p, std=std)
            elif "bias" in name:
                nn.init.zeros_(p)
        # keep the initial bin distribution close to uniform (small logits)
        last = self.net[-1]
        nn.init.trunc_normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, last_token_hidden: torch.Tensor) -> torch.Tensor:
        """(B, H) -> logits (B, action_dim, num_bins)."""
        last_token_hidden = last_token_hidden.to(self.net[0].weight.dtype)
        return self.net(last_token_hidden).view(-1, self.action_dim, self.num_bins)

    def quantize(self, actions: torch.Tensor) -> torch.Tensor:
        """(B,7) in [-1,1] -> bin indices (B,7) in [0, num_bins-1]."""
        q = ((actions.clamp(-1, 1) + 1) / 2 * (self.num_bins - 1)).round().long()
        return q.clamp(0, self.num_bins - 1)

    def expected_action(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable E[action] under the bin distribution -> (B,7)."""
        probs = torch.softmax(logits.float(), dim=-1)
        centers = self.bin_centers.float().to(logits.device)
        return (probs * centers).sum(-1)

    def logp(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Categorical log-probability of the quantized actions -> (B,)."""
        q = self.quantize(actions.to(logits.device))
        logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.gather(-1, q.unsqueeze(-1)).squeeze(-1).sum(-1)

    @torch.no_grad()
    def decode(self, logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        """(B,7,num_bins) -> actions (B,7). temperature=0 -> argmax."""
        if temperature and temperature > 0:
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            q = torch.multinomial(probs.reshape(-1, self.num_bins), 1)
            q = q.view(-1, self.action_dim)
        else:
            q = logits.argmax(-1)
        return self.bin_centers.float().to(logits.device)[q]

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, temperature: float = 1.0):
        """Sample actions + logp under the sampling distribution (GRPO rollout)."""
        temp = max(float(temperature), 1e-6)
        probs = torch.softmax(logits.float() / temp, dim=-1)
        q = torch.multinomial(probs.reshape(-1, self.num_bins), 1)
        q = q.view(-1, self.action_dim)
        action = self.bin_centers.float().to(logits.device)[q]
        logp = torch.log_softmax(logits.float() / temp, dim=-1)
        logp_old = logp.gather(-1, q.unsqueeze(-1)).squeeze(-1).sum(-1)
        return action, logp_old


class NativeActionHead(nn.Module):
    """Thin adapter so OpenVLA's built-in head can be trained with the same trainer
    code path as the shared heads (same forward signature)."""

    def __init__(self, native_head: nn.Module):
        super().__init__()
        self.native = native_head
        self.action_dim = 7

    def forward(self, last_token_hidden: torch.Tensor) -> torch.Tensor:
        return self.native(last_token_hidden)
