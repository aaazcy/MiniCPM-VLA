"""src/models/action_head.py — shared action head appended to the LLM last-token hidden.

Spec-exact architecture (used by BOTH model families):
    Sequential(Linear(hidden, 512), LayerNorm(512), GELU,
               Linear(512, 256), LayerNorm(256), GELU,
               Linear(256, 7), Tanh)
Outputs 7-D continuous action [dx, dy, dz, rx, ry, rz, gripper], normalized [-1, 1].
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


class NativeActionHead(nn.Module):
    """Thin adapter so OpenVLA's built-in head can be trained with the same trainer
    code path as the shared ActionHead (same forward signature)."""

    def __init__(self, native_head: nn.Module):
        super().__init__()
        self.native = native_head
        self.action_dim = 7

    def forward(self, last_token_hidden: torch.Tensor) -> torch.Tensor:
        return self.native(last_token_hidden)
