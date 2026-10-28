"""src/training/rewards.py — the fixed reward function (identical for ALL groups).

R = R_success + lambda_dist * R_dist + lambda_collide * R_collide + lambda_action * R_action

  * R_success = +10 on full task success (all subtasks of the instruction chain done)
  * R_dist    = negative L2 distance end-effector -> target
                (CALVIN does not expose a per-subtask target distance; when unavailable
                 we use a subtask-progress proxy: -(1 - progress) * 10. Identical for all
                 groups, so relative comparisons remain valid — see limitations.md.)
  * R_collide = -2 per collision step
  * R_action  = -mean L2 jitter between consecutive actions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Trajectory:
    obs: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False
    success: bool = False
    audio_wav: Optional[np.ndarray] = None   # B3: instruction wav (rollout = update)
    step_logps: List[float] = field(default_factory=list)  # logp_old per step (GRPO ratio)

    @property
    def length(self) -> int:
        return len(self.actions)


def compute_reward(traj: Trajectory, cfg) -> float:
    """Total reward for one rollout trajectory (sum over steps)."""
    r = cfg.reward
    total = 0.0
    steps = max(1, traj.length)
    for i, info in enumerate(traj.infos):
        # ---- success (terminal, +10) ----
        if info.get("success", False):
            total += r.success

        # ---- distance to target ----
        dist = info.get("ee_target_dist")
        if dist is not None and np.isfinite(dist):
            r_dist = -float(dist)
        else:
            # progress proxy: distance shrinks as subtasks complete
            done_sub = float(info.get("subtasks_completed", 0.0))
            tot_sub = max(float(info.get("subtasks_total", 1.0)), 1.0)
            progress = done_sub / tot_sub
            r_dist = -(1.0 - progress) * 10.0 if r.use_progress_proxy_for_dist else 0.0
        total += r.lambda_dist * r_dist

        # ---- collision ----
        if info.get("collision", False):
            total += r.lambda_collide * r.collide_penalty

        # ---- action jitter ----
        if i > 0:
            jitter = float(np.linalg.norm(traj.actions[i] - traj.actions[i - 1]))
            total += r.lambda_action * (-jitter)

    return float(total)


def grpo_advantage(rewards: np.ndarray, group_size: int) -> np.ndarray:
    """GRPO group-relative advantage: (R - mean(R_group)) / (std(R_group) + eps).
    rewards has shape (n_groups, group_size)."""
    mu = rewards.mean(axis=-1, keepdims=True)
    std = rewards.std(axis=-1, keepdims=True)
    return (rewards - mu) / (std + 1e-8)
