#!/usr/bin/env python3
"""scripts/smoke_test.py — CPU-only smoke tests (Phase 0, no GPU required).

Validates the pieces that must be correct before any GPU hour is spent:
config loading, split leakage guard, action-head spec, reward function, GRPO
advantage math, and TTS pipeline availability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    assert cond, f"{name} FAILED: {detail}"
    PASS += 1
    print(f"  ok: {name} {detail}")


def test_configs():
    print("[1] configs")
    from src.config import load_config
    for g in ["a0", "a1", "b0", "b1", "b2", "b3"]:
        cfg = load_config(g)
        check(f"cfg {g}", cfg.group.id == g.upper())
    cfg = load_config("b2")
    check("grpo hyperparams locked", (cfg.grpo.group_size, cfg.grpo.rollout_per_group,
                                       cfg.grpo.clip_eps, cfg.grpo.beta_kl) == (8, 2, 0.2, 0.05))
    check("lr split", (cfg.sft.lr_action_head, cfg.sft.lr_lora) == (1e-4, 2e-5))
    check("reward lambdas", (cfg.reward.lambda_dist,
                             cfg.reward.lambda_action) == (0.5, 0.1))
    b3 = load_config("b3")
    check("b3 ablation flag", b3.group.is_ablation and b3.group.audio)
    a1 = load_config("a1")
    check("a1 is vla family", a1.group.family == "vla"
          and a1.group.model_id.endswith("openvla-7b"))


def test_action_head():
    print("[2] action head spec")
    from src.models.action_head import ActionHead
    head = ActionHead(hidden_dim=4096, dims=(512, 256), action_dim=7)
    x = torch.randn(4, 4096)
    y = head(x)
    check("output shape", y.shape == (4, 7), str(y.shape))
    check("output range [-1,1]", bool((y >= -1).all() and (y <= 1).all()))
    check("layer chain", [type(m).__name__ for m in head.net] ==
          ["Linear", "LayerNorm", "GELU", "Linear", "LayerNorm", "GELU", "Linear", "Tanh"])


def test_rewards():
    print("[3] reward function")
    from src.config import load_config
    from src.training.rewards import Trajectory, compute_reward, grpo_advantage
    cfg = load_config("a1")
    traj = Trajectory()
    traj.actions = [np.zeros(7), np.ones(7) * 0.5]
    traj.infos = [{"success": False, "subtasks_completed": 1, "subtasks_total": 2,
                   "action_jitter": 0.0},
                  {"success": True, "subtasks_completed": 2, "subtasks_total": 2,
                   "action_jitter": 3.0}]
    r = compute_reward(traj, cfg)
    # step0: progress proxy -(1-0.5)*10*0.5 = -2.5
    # step1: +10 success, progress 1.0 -> 0, jitter ||0.5*1_7|| = sqrt(1.75)
    expected = -2.5 + 10.0 - 0.1 * np.sqrt(1.75)
    check("reward value", abs(r - expected) < 1e-6, f"{r} vs {expected}")
    R = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 10.0, 5.0, 5.0]])
    A = grpo_advantage(R, 4)
    check("group-relative advantage zero-mean", bool(np.allclose(A.mean(axis=1), 0, atol=1e-6)))


def test_splits():
    print("[4] split leakage guard")
    import tempfile
    from src.data.splits import assert_no_leak, make_sft_splits
    s = make_sft_splits([f"t{i}" for i in range(100)], 0.85, 42)
    check("85/15 sizes", len(s["sft_train"]) == 85 and len(s["sft_val"]) == 15)
    assert_no_leak({"sft_train": set(s["sft_train"]), "sft_val": set(s["sft_val"]),
                    "id_test": {"x1", "x2"}, "ood_test": {"y1", "y2"}})
    leaked = False
    try:
        assert_no_leak({"sft_train": {"a", "b"}, "sft_val": set(),
                        "id_test": {"a"}})
    except AssertionError:
        leaked = True
    check("leak detection works", leaked)


def test_csi_align():
    print("[5] CSI frame alignment")
    from src.data.csi import align_wav_to_windows
    wav = np.linspace(0, 1, 16000, dtype=np.float32)
    segs = align_wav_to_windows(wav, 100, 8)
    check("13 windows for 100 frames", len(segs) == 13, str(len(segs)))


def main():
    test_configs()
    test_action_head()
    test_rewards()
    test_splits()
    test_csi_align()
    print(f"\nAll {PASS} smoke checks passed (CPU-only).")


if __name__ == "__main__":
    main()
