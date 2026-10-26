#!/usr/bin/env python3
"""scripts/smoke_grpo.py — GRPO pipeline smoke test: CALVIN env + rollout + 1 update.

Runs on a spare GPU (e.g. GPU1) while another group trains on GPU0.
    python scripts/smoke_grpo.py --group a1 --device cuda

Validates: env instantiation (EGL rendering), policy rollout through HFEngine,
reward computation, and one GRPO update (group advantage + KL vs frozen reference).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

import numpy as np
import torch

from src.config import load_config
from src.models.factory import build_model
from src.rollout.engine import make_engine
from src.training.grpo import GRPOTrainer
from src.training.rewards import compute_reward
from src.utils.instruction_sampler import load_instruction_pools
from src.utils.ckpt import find_latest_ckpt


class _Sampler:
    def __init__(self, instrs):
        self.instrs = instrs

    def sample(self):
        return self.instrs[0]

    def sample_eval(self, n):
        return self.instrs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="a1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = load_config(args.group)
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    # SFT checkpoint: explicit, latest, or the smoke checkpoint
    ckpt = Path(args.ckpt) if args.ckpt else None
    if ckpt is None:
        ckpt = find_latest_ckpt(cfg.sft_ckpt_dir) or (cfg.sft_ckpt_dir / "smoke")
    print(f"using SFT ckpt: {ckpt}")

    policy = build_model(cfg, stage="sft")
    if (ckpt / "action_head.pt").exists() or any((ckpt).glob("adapter*")):
        policy.load_adapter(str(ckpt))
    policy.to(args.device)
    policy.eval()

    reference = build_model(cfg, stage="ref")
    if (ckpt / "action_head.pt").exists() or any((ckpt).glob("adapter*")):
        reference.load_adapter(str(ckpt))
    reference.to(args.device)
    reference.eval()

    engine = make_engine(cfg, policy=policy, processor=policy.processor,
                         lora_path=str(ckpt), device=args.device)

    # 1. env + rollout (one trajectory)
    sampler = _Sampler(["open the drawer"])  # dummy instruction; env ignores
    trainer = GRPOTrainer(policy, reference, cfg, engine, sampler, device=args.device)

    instr = "open the drawer"
    traj = trainer.rollout_one(instr, temperature=1.0)
    r = compute_reward(traj, cfg)
    print(f"[rollout] len={traj.length} success={traj.success} reward={r:.3f} "
          f"subtasks={traj.infos[-1].get('subtasks_completed') if traj.infos else '?'}")
    print(f"[obs] image shape={traj.obs[0]['pixel_values'].shape} "
          f"action shape={traj.actions[0].shape}")

    # 2. one GRPO update (single group of 4 trajectories)
    groups = [[trainer.rollout_one(instr, temperature=1.0) for _ in range(4)]]
    stats = trainer.update(groups, LoggerStub())
    print(f"[update] {stats}")

    print("GRPO SMOKE OK")


class LoggerStub:
    def log(self, metrics, step=None, commit=False):
        print(f"  [wandb-stub] {metrics}")


if __name__ == "__main__":
    main()
