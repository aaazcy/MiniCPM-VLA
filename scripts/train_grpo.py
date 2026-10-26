#!/usr/bin/env python3
"""scripts/train_grpo.py — Stage 2 entry point (all groups).

Usage:
    python scripts/train_grpo.py --group a1
    python scripts/train_grpo.py --group b2 --resume latest
    python scripts/train_grpo.py --group b2 --device cuda:1

Requires:
  * the SFT checkpoint at runs/<group>/sft/best (policy init + frozen KL reference)
  * preprocessed data manifests (Phase 0)
Rollout engine: vLLM for OpenVLA groups, HF engine for MiniCPM-o groups and B3
(see src/rollout/engine.py module docstring for the reasoning).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.rollout.engine import make_engine  # noqa: E402
from src.training.grpo import GRPOTrainer  # noqa: E402
from src.utils.ckpt import resolve_resume  # noqa: E402
from src.utils.instruction_sampler import load_instruction_pools  # noqa: E402
from src.utils.logger import Logger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["a1", "b1", "b2", "b3"])
    ap.add_argument("--resume", default=None, help="checkpoint dir | 'latest'")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.group)
    assert not cfg.group.sft_only, f"{cfg.group.id} is SFT-only (no GRPO stage)"
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)

    sft_ckpt = Path(cfg.sft_ckpt_dir) / "best"
    if not (sft_ckpt / "trainer_state.pt").exists():
        raise FileNotFoundError(f"SFT checkpoint missing at {sft_ckpt}; "
                                f"run train_sft.py --group {cfg.group.id} first")

    logger = Logger(cfg, cfg.group.id, tags=["grpo", cfg.group.family])

    # policy = SFT checkpoint + trainable LoRA/head
    policy = build_model(cfg, stage="grpo")
    policy.load_adapter(str(sft_ckpt))
    policy.to(args.device)

    # frozen SFT reference for the GRPO KL penalty (never updated)
    reference = build_model(cfg, stage="ref")
    reference.load_adapter(str(sft_ckpt))
    reference.to(args.device)
    reference.eval()

    sampler = load_instruction_pools(cfg)
    engine = make_engine(cfg, policy=policy, processor=policy.processor,
                         lora_path=str(sft_ckpt), device=args.device)

    trainer = GRPOTrainer(policy, reference, cfg, engine, sampler, device=args.device)
    resume = resolve_resume(args.resume, cfg.grpo_ckpt_dir, "grpo")
    final = trainer.train(logger, resume_from=resume)

    logger.log({"grpo/final_ckpt": str(final)})
    logger.finish()
    print(f"GRPO done. Final checkpoint: {final}")


if __name__ == "__main__":
    main()
