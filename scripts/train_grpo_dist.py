#!/usr/bin/env python3
"""scripts/train_grpo_dist.py — 3-GPU synchronous rollout-parallel GRPO.

    torchrun --nproc_per_node=3 scripts/train_grpo_dist.py --group b2

Pattern (actor-learner, minimal synchronous form):
  * rank 0 samples ONE instruction per group (G=8 rollouts of the same task)
  * ALL ranks hold a policy replica and sample rollouts in parallel (~6 each)
  * trajectories gathered to rank 0 (in slot order -> regrouped)
  * rank 0 computes group-relative advantages + the clipped-ratio update
    (LoRA + action head only) and broadcasts the TRAINABLE weights back
  * repeat until total_env_steps

NOT FSDP/DDP: the parallelized stage is ENV SAMPLING (the GRPO bottleneck),
not gradient computation. The update stays on one GPU (dual-phase offload;
the frozen reference for the KL penalty only exists on rank 0).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

import numpy as np
import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.rollout.engine import make_engine  # noqa: E402
from src.training.grpo import GRPOTrainer  # noqa: E402
from src.utils.ckpt import resolve_resume  # noqa: E402
from src.utils.instruction_sampler import load_instruction_pools  # noqa: E402
from src.utils.logger import Logger  # noqa: E402

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["a1", "b1", "b2", "b3"])
    ap.add_argument("--resume", default=None, help="checkpoint dir | 'latest'")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

    cfg = load_config(args.group)
    # per-rank seeds -> diverse instructions/trajectories across workers
    random.seed(cfg.experiment.seed + rank * 1009)
    np.random.seed(cfg.experiment.seed + rank * 1009)
    torch.manual_seed(cfg.experiment.seed + rank * 1009)

    sft_ckpt = Path(cfg.sft_ckpt_dir) / "best"
    if not (sft_ckpt / "trainer_state.pt").exists():
        raise FileNotFoundError(f"SFT checkpoint missing at {sft_ckpt}")

    from src.utils.ckpt import head_compatible
    if not head_compatible(sft_ckpt, cfg):
        raise SystemExit(
            f"SFT checkpoint {sft_ckpt} uses a different action-head config than "
            f"the current one ({cfg.model.action_head}). Retrain SFT or fix the "
            f"config.")

    logger = Logger(cfg, cfg.group.id, tags=["grpo-dist", cfg.group.family]) \
        if rank == 0 else None

    policy = build_model(cfg, stage="grpo")
    policy.load_adapter(str(sft_ckpt))
    policy.to(f"cuda:{rank}")
    policy.eval()

    sampler = load_instruction_pools(cfg)
    engine = make_engine(cfg, policy=policy, processor=policy.processor,
                         lora_path=str(sft_ckpt), device=f"cuda:{rank}")

    # reference only on rank 0 (KL anchor, stays on CPU between phases)
    reference = build_model(cfg, stage="ref") if rank == 0 else None
    if reference is not None:
        reference.load_adapter(str(sft_ckpt))
        reference.eval()
    trainer = GRPOTrainer(policy, reference, cfg, engine, sampler, device=f"cuda:{rank}")
    if rank == 0:
        resume = resolve_resume(args.resume, cfg.grpo_ckpt_dir, "grpo")
        if resume is not None:
            trainer._load_state(resume)

    g = cfg.grpo
    n_total = g.rollout_per_group * g.group_size
    per_rank = (n_total + world - 1) // world
    ckpt_dir = cfg.grpo_ckpt_dir
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    while True:
        # ---- rollout phase: rank 0 samples ONE instruction per group, broadcast ----
        # Textbook GRPO: a group = group_size rollouts of the SAME instruction, so
        # the group-relative baseline compares responses to one task. Slots
        # [0, n_total) map to group `slot // group_size`; each rank rolls out its
        # share and trajectories come back in slot order.
        if rank == 0:
            group_instrs = [sampler.sample() for _ in range(g.rollout_per_group)]
        else:
            group_instrs = [None] * g.rollout_per_group
        if world > 1:
            dist.broadcast_object_list(group_instrs, src=0)

        local = []
        for j in range(per_rank):
            slot = rank * per_rank + j
            if slot >= n_total:
                break
            instr, task = group_instrs[slot // g.group_size]
            traj = trainer.rollout_one(instr, temperature=cfg.rollout.temperature,
                                       task=task)
            local.append(traj)
        gathered = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(local, gathered, dst=0)
        else:
            gathered = [local]

        weights = [None]
        progress = [0]
        if rank == 0:
            trajs = [t for lst in gathered for t in lst][:n_total]
            groups = [trajs[i * g.group_size:(i + 1) * g.group_size]
                      for i in range(g.rollout_per_group)]
            steps_before = trainer.env_steps
            for t in trajs:
                trainer.env_steps += t.length
            stats = trainer.update(groups, logger)
            stats["env_steps"] = trainer.env_steps
            trainer.metrics_history.append(stats)
            if trainer.env_steps // g.eval_every_episodes > \
                    steps_before // g.eval_every_episodes:
                eval_metrics = trainer.deterministic_eval(logger)
                logger.log(eval_metrics, step=trainer.env_steps)
                trainer.save(ckpt_dir / f"step_{trainer.env_steps}")
            trainer.save(ckpt_dir / "last")
            # only TRAINABLE weights changed (LoRA + action head)
            weights = [{n: p.detach().cpu()
                        for n, p in policy.named_parameters() if p.requires_grad}]
            progress = [trainer.env_steps]

        if world > 1:
            dist.broadcast_object_list(weights, src=0)
            dist.broadcast_object_list(progress, src=0)

        if rank != 0 and weights[0] is not None:
            cur = dict(policy.named_parameters())
            for n, w in weights[0].items():
                cur[n].data.copy_(w.to(cur[n].device))

        if progress[0] >= g.total_env_steps:
            break

    if rank == 0:
        log.info("[GRPO-dist] finished: %d env steps, %d updates",
                 trainer.env_steps, trainer.update_count)
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(trainer.metrics_history, f, indent=2)
        logger.log({"grpo/final_ckpt": str(ckpt_dir / "last")})
        logger.finish()
        print(f"GRPO done. Final checkpoint: {ckpt_dir / 'last'}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
