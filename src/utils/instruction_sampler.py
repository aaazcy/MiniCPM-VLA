"""src/utils/instruction_sampler.py — language instruction sampling for RL/eval.

PROTOCOL (leak-free):
  * GRPO rollout pool  = calvin_sft_train (grpo_eval excluded)
  * GRPO periodic deterministic eval pool = calvin_grpo_eval (held-out from training)
  * calvin_id_test is NEVER touched during training — final evaluation only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np


class InstructionSampler:
    def __init__(self, train_instructions: List[str], eval_instructions: List[str],
                 seed: int = 42):
        self.train = list(train_instructions)
        self.eval = list(eval_instructions)
        self.rng = random.Random(seed)

    def sample(self) -> str:
        return self.rng.choice(self.train)

    def sample_eval(self, n: int) -> List[str]:
        rng = random.Random(0)  # fixed order for eval determinism
        pool = list(self.eval)
        rng.shuffle(pool)
        return pool[:n]


def _load_instr(root: Path, manifest_name: str, fallback_dir: str) -> List[str]:
    instrs = []
    with open(root / "splits" / f"{manifest_name}.json") as f:
        ids = json.load(f)["ids"]
    for tid in ids:
        p = root / fallback_dir / f"{tid}.npz"
        if not p.exists():
            p = root / "sft_train" / f"{tid}.npz"
        with np.load(p) as f_:
            instrs.append(str(f_["lang"]))
    return instrs


def load_instruction_pools(cfg) -> InstructionSampler:
    """GRPO rollouts sample sft_train; deterministic eval uses grpo_eval."""
    root = Path(cfg.dataset.calvin_root)
    train_instr = _load_instr(root, "calvin_sft_train", "sft_train")
    eval_instr = _load_instr(root, "calvin_grpo_eval", "sft_train")
    return InstructionSampler(train_instr, eval_instr, seed=cfg.experiment.seed)
