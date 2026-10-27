#!/usr/bin/env python3
"""Snapshot the running SFT's `last` checkpoint to `epoch_N` after each epoch.

The currently-running SFT processes predate the per-epoch save in sft.py, so this
watcher preserves each epoch's weights (read the epoch from trainer_state.pt, which
is written last in SFTTrainer.save, so it is safe to use as a completion signal).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["B2", "A1"]


def snapshot():
    for g in GROUPS:
        state_file = ROOT / "runs" / g / "sft" / "last" / "trainer_state.pt"
        if not state_file.exists():
            continue
        try:
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            epoch = int(state.get("epoch", -1))
        except Exception:
            continue
        src = ROOT / "runs" / g / "sft" / "last"
        dst = ROOT / "runs" / g / "sft" / f"epoch_{epoch}"
        if src.exists() and not dst.exists() and epoch >= 0:
            shutil.copytree(src, dst)
            print(f"[snapshot] {g} -> epoch_{epoch}", flush=True)


if __name__ == "__main__":
    while True:
        snapshot()
        time.sleep(20)
