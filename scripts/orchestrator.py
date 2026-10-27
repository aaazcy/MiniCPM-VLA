#!/usr/bin/env python3
"""Autonomous training orchestrator.

Watches for new per-epoch SFT checkpoints (runs/<G>/sft/epoch_N, created by the
snapshot watcher) and immediately launches a full ID evaluation for each on an idle
GPU. Keeps GPUs busy, never batches evaluations, and persists its state so it can be
restarted safely. Also reports when SFT finishes (so GRPO can be started).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "orchestrator_state.json"
GROUPS = ["B2", "A1"]
MODEL_NEED_GB = 26  # a free GPU must have at least this much free (GiB)


def gpu_free_gib(gpu_id: int) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode().strip().split("\n")
        return int(out[gpu_id]) / 1024.0
    except Exception:
        return 0.0


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def main():
    state = load_state()
    state.setdefault("launched", [])   # list of keys already launched
    active = {}                        # gpu_id -> (key, Popen)

    while True:
        # reap finished evals
        for gpu, (key, p) in list(active.items()):
            if p.poll() is not None:
                print(f"[orch] done: {key} (rc={p.returncode})", flush=True)
                del active[gpu]

        # find new epoch checkpoints
        for g in GROUPS:
            sft_dir = ROOT / "runs" / g / "sft"
            if not sft_dir.exists():
                continue
            for p in sorted(sft_dir.glob("epoch_*")):
                if not (p / "trainer_state.pt").exists():
                    continue
                epoch = int(p.name.split("_")[1])
                key = f"{g}_id_epoch{epoch}"
                if key in state["launched"] or any(k == key for k, _ in active.values()):
                    continue
                # find an idle GPU (not running an eval) with enough free memory
                gpu = None
                for i in range(3):
                    if i in active:
                        continue
                    if gpu_free_gib(i) >= MODEL_NEED_GB:
                        gpu = i
                        break
                if gpu is None:
                    print(f"[orch] no free GPU for {key}; will retry", flush=True)
                    break
                log = ROOT / "runs" / f"eval_{g}_id_epoch{epoch}.log"
                cmd = (
                    f"cd {ROOT} && export PATH=/root/miniconda3/bin:$PATH "
                    f"export PYOPENGL_PLATFORM=egl "
                    f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                    f"export TOKENIZERS_PARALLELISM=false && "
                    f"/root/miniconda3/bin/python -u scripts/evaluate.py "
                    f"--group {g.lower()} --split id --ckpt {p} --device cuda:{gpu} "
                    f"> {log} 2>&1"
                )
                proc = subprocess.Popen(cmd, shell=True)
                active[gpu] = (key, proc)
                state["launched"].append(key)
                save_state(state)
                print(f"[orch] launched {key} on cuda:{gpu}", flush=True)

        time.sleep(30)


if __name__ == "__main__":
    main()
