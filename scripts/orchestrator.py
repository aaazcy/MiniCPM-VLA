#!/usr/bin/env python3
"""Full-pipeline orchestrator for the 3-arm experiment (A / B2 / B3).

Sequence (strict, restart-safe via runs/orchestrator_state.json):
  1. Wait for A1 SFT (already running, 3-card DDP, 4 epochs) to finish.
  2. A1 best ID eval  -> 3-card shard, merge, report.
  3. B2 SFT (fresh, 3-card DDP, 4 epochs).
  4. B2 best ID eval  -> 3-card shard, merge, report.
  5. B2 GRPO (3-card rollout-parallel).

Only the LOWEST-VAL-LOSS checkpoint (sft/best) is evaluated — no per-epoch
evaluation. Each eval is sharded across all 3 GPUs so none sit idle.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "orchestrator_state.json"
ENV = (
    "export PATH=/root/miniconda3/bin:$PATH && "
    "export HF_ENDPOINT=https://hf-mirror.com && "
    "export PYOPENGL_PLATFORM=egl && "
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    "export TOKENIZERS_PARALLELISM=false && "
)
PY = "/root/miniconda3/bin/python"


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def pgrep_running(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
    return r.returncode == 0


def log_done(procs: dict) -> None:
    for tag, p in list(procs.items()):
        if p.poll() is not None:
            print(f"[orch] done: {tag} (rc={p.returncode})", flush=True)
            del procs[tag]


def launch(cmd: str, logfile: Path):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as f:
        return subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)


def sft_cmd(group: str, epochs: int) -> str:
    return (f"cd {ROOT} && {ENV} "
            f"torchrun --nproc_per_node=3 scripts/train_sft.py "
            f"--group {group} --epochs {epochs}")


def eval_cmd(group: str, ckpt: str, gpu: int, k: int, n: int) -> str:
    return (f"cd {ROOT} && {ENV} "
            f"{PY} -u scripts/evaluate.py --group {group} --split id "
            f"--ckpt {ckpt} --device cuda:{gpu} --shard {k} {n}")


def grpo_cmd(group: str) -> str:
    return (f"cd {ROOT} && {ENV} "
            f"torchrun --nproc_per_node=3 scripts/train_grpo_dist.py --group {group}")


def merge_shards(run_dir: Path, split: str = "id", modality: str = "native",
                 n_shards: int = 3):
    acc, wsum, total = {}, {}, 0
    per_instr = []
    for k in range(n_shards):
        f = run_dir / f"eval_{split}_{modality}_shard{k}of{n_shards}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        n = d.get("n_instructions", 0)
        total += n
        for key, val in d.get("metrics", {}).items():
            if not isinstance(val, (int, float)):
                continue
            acc[key] = acc.get(key, 0.0) + val * n
            wsum[key] = wsum.get(key, 0.0) + n
        # collect per-instruction detail from the shard's *_per_instr.json
        pf = run_dir / f"eval_{split}_{modality}_shard{k}of{n_shards}_per_instr.json"
        if pf.exists():
            per_instr.extend(json.loads(pf.read_text()).get("per_instruction", []))
    merged = {k: acc[k] / wsum[k] for k in acc}
    merged["n_instructions"] = total
    out = run_dir / f"eval_{split}_{modality}_merged.json"
    out.write_text(json.dumps({"merged": merged}, indent=2))
    # append the merged result to the central ledger (append-only)
    group_id = run_dir.name
    ledger = ROOT / "runs" / "results.jsonl"
    entry = {"ts": datetime.now().isoformat(timespec="seconds"),
             "group": group_id, "split": split, "modality": modality,
             "shard": None, "merged": True, "ckpt": None,
             "metrics": merged, "per_instruction": per_instr}
    with open(ledger, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return merged


def wait_sft_done(group_id: str, logfile: Path) -> None:
    """Block until the SFT process for group_id is gone AND its log says done."""
    while True:
        if (not pgrep_running(f"train_sft.py --group {group_id}")
                and logfile.exists()
                and "SFT done" in logfile.read_text()):
            return
        time.sleep(30)


def launch_eval_shards(group: str, ckpt: str, prefix: str) -> dict:
    """Launch a 3-card sharded eval; returns tag -> Popen."""
    procs = {}
    for gpu in range(3):
        log = ROOT / "runs" / f"eval_{prefix}_shard{gpu}.log"
        procs[f"{prefix}_shard{gpu}"] = launch(
            eval_cmd(group, ckpt, gpu, gpu, 3), log)
    return procs


def procs_all_failed(procs: dict) -> bool:
    """True if every launched proc has exited with a non-zero return code."""
    if not procs:
        return False
    return all(p.poll() is not None and p.returncode != 0 for p in procs.values())


def main():
    state = load_state()
    state.setdefault("stage", "a1_sft")
    state.setdefault("procs", {})

    while True:
        stage = state["stage"]

        # ---------------- Stage 1: wait A1 SFT ----------------
        if stage == "a1_sft":
            logf = ROOT / "runs" / "train_a1_sft_ddp_lmhead.log"
            if (not pgrep_running("train_sft.py --group a1")
                    and logf.exists() and "SFT done" in logf.read_text()):
                print("[orch] A1 SFT finished -> evaluate best (3-card shard)", flush=True)
                main.procs = launch_eval_shards("a1", f"{ROOT}/runs/A1/sft/best", "A1_id")
                state["stage"] = "a1_eval"
                state["eval_retries"] = 0
                save_state(state)
                continue
            time.sleep(30)
            continue

        # ---------------- Stage 2: A1 eval -> merge -> B2 SFT ----------------
        if stage == "a1_eval":
            procs = getattr(main, "procs", {})
            alive = [t for t, p in procs.items() if p.poll() is None]
            if alive:
                time.sleep(30)
                continue
            merged = merge_shards(ROOT / "runs" / "A1")
            if merged is None:
                retries = state.get("eval_retries", 0)
                if procs_all_failed(procs) and retries < 3:
                    print(f"[orch] A1 eval failed -> relaunching (retry {retries + 1}/3)",
                          flush=True)
                    main.procs = launch_eval_shards("a1", f"{ROOT}/runs/A1/sft/best", "A1_id")
                    state["eval_retries"] = retries + 1
                    save_state(state)
                    continue
                print("[orch] A1 shards missing; waiting", flush=True)
                time.sleep(30)
                continue
            print(f"[orch] A1 best ID eval: {json.dumps(merged)}", flush=True)
            print("[orch] -> launching B2 SFT (fresh, 3-card DDP, 4 epochs)", flush=True)
            log = ROOT / "runs" / "train_b2_sft_ddp4.log"
            main.b2_proc = launch(sft_cmd("b2", 4), log)
            state["stage"] = "b2_sft"
            state.pop("eval_retries", None)
            save_state(state)
            continue

        # ---------------- Stage 3: wait B2 SFT ----------------
        if stage == "b2_sft":
            p = getattr(main, "b2_proc", None)
            if p is not None and p.poll() is None:
                time.sleep(30)
                continue
            logf = ROOT / "runs" / "train_b2_sft_ddp4.log"
            if logf.exists() and "SFT done" in logf.read_text():
                print("[orch] B2 SFT finished -> evaluate best (3-card shard)", flush=True)
                main.procs = launch_eval_shards("b2", f"{ROOT}/runs/B2/sft/best", "B2_id")
                state["stage"] = "b2_eval"
                state["eval_retries"] = 0
                save_state(state)
                continue
            time.sleep(30)
            continue

        # ---------------- Stage 4: B2 eval -> merge -> GRPO ----------------
        if stage == "b2_eval":
            procs = getattr(main, "procs", {})
            alive = [t for t, p in procs.items() if p.poll() is None]
            if alive:
                time.sleep(30)
                continue
            merged = merge_shards(ROOT / "runs" / "B2")
            if merged is None:
                retries = state.get("eval_retries", 0)
                if procs_all_failed(procs) and retries < 3:
                    print(f"[orch] B2 eval failed -> relaunching (retry {retries + 1}/3)",
                          flush=True)
                    main.procs = launch_eval_shards("b2", f"{ROOT}/runs/B2/sft/best", "B2_id")
                    state["eval_retries"] = retries + 1
                    save_state(state)
                    continue
                print("[orch] B2 shards missing; waiting", flush=True)
                time.sleep(30)
                continue
            print(f"[orch] B2 best ID eval: {json.dumps(merged)}", flush=True)
            print("[orch] -> launching B2 GRPO (3-card rollout-parallel)", flush=True)
            log = ROOT / "runs" / "train_b2_grpo_dist.log"
            main.b2_grpo_proc = launch(grpo_cmd("b2"), log)
            state["stage"] = "b2_grpo"
            state.pop("eval_retries", None)
            save_state(state)
            continue

        # ---------------- Stage 5: wait B2 GRPO ----------------
        if stage == "b2_grpo":
            p = getattr(main, "b2_grpo_proc", None)
            if p is not None and p.poll() is None:
                time.sleep(60)
                continue
            logf = ROOT / "runs" / "train_b2_grpo_dist.log"
            if logf.exists() and "GRPO done" in logf.read_text():
                print("[orch] B2 GRPO finished.", flush=True)
                state["stage"] = "done"
                save_state(state)
                continue
            time.sleep(60)
            continue

        if stage == "done":
            print("[orch] pipeline complete.", flush=True)
            time.sleep(3600)

        time.sleep(30)


main.procs = {}
main.b2_proc = None
main.b2_grpo_proc = None

if __name__ == "__main__":
    main()
