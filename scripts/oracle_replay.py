#!/usr/bin/env python3
"""Oracle-replay benchmark: replay demo actions on every id_test trajectory.

Establishes the CEILING success rate (what perfect behavior cloning could reach)
and verifies the CALVIN success detector across ALL task types — not just a single
task. Any task that errors or never succeeds flags a task-definition / object-name
mismatch to fix before interpreting model results.

Usage:
    python scripts/oracle_replay.py --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.calvin_env import CalvinEnv  # noqa: E402

ROOT = PROJECT_ROOT / "data" / "calvin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="id_test")
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()

    with open(ROOT / "splits" / f"calvin_{args.split}.json") as f:
        ids = json.load(f)["ids"]

    results = []
    for idx, tid in enumerate(ids):
        p = ROOT / args.split / f"{tid}.npz"
        if not p.exists():
            p = ROOT / "sft_train" / f"{tid}.npz"
        d = np.load(p, allow_pickle=True)
        task = str(d["task"]) if "task" in d else None
        lang = str(d["lang"])
        actions = d["actions"]
        robot_obs = d["init_robot_obs"].astype(np.float32)
        scene_obs = d["init_scene_obs"].astype(np.float32)

        success, step = False, None
        try:
            env = CalvinEnv(max_episode_steps=len(actions) + 5, cameras=["rgb_static"],
                            image_size=224, calvin_root=ROOT, scene="calvin_scene_A")
            obs = env.reset(lang, robot_obs=robot_obs, scene_obs=scene_obs, task=task)
            for i, a in enumerate(actions):
                obs, r, done, info = env.step(a)
                if info.get("success"):
                    success, step = True, i
                    break
                if done:
                    break
            env.close()
            env = None  # CRITICAL: free NOW (pybullet reuses DIRECT ids; a delayed
                        # __del__ would kill the NEXT env's connection)
        except Exception as e:
            env = None
            results.append((tid, task, lang, f"ERROR {type(e).__name__}: {e}"))
            print(f"  [{idx + 1}/{len(ids)}] {task}: ERROR {type(e).__name__}: {e}")
            continue

        results.append((tid, task, lang, "SUCCESS" if success else "fail"))
        print(f"  [{idx + 1}/{len(ids)}] {task:28s} {'SUCCESS' if success else 'fail'}"
              f"{'' if step is None else f' @step{step}'}")

    n_ok = sum(1 for r in results if r[3] == "SUCCESS")
    n_err = sum(1 for r in results if r[3].startswith("ERROR"))
    n_fail = sum(1 for r in results if r[3] == "fail")
    print(f"\n=== Oracle replay ({args.split}) ===")
    print(f"success: {n_ok}/{len(results)} = {n_ok / len(results) * 100:.1f}%")
    print(f"fail (never detected): {n_fail} | errors: {n_err}")
    # group failures by task
    fails = {}
    for tid, task, lang, st in results:
        if st != "SUCCESS":
            fails.setdefault(task, 0)
            fails[task] += 1
    if fails:
        print("non-success by task:")
        for task, c in sorted(fails.items(), key=lambda x: -x[1]):
            print(f"  {task}: {c}")


if __name__ == "__main__":
    main()
