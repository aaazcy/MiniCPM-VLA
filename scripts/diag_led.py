#!/usr/bin/env python3
"""Focused: replay the id_test `turn_on_led` demos and track the LED logical_state
step-by-step, to tell apart (a) replay never toggles the LED (drift) vs (b) it toggles
but detection fails (bug)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.calvin_env import CalvinEnv  # noqa: E402

ROOT = PROJECT_ROOT / "data" / "calvin"


def main():
    with open(ROOT / "splits" / "calvin_id_test.json") as f:
        ids = json.load(f)["ids"]

    for tid in ids:
        p = ROOT / "id_test" / f"{tid}.npz"
        if not p.exists():
            p = ROOT / "sft_train" / f"{tid}.npz"
        d = np.load(p, allow_pickle=True)
        task = str(d["task"])
        if task not in ("turn_on_led", "turn_off_led"):
            continue
        lang = str(d["lang"])
        actions = d["actions"]
        env = CalvinEnv(max_episode_steps=len(actions) + 5, cameras=["rgb_static"],
                        image_size=224, calvin_root=ROOT, scene="calvin_scene_A")
        obs = env.reset(lang, robot_obs=d["init_robot_obs"].astype(np.float32),
                        scene_obs=d["init_scene_obs"].astype(np.float32), task=task)

        led_hist = []
        robot_hist = []
        success = False
        for i, a in enumerate(actions):
            obs, r, done, info = env.step(a)
            si = info["scene_info"]
            led_hist.append(si["lights"]["led"]["logical_state"])
            robot_hist.append(np.round(si["robot_info"]["tcp_pos"][:2], 3)
                              if "tcp_pos" in si.get("robot_info", {}) else None)
            if info.get("success"):
                success = True
                print(f"[{tid} {task}] lang={lang!r} n={len(actions)} "
                      f"SUCCESS@step{i} led 0->1 detected")
                break
        env.close(); env = None
        if not success:
            n_toggle = sum(1 for k in range(1, len(led_hist)) if led_hist[k] != led_hist[k - 1])
            print(f"[{tid} {task}] lang={lang!r} n={len(actions)} "
                  f"led transitions={n_toggle} start={led_hist[0]} end={led_hist[-1]} "
                  f"unique={sorted(set(led_hist))}")


if __name__ == "__main__":
    main()
