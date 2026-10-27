#!/usr/bin/env python3
"""Diagnose the low oracle-replay success rate: is it replay drift, a strict
threshold, or a detection bug? Dumps the raw scene_info evolution for a few
representative failing tasks."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.calvin_env import CalvinEnv  # noqa: E402

ROOT = PROJECT_ROOT / "data" / "calvin"


def find_traj(task_name):
    for f in sorted(glob.glob(str(ROOT / "sft_train" / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d.get("task")) == task_name:
            return f, d
    for f in sorted(glob.glob(str(ROOT / "id_test" / "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if str(d.get("task")) == task_name:
            return f, d
    return None, None


def replay(task_name, probe):
    f, d = find_traj(task_name)
    if f is None:
        print(f"[{task_name}] no trajectory found")
        return
    lang = str(d["lang"])
    actions = d["actions"]
    env = CalvinEnv(max_episode_steps=len(actions) + 5, cameras=["rgb_static"],
                    image_size=224, calvin_root=ROOT, scene="calvin_scene_A")
    obs = env.reset(lang, robot_obs=d["init_robot_obs"].astype(np.float32),
                    scene_obs=d["init_scene_obs"].astype(np.float32), task=task_name)
    start = probe(env, obs, None)
    print(f"[{task_name}] lang={lang!r} n={len(actions)}")
    print(f"  start: {start}")
    end = None
    for i, a in enumerate(actions):
        obs, r, done, info = env.step(a)
        end = probe(env, obs, info)
        if info.get("success"):
            print(f"  SUCCESS @step{i}")
            break
    print(f"  end:   {end}")
    env.close()


def led_probe(env, obs, info):
    # read LED logical state from the raw env info
    si = info["scene_info"] if (info is not None and "scene_info" in info) else env.env.get_info()["scene_info"]
    try:
        return {"led": si["lights"]["led"]["logical_state"],
                "lightbulb": si["lights"]["lightbulb"]["logical_state"]}
    except Exception as e:
        return {"err": str(e)}


def red_block_z(env, obs, info):
    si = info["scene_info"] if (info is not None and "scene_info" in info) else env.env.get_info()["scene_info"]
    try:
        return {"block_red_z": round(float(si["movable_objects"]["block_red"]["current_pos"][2]), 4),
                "block_red_pos": np.round(si["movable_objects"]["block_red"]["current_pos"], 3)}
    except Exception as e:
        return {"err": str(e)}


def block_pos(env, obs, info):
    si = info["scene_info"] if (info is not None and "scene_info" in info) else env.env.get_info()["scene_info"]
    try:
        return {k: np.round(v["current_pos"], 3)
                for k, v in si["movable_objects"].items()}
    except Exception as e:
        return {"err": str(e)}


if __name__ == "__main__":
    print("=== turn_on_led (3/3 fail) ===")
    replay("turn_on_led", led_probe)
    print("\n=== lift_red_block_table (fail; threshold z>0.05) ===")
    replay("lift_red_block_table", red_block_z)
    print("\n=== rotate_red_block_right (fail; threshold z>60deg) ===")
    replay("rotate_red_block_right", block_pos)
    print("\n=== push_red_block_left (fail; threshold x<-0.1) ===")
    replay("push_red_block_left", block_pos)
