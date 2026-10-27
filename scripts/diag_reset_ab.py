#!/usr/bin/env python3
"""A/B test: does reset_from_storage (serialized full state) replay better than
reset(robot_obs, scene_obs)? Replays id_test LED demos under both and compares
whether the LED logical_state ever toggles."""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calvin_env.envs.play_table_env import get_env  # noqa: E402

ROOT = PROJECT_ROOT / "data" / "calvin"


def replay_led(env, actions):
    """Replay actions on the raw env, return (toggled, led_start, led_end, n_trans)."""
    led_states = []
    for a in actions:
        obs, _, _, info = env.step(a)
        led_states.append(info["scene_info"]["lights"]["led"]["logical_state"])
    n_trans = sum(1 for k in range(1, len(led_states)) if led_states[k] != led_states[k - 1])
    return n_trans > 0, led_states[0], led_states[-1], n_trans


def main():
    with open(ROOT / "splits" / "calvin_id_test.json") as f:
        ids = json.load(f)["ids"]

    tasks = []
    for tid in ids:
        p = ROOT / "id_test" / f"{tid}.npz"
        if not p.exists():
            p = ROOT / "sft_train" / f"{tid}.npz"
        d = np.load(p, allow_pickle=True)
        if str(d["task"]) in ("turn_on_led", "turn_off_led"):
            tasks.append(d)

    for d in tasks:
        task = str(d["task"])
        actions = d["actions"]
        robot = d["init_robot_obs"].astype(np.float32)
        scene = d["init_scene_obs"].astype(np.float32)

        env = get_env(ROOT, show_gui=False,
                      obs_space={"rgb_obs": ["rgb_static"], "depth_obs": [],
                                 "state_obs": ["robot_obs"], "language": True},
                      scene="calvin_scene_A")

        # Method A: reset(robot_obs, scene_obs)
        env.reset(robot_obs=robot, scene_obs=scene)
        a = replay_led(env, actions)

        # Method B: reset + serialize + reset_from_storage
        env.reset(robot_obs=robot, scene_obs=scene)
        state = env.serialize()
        # reset_from_storage expects a full data dict (robot+scene+state_obs+done+info)
        state["state_obs"] = env.get_obs()
        state["done"] = False
        state["info"] = env.get_info()
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(tmp.name, "wb") as f:
            pickle.dump(state, f)
        env.reset_from_storage(tmp.name)
        b = replay_led(env, actions)

        print(f"[{task}] n={len(actions)}")
        print(f"  A reset(obs)         : toggled={a[0]} led {a[1]}->{a[2]} ({a[3]} trans)")
        print(f"  B reset_from_storage : toggled={b[0]} led {b[1]}->{b[2]} ({b[3]} trans)")
        env.close()


if __name__ == "__main__":
    main()
