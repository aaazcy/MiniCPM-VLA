#!/usr/bin/env python3
"""End-to-end check of CALVIN success detection: replay demo actions for a real
trajectory and confirm the env's `success`/`subtasks_completed` info is populated."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.envs.calvin_env import CalvinEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data" / "calvin"


def main():
    import glob
    # pick a simple, unambiguous task to replay
    found = None
    for task_name in ["open_drawer", "turn_on_lightbulb", "move_slider_left",
                      "lift_red_block_table", "push_red_block_left"]:
        for f in sorted(glob.glob(str(ROOT / "sft_train" / "*.npz"))):
            d = np.load(f, allow_pickle=True)
            if str(d["task"]) == task_name:
                found = (f, d)
                break
        if found:
            break
    assert found, "no replayable trajectory found"
    f, d = found
    task = str(d["task"])
    lang = str(d["lang"])
    actions = d["actions"]
    robot_obs = d["init_robot_obs"].astype(np.float32)
    scene_obs = d["init_scene_obs"].astype(np.float32)
    print(f"replaying: task={task} lang={lang!r} n_steps={len(actions)}")

    env = CalvinEnv(max_episode_steps=len(actions) + 5, cameras=["rgb_static"],
                    image_size=224, calvin_root=ROOT, scene="calvin_scene_A")
    obs = env.reset(lang, robot_obs=robot_obs, scene_obs=scene_obs, task=task)
    success_step = None
    for i, a in enumerate(actions):
        obs, r, done, info = env.step(a)
        if info.get("success"):
            success_step = i
            print(f"  SUCCESS at step {i}: "
                  f"subtasks_completed={info['subtasks_completed']} "
                  f"total={info['subtasks_total']}")
            break
        if done and not info.get("success"):
            print(f"  truncated at step {i} without success")
            break
    if success_step is None:
        # check whether the task condition was ever satisfied (oracle vs episode end)
        print("  result: NO SUCCESS detected")
    env.close()
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
