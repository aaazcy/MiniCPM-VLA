#!/usr/bin/env python3
"""Diagnose CALVIN env rendering: is the getCameraImage patch applied? what does
the env's rgb_obs actually contain?"""
import sys
sys.path.insert(0, "/root/autodl-tmp/vla-omni-bench")
import numpy as np
import pybullet as p

calls = []


def main():
    import pybullet as p2
    orig = p2.getCameraImage
    from src.envs.calvin_env import CalvinEnv

    env = CalvinEnv(max_episode_steps=10, cameras=["rgb_static"], image_size=224,
                    calvin_root="/root/autodl-tmp/vla-omni-bench/data/calvin",
                    scene="calvin_scene_A")
    print("env created; patch applied?", p2.getCameraImage is not orig)
    cam = env.env.cameras[0]
    print("cam type:", type(cam).__name__, "width/height:", cam.width, cam.height)
    for i in range(3):
        rgb, depth = cam.render()
        arr = np.asarray(rgb)
        print(f"cam.render() #{i}:", arr.shape, arr.dtype, "| mean:", float(arr.mean()) if arr.size else "empty")
    obs = env.reset("open the drawer")
    img = obs["pixel_values"]
    print("obs image:", img.shape, img.dtype, "| mean:", img.mean())
    # raw check
    w, h, rgba, dep, mask = p.getCameraImage(64, 64, renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                             physicsClientId=env.env.p._client)
    arr = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)
    print("raw camera after env:", arr.shape, "mean:", arr.mean())
    env.close()
    print("DIAG ENV OK")


if __name__ == "__main__":
    main()
