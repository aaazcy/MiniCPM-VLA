"""src/envs/calvin_env.py — CALVIN gym wrapper for RL rollouts and evaluation.

Franka robot, compositional long-horizon manipulation (push block / open drawer /
pick-place), max episode length = 100 steps, single static camera 224x224.
Action: 7-D continuous [dx, dy, dz, rx, ry, rz, gripper], normalized [-1, 1];
the wrapper denormalizes into the CALVIN relative-action range before stepping.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

ACTION_DIM = 7
# CALVIN relative actions are consumed by the env in dataset units (already normalized
# to [-1,1]: pos factor 50, rot factor 20, gripper ±1) — see the official dataset README.


class CalvinEnv:
    """Thin, dependency-safe wrapper. Import of calvin_env is lazy so the package
    imports fine on CPU-only machines before the GPU phase."""

    def __init__(self, task_cfg=None, max_episode_steps: int = 100,
                 cameras: List[str] = None, image_size: int = 224, seed: Optional[int] = None,
                 calvin_root: Optional[Path] = None, scene: str = "calvin_scene_A"):
        from calvin_env.envs.play_table_env import get_env  # lazy import (needs pybullet)
        import pybullet as p

        # vGPU: eglRenderer PLUGIN fails to load ("failed to EGL with glad"), but
        # ER_BULLET_HARDWARE_OPENGL still renders (verified: 64x64 image returned).
        # Force it for ALL camera calls inside calvin_env.
        _orig_get_camera_image = p.getCameraImage

        def _patched_get_camera_image(*args, **kwargs):
            kwargs["renderer"] = p.ER_BULLET_HARDWARE_OPENGL
            kwargs.setdefault("flags", p.ER_NO_SEGMENTATION_MASK)
            return _orig_get_camera_image(*args, **kwargs)

        p.getCameraImage = _patched_get_camera_image

        cameras = cameras or ["rgb_static"]
        # dataset dir is required by calvin_env for state/animation loading
        dataset_dir = Path(calvin_root) if calvin_root is not None else Path("calvin")
        env = get_env(
            dataset_dir,
            show_gui=False,   # NOTE: parameter is show_gui (was typo'd show_ui -> GUI mode -> X server)
            obs_space={"rgb_obs": cameras, "depth_obs": [], "state_obs": ["robot_obs"],
                       "language": True},
            scene=scene,   # calvin_scene_A (train/ID) or calvin_scene_D (OOD)
        )
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.cameras = cameras
        self.image_size = image_size
        self.seed = seed
        self._steps = 0
        self._instruction: Optional[str] = None
        self._lang_annotation: Optional[Any] = None
        self._last_action: Optional[np.ndarray] = None
        self._collision = False

    # ------------------------------------------------------------------ #
    def reset(self, instruction: str, robot_obs: Optional[np.ndarray] = None,
              scene_obs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Reset with a (possibly paraphrased/long) language instruction.

        robot_obs/scene_obs: optional recorded initial state (official CALVIN eval
        protocol resets each episode from its recorded start state).
        """
        # The calvin_env robot/scene reset code indexes these with lists of ints and
        # np.split — which only works for flat float32 ndarrays. Robustness manifests
        # store JSON lists; some npz variants store 2D/object arrays. Normalize.
        if robot_obs is not None:
            robot_obs = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
        if scene_obs is not None:
            scene_obs = np.asarray(scene_obs, dtype=np.float32).reshape(-1)
        obs = self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self._instruction = instruction
        self._lang_annotation = self.env.get_lang_ann() if hasattr(self.env, "get_lang_ann") else None
        self._steps = 0
        self._last_action = None
        self._collision = False
        return self._process_obs(obs)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Step with a 7-D action in [-1, 1]; returns (obs, reward, done, info).

        CALVIN consumes relative actions ALREADY in dataset units (pos normalized by
        factor 50, rot by factor 20, gripper ±1 — see the official dataset README),
        i.e. exactly our [-1,1] action space. No denormalization needed.
        """
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # CALVIN expects a BINARY gripper action (-1 close / +1 open); the policy
        # head outputs continuous values -> threshold at 0.
        act[6] = 1.0 if act[6] >= 0.0 else -1.0
        obs, _, _, info = self.env.step(act)
        self._steps += 1
        done = bool(info.get("success", False)) or self._steps >= self.max_episode_steps
        info["truncated"] = self._steps >= self.max_episode_steps
        info["subtasks_completed"] = float(info.get("subtasks_completed", 0.0))
        info["subtasks_total"] = float(info.get("subtasks_total", 0.0))
        info["collision"] = bool(info.get("collision", self._collision))
        info["ee_target_dist"] = info.get("ee_target_dist", None)  # None -> progress proxy
        if self._last_action is not None:
            info["action_jitter"] = float(np.linalg.norm(action - self._last_action))
        else:
            info["action_jitter"] = 0.0
        self._last_action = action
        return self._process_obs(obs), 0.0, done, info

    # ------------------------------------------------------------------ #
    def _process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        img = np.asarray(obs["rgb_obs"][self.cameras[0]])
        # pybullet on this vGPU returns effective pixels as int64 (and occasionally a
        # 1x1 placeholder); normalize dtype and re-render through the camera object
        # if the frame is degenerate.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[0] <= 1 or img.shape[1] <= 1:
            rgb_obs, _ = self.env.get_camera_obs()
            img = np.asarray(rgb_obs[self.cameras[0]])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[0] != self.image_size:
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((self.image_size, self.image_size)))
        # proprio: CALVIN robot_obs = [eef_pos(3), eef_quat(4), gripper(1), ...]
        # (stored for completeness; current groups are image-only)
        proprio = None
        if "robot_obs" in obs:
            proprio = np.asarray(obs["robot_obs"], dtype=np.float32)[:8]
        return {"pixel_values": img, "instruction": self._instruction, "proprio": proprio}

    @staticmethod
    def normalize_action(raw: np.ndarray) -> np.ndarray:
        """Dataset rel_actions are ALREADY in [-1,1] normalized units (official README:
        pos factor 50, rot factor 20). Identity kept for API compatibility."""
        return np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)

    def close(self):
        # Idempotent + defang the inner env's __del__: pybullet REUSES DIRECT
        # connection ids, and a delayed GC of an old env runs close() again,
        # disconnecting the id a NEW env just reconnected with (kills rollouts /
        # evals with 'Not connected to physics server'). Setting p=None makes the
        # inner close() skip its disconnect so only OUR explicit close disconnects.
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env.p = None   # inner close() checks `self.p is not None`
