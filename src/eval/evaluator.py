"""src/eval/evaluator.py — shared evaluation protocol (identical for all groups).

Metrics (configs/eval.yaml):
  * full-task success rate (primary): ID (held-out calvin_A val) and OOD (calvin_D)
  * sample-efficiency curve: env steps to 50% / 60% success
  * mean episode return, mean steps to finish
  * action smoothness (mean L2 between consecutive actions)
  * mean end-effector positional error
  * robustness degradation (paraphrased / reordered / long instructions)
Qualitative: 50 sampled failed trajectories per group with failure labels (for the
failure pie chart). Auxiliary VLM checks: target-object recall & instruction
understanding accuracy (action head detached).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..envs.calvin_env import CalvinEnv
from ..rollout.engine import RolloutEngine
from ..training.rewards import Trajectory, compute_reward
from ..utils.logger import Logger

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: Config, policy, engine: Optional[RolloutEngine] = None,
                 processor=None, device="cuda"):
        self.cfg = cfg
        self.policy = policy
        self.device = device
        if engine is None:
            from ..rollout.engine import make_engine
            engine = make_engine(cfg, policy=policy, processor=processor or policy.processor,
                                 device=device)
        self.engine = engine

    # ------------------------------------------------------------------ #
    def _load_instruction_audio(self, instruction: str) -> Optional[np.ndarray]:
        """B3: synthesized wav keyed by sha256(instruction) (same convention as
        rollout and preprocess_csi_tts.py)."""
        import hashlib
        import soundfile as sf
        wav_dir = Path(self.cfg.dataset.csi_root) / "wav"
        name = hashlib.sha256(instruction.encode()).hexdigest()[:16]
        p = wav_dir / f"{name}.wav"
        if not p.exists():
            return None
        audio, _ = sf.read(str(p), dtype="float32")
        return audio

    def evaluate_episode(self, instruction: str, seeds: int, temperature: float,
                         audio_wav: Optional[np.ndarray] = None,
                         init_robot_obs: Optional[np.ndarray] = None,
                         init_scene_obs: Optional[np.ndarray] = None,
                         scene: str = "calvin_scene_A",
                         task: Optional[str] = None) -> Dict:
        """Repeat one episode `seeds` times -> mean +/- std of all metrics.

        init_robot_obs/init_scene_obs: recorded episode start state (official CALVIN
        protocol). scene: calvin_scene_A (ID) or calvin_scene_D (OOD).
        task: CALVIN subtask name (for success detection via the Tasks oracle).
        """
        if self.cfg.group.audio and audio_wav is None:
            audio_wav = self._load_instruction_audio(instruction)
        successes, returns, steps_list, smoothness, ee_errs = [], [], [], [], []
        trajs: List[Trajectory] = []
        for seed in range(seeds):
            env = CalvinEnv(max_episode_steps=self.cfg.environment.max_episode_steps,
                            cameras=self.cfg.environment.cameras,
                            image_size=self.cfg.environment.image_size,
                            seed=seed, calvin_root=Path(self.cfg.dataset.calvin_root),
                            scene=scene)
            obs = env.reset(instruction, robot_obs=init_robot_obs,
                            scene_obs=init_scene_obs, task=task)
            traj = Trajectory()
            for _ in range(self.cfg.environment.max_episode_steps):
                act = self.engine.act(obs, instruction, audio_wav=audio_wav,
                                      temperature=temperature)
                traj.obs.append(obs)
                traj.actions.append(act)
                obs, _, done, info = env.step(act)
                traj.infos.append(info)
                if done:
                    break
            traj.success = bool(traj.infos[-1].get("success", False)) if traj.infos else False
            trajs.append(traj)
            env.close()
            # CRITICAL: free the env NOW. pybullet reuses DIRECT connection ids;
            # if the old env's __del__ runs AFTER the next env connects (Python
            # evaluates the RHS of `env = CalvinEnv(...)` before rebinding), its
            # second disconnect() kills the new env's connection ("Not connected").
            env = None

            successes.append(float(traj.success))
            returns.append(compute_reward(traj, self.cfg))
            steps_list.append(float(traj.length))
            if traj.length > 1:
                smoothness.append(np.mean([np.linalg.norm(traj.actions[i] - traj.actions[i - 1])
                                           for i in range(1, traj.length)]))
            else:
                smoothness.append(0.0)
            ee_err = np.mean([info.get("ee_target_dist", np.nan)
                              for info in traj.infos if info.get("ee_target_dist") is not None]) \
                if any(info.get("ee_target_dist") is not None for info in traj.infos) else np.nan
            ee_errs.append(ee_err)

        return {
            "success_rate": float(np.mean(successes)),
            "success_std": float(np.std(successes)),
            "mean_return": float(np.mean(returns)),
            "mean_steps": float(np.mean(steps_list)),
            "action_smoothness": float(np.mean(smoothness)),
            "mean_ee_error": float(np.nanmean(ee_errs)) if not np.all(np.isnan(ee_errs)) else None,
        }

    def evaluate_subset(self, instructions: List[str], seeds: int,
                        temperature: float, scenes: List[str] = None,
                        init_states: List = None, tasks: List = None) -> Dict:
        instructions = instructions[: self.cfg.eval.max_episodes]
        if tasks is not None:
            tasks = tasks[: self.cfg.eval.max_episodes]
        aggr = {"success_rate": [], "mean_return": [], "mean_steps": [],
                "action_smoothness": [], "mean_ee_error": []}
        for i, instr in enumerate(instructions):
            scene = scenes[i] if scenes else "calvin_scene_A"
            rs, ss = None, None
            if init_states and init_states[i] is not None:
                rs, ss = init_states[i]
            task = tasks[i] if tasks else None
            m = self.evaluate_episode(instr, seeds, temperature,
                                      init_robot_obs=rs, init_scene_obs=ss, scene=scene,
                                      task=task)
            for k in aggr:
                v = m[k]
                if v is not None:
                    aggr[k].append(v)
        return {k: float(np.mean(v)) for k, v in aggr.items() if v}

    # ------------------------------------------------------------------ #
    def run_full(self, id_instructions: List[str], ood_instructions: List[str],
                 robust_instructions: Optional[List[str]] = None,
                 logger: Optional[Logger] = None, out_dir: Path = Path("runs")) -> Dict:
        """Full protocol: ID + OOD + robustness, 20 seeds per episode."""
        cfg = self.cfg
        results = {}
        results["id"] = self.evaluate_subset(id_instructions, cfg.eval.seeds, 0.0)
        results["ood"] = self.evaluate_subset(ood_instructions, cfg.eval.seeds, 0.0)
        if robust_instructions:
            base = self.evaluate_subset(id_instructions, cfg.eval.seeds, 0.0)
            rob = self.evaluate_subset(robust_instructions, cfg.eval.seeds, 0.0)
            results["robustness"] = {
                "robust_success": rob["success_rate"],
                "base_success": base["success_rate"],
                "degradation": base["success_rate"] - rob["success_rate"],
            }
        if logger is not None:
            logger.log({"eval/id_success": results["id"]["success_rate"],
                        "eval/ood_success": results["ood"]["success_rate"]}, commit=True)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    # ------------------------------------------------------------------ #
    # Qualitative: failure labeling (feeds viz/failure pie chart)
    # ------------------------------------------------------------------ #
    def collect_failures(self, instructions: List[str], n_failed: int = 20,
                         out_path: Path = Path("runs/failures.json"),
                         tasks: List = None) -> List[Dict]:
        """Sample failed trajectories and dump raw data for (manual/semi-automatic)
        failure labeling: language_misunderstanding / visual_localization_error /
        action_drift / wrong_temporal_planning / rl_local_loop."""
        failures = []
        for i, instr in enumerate(instructions):
            if len(failures) >= n_failed:
                break
            task = tasks[i] if tasks else None
            env = CalvinEnv(max_episode_steps=self.cfg.environment.max_episode_steps,
                            cameras=self.cfg.environment.cameras,
                            image_size=self.cfg.environment.image_size,
                            seed=0, calvin_root=Path(self.cfg.dataset.calvin_root))
            obs = env.reset(instr, task=task)
            traj = Trajectory()
            for _ in range(self.cfg.environment.max_episode_steps):
                act = self.engine.act(obs, instr, temperature=0.0)
                traj.obs.append(obs)
                traj.actions.append(act)
                obs, _, done, info = env.step(act)
                traj.infos.append(info)
                if done:
                    break
            env.close()
            if not (traj.infos and traj.infos[-1].get("success")):
                failures.append({"instruction": instr, "length": traj.length,
                                 "actions": [a.tolist() for a in traj.actions],
                                 "label": None})  # label filled by viz/labeling step
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(failures, f, indent=2)
        return failures

    # ------------------------------------------------------------------ #
    # Auxiliary VLM checks (action head detached)
    # ------------------------------------------------------------------ #
    def auxiliary_vlm_check(self, instruction: str, target_object: str) -> Dict:
        """Target-object recall + instruction understanding, using the LLM's language
        capability WITHOUT the action head (visual + text inputs only).

        NOTE: relies on the base model's text-generation path; for MiniCPM-o the TTS
        branch is removed but the LLM itself can still answer. Implementation kept
        generic; revisit exact prompting per model family in Phase 1.
        """
        raise NotImplementedError(
            "auxiliary VLM check needs the model-specific generation path; implement in "
            "Phase 1 after the checkpoint is verified (see docs/limitations.md §10)"
        )
