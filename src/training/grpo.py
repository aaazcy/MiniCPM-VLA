"""src/training/grpo.py — Stage 2: GRPO reinforcement-learning post-training.

Fixed hyperparameters for ALL groups (locked in configs/base.yaml):
    group_size=4, rollout_per_group=4, clip_eps=0.2, beta_kl=0.05
    lr_action_head=1e-4, lr_lora=2e-5, max_grad_norm=1.0, AdamW

Loop (per spec):
    1. Rollout trajectories in CALVIN with the policy (vLLM / HF engine).
    2. Compute group-relative advantages (GRPO — no separate critic).
    3. Clipped policy loss + KL penalty against the frozen SFT reference checkpoint.
    4. Backprop ONLY LoRA + action head.
    5. Every 200 env episodes: pause, deterministic evaluation (temp=0, no noise),
       log metrics and save a checkpoint.

Resumable: checkpoints store trainer state + adapter weights.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..envs.calvin_env import CalvinEnv
from ..rollout.engine import RolloutEngine, HFEngine
from ..training.rewards import Trajectory, compute_reward, grpo_advantage
from ..utils.logger import Logger

log = logging.getLogger(__name__)


class GRPOTrainer:
    def __init__(self, policy, reference, cfg: Config, engine: RolloutEngine,
                 instruction_sampler, device="cuda"):
        self.policy = policy
        self.reference = reference          # frozen SFT checkpoint (KL anchor)
        self.cfg = cfg
        self.engine = engine
        self.instruction_sampler = instruction_sampler  # yields training instructions
        self.device = device
        self.policy.to(device)

        from .sft import make_optimizer
        self.opt = make_optimizer(policy, cfg.grpo.lr_action_head, cfg.grpo.lr_lora,
                                  weight_decay=cfg.sft.weight_decay)

        self.env_steps = 0
        self.update_count = 0
        self.metrics_history: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    def rollout_one(self, instruction: str, temperature: float) -> Trajectory:
        env = CalvinEnv(max_episode_steps=self.cfg.environment.max_episode_steps,
                        cameras=self.cfg.environment.cameras,
                        image_size=self.cfg.environment.image_size,
                        calvin_root=Path(self.cfg.dataset.calvin_root))
        obs = env.reset(instruction)
        traj = Trajectory()
        audio_wav = None
        if self.cfg.group.audio:
            # B3: load pre-synthesized wav for this instruction (see csi.py)
            audio_wav = self._load_instruction_audio(instruction)

        for _ in range(self.cfg.grpo.max_traj_len):
            action = self.engine.act(obs, instruction, audio_wav=audio_wav,
                                     temperature=temperature)
            traj.obs.append(obs)
            traj.actions.append(action)
            obs, _, done, info = env.step(action)
            traj.infos.append(info)
            if done:
                break
        traj.done = True
        traj.success = bool(traj.infos[-1].get("success", False)) if traj.infos else False
        traj.audio_wav = audio_wav   # keep the wav so the GRPO update uses the SAME input
        env.close()
        return traj

    def _load_instruction_audio(self, instruction: str) -> Optional[np.ndarray]:
        import soundfile as sf
        import hashlib
        wav_dir = Path(self.cfg.dataset.csi_root) / "wav"
        name = hashlib.sha256(instruction.encode()).hexdigest()[:16]
        p = wav_dir / f"{name}.wav"
        if not p.exists():
            return None  # rollout without audio (graceful degradation)
        audio, _ = sf.read(str(p), dtype="float32")
        return audio

    # ------------------------------------------------------------------ #
    # Update (GRPO loss)
    # ------------------------------------------------------------------ #
    CHUNK = 4  # batch size for batched logp re-encoding (VRAM-bounded: bwd batch4 ~23G)

    def update(self, groups: List[List[Trajectory]], logger: Logger) -> Dict:
        """One GRPO update over rollout groups. `groups`: n_groups x group_size."""
        n_groups = len(groups)
        g = self.cfg.grpo

        # ---- step 2: group-relative advantage ----
        R = np.array([[compute_reward(t, self.cfg) for t in group] for group in groups])
        A = grpo_advantage(R, g.group_size)                     # (n_groups, group_size)
        A = torch.from_numpy(A).float().to(self.device)

        # ---- step 3: batched logp re-encoding ----
        # policy and frozen reference are BOTH ~15-19GB; they cannot coexist on a
        # 32GB card. Chunked two-phase: reference (no-grad) first, offloaded to CPU;
        # then policy (grad) alone on the GPU.
        trajs = [t for group in groups for t in group]
        traj_sizes = [t.length for t in trajs]
        steps = []  # (img, lang, act, audio) flattened
        for traj in trajs:
            audio = [traj.audio_wav] if traj.audio_wav is not None else None
            for obs, act in zip(traj.obs, traj.actions):
                img, lang = self._obs_to_tensor(obs)
                steps.append((img, lang,
                              torch.from_numpy(act).float().to(self.device), audio))

        def _batch_logps(model, step_list, grad: bool) -> torch.Tensor:
            out = []
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                for i in range(0, len(step_list), self.CHUNK):
                    chunk = step_list[i:i + self.CHUNK]
                    imgs = torch.cat([s[0] for s in chunk], dim=0)
                    langs = [s[1] for s in chunk]
                    acts = torch.stack([s[2] for s in chunk])
                    audio = [s[3] for s in chunk] if any(s[3] is not None for s in chunk) \
                        else None
                    out.append(model.logp_actions(imgs, langs, acts, audio))
            return torch.cat(out)

        # phase A: reference logp (frozen, detached) — policy offloaded to CPU first
        self.policy.to("cpu")
        self.reference.to(self.device)
        ref = _batch_logps(self.reference, steps, grad=False).detach()
        self.reference.to("cpu")
        torch.cuda.empty_cache()

        # phase B: policy logp (grad) — alone on the GPU
        self.policy.to(self.device)
        pi = _batch_logps(self.policy, steps, grad=True)

        # regroup into (n_groups, group_size) sums per trajectory
        logp_pi, logp_ref, logp_old = [], [], []
        start = 0
        for size in traj_sizes:
            logp_pi.append(pi[start:start + size].sum())
            logp_ref.append(ref[start:start + size].sum())
            start += size
        logp_pi = torch.stack(logp_pi).reshape(n_groups, g.group_size)
        logp_ref = torch.stack(logp_ref).reshape(n_groups, g.group_size)
        logp_old = logp_pi.detach()

        ratio = torch.exp(logp_pi - logp_old)
        clipped = torch.clamp(ratio, 1 - g.clip_eps, 1 + g.clip_eps)
        policy_loss = -torch.min(ratio * A, clipped * A).mean()
        kl = (logp_ref - logp_pi).mean()          # KL(policy || ref) approximation
        loss = policy_loss + g.beta_kl * kl

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), g.max_grad_norm)
        self.opt.step()

        stats = {"grpo/loss": float(loss), "grpo/policy_loss": float(policy_loss),
                 "grpo/kl": float(kl), "grpo/mean_reward": float(R.mean()),
                 "grpo/success_rate": float(np.mean([t.success for grp in groups for t in grp]))}
        logger.log({**stats, "grpo/update": self.update_count}, step=self.env_steps)
        self.update_count += 1
        return stats

    def _traj_logps(self, traj: Trajectory):
        """Sum of step logp under policy (grad) and reference (no grad)."""
        audio = [traj.audio_wav] if traj.audio_wav is not None else None
        sum_pi, sum_ref = 0.0, 0.0
        for obs, act in zip(traj.obs, traj.actions):
            img, lang = self._obs_to_tensor(obs)
            act_t = torch.from_numpy(act).float().to(self.device)
            lp = self.policy.logp_actions(img, [lang], act_t.unsqueeze(0), audio)
            sum_pi = sum_pi + lp[0]
            with torch.no_grad():
                lp_r = self.reference.logp_actions(img, [lang], act_t.unsqueeze(0), audio)
            sum_ref = sum_ref + lp_r[0]
        return sum_pi, sum_ref

    def _obs_to_tensor(self, obs):
        img = obs["pixel_values"]
        if img.max() > 1.5:
            img = img.astype(np.float32) / 255.0
        px = torch.from_numpy(np.ascontiguousarray(img)).to(self.device, self.policy.dtype)
        if px.dim() == 3:
            px = px.permute(2, 0, 1).unsqueeze(0)
        return px, obs["instruction"]

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def train(self, logger: Logger, resume_from: Optional[Path] = None):
        if resume_from is not None:
            self._load_state(resume_from)

        g = self.cfg.grpo
        ckpt_dir = self.cfg.grpo_ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        while self.env_steps < g.total_env_steps:
            steps_before = self.env_steps
            # step 1: rollout g.rollout_per_group groups x group_size trajectories
            groups = []
            for _ in range(g.rollout_per_group):
                group = []
                for _ in range(g.group_size):
                    instr = self.instruction_sampler.sample()
                    traj = self.rollout_one(instr, temperature=self.cfg.rollout.temperature)
                    group.append(traj)
                    self.env_steps += traj.length
                groups.append(group)

            # steps 2-4
            stats = self.update(groups, logger)
            stats["env_steps"] = self.env_steps
            self.metrics_history.append(stats)

            # step 5: periodic deterministic eval + checkpoint (every 200 env episodes)
            if self.env_steps // g.eval_every_episodes > steps_before // g.eval_every_episodes:
                eval_metrics = self.deterministic_eval(logger)
                logger.log(eval_metrics, step=self.env_steps)
                self.save(ckpt_dir / f"step_{self.env_steps}")
            self.save(ckpt_dir / "last")

        log.info("[GRPO] finished: %d env steps, %d updates", self.env_steps, self.update_count)
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        return ckpt_dir / "last"

    def deterministic_eval(self, logger: Logger, n_episodes: int = 5) -> Dict:
        """Deterministic evaluation: temperature=0, no exploration noise."""
        from ..eval.evaluator import Evaluator
        eval_cfg = self.cfg.model_copy(deep=True)
        eval_cfg.eval.seeds = 1
        evaluator = Evaluator(eval_cfg, self.policy, engine=self.engine)
        metrics = evaluator.evaluate_subset(self.instruction_sampler.sample_eval(n_episodes),
                                            seeds=1, temperature=0.0)
        return {f"grpo_eval/{k}": v for k, v in metrics.items()}

    # ------------------------------------------------------------------ #
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.policy.save_adapter(str(path))
        torch.save({"opt": self.opt.state_dict(), "update_count": self.update_count,
                    "env_steps": self.env_steps, "history": self.metrics_history,
                    "cfg": self.cfg.to_dict()}, path / "trainer_state.pt")

    def _load_state(self, path: Path):
        state = torch.load(path / "trainer_state.pt", map_location="cpu")
        self.opt.load_state_dict(state["opt"])
        self.update_count = state["update_count"]
        self.env_steps = state["env_steps"]
        self.metrics_history = state["history"]
        self.policy.load_adapter(str(path))
        log.info("[GRPO] resumed from %s (env_steps=%d)", path, self.env_steps)
