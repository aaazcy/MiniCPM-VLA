"""src/rollout/engine.py — rollout inference abstraction.

Two engines behind one interface:
  * VLLMEngine  — vision-only groups (OpenVLA A0/A1). OpenVLA's native action head is
                  decoded from generated action tokens (standard vLLM OpenVLA path).
                  LoRA adapters are attached via vLLM's LoRA support.
  * HFEngine    — MiniCPM-o groups (B0/B1/B2) and B3 (audio). Stock vLLM has no
                  Whisper-audio path for MiniCPM-o, and our custom ActionHead lives
                  outside vLLM, so these groups run with transformers/accelerate.

NOTE (adjustment): the spec asks for vLLM rollout for all groups; vLLM supports the
OpenVLA path natively, but MiniCPM-o + custom action head requires a custom vLLM
model class (TODO in docs). HFEngine keeps all groups runnable today.
"""
from __future__ import annotations

import abc
from typing import Dict, List, Optional

import numpy as np
import torch


class RolloutEngine(abc.ABC):
    @abc.abstractmethod
    def act(self, obs: Dict, lang: str, audio_wav: Optional[np.ndarray] = None,
            temperature: float = 1.0) -> np.ndarray:
        """Return a 7-D action in [-1, 1]."""


# --------------------------------------------------------------------------- #
class HFEngine(RolloutEngine):
    """Direct model inference (works for OpenVLA, MiniCPM-o, and B3 audio)."""

    def __init__(self, policy, processor, device="cuda", dtype=torch.bfloat16,
                 image_size: int = 224):
        self.policy = policy
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self.policy.eval()
        # Gaussian exploration for GRPO rollouts (eval uses temperature=0 ->
        # fully deterministic). WITHOUT this the MiniCPM-o MLP action head is
        # deterministic, all 4 rollouts in a group are near-identical, the
        # group-relative advantage collapses to ~0 and GRPO learns nothing.
        self.rng = np.random.default_rng()

    @torch.no_grad()
    def act(self, obs, lang, audio_wav=None, temperature=1.0) -> np.ndarray:
        img = obs["pixel_values"]
        if img.max() > 1.5:  # uint8
            img = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device, self.dtype)
        if img_t.dim() == 3:
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        audio_l = [torch.from_numpy(audio_wav).float()] if audio_wav is not None else None
        out = self.policy.forward_sft(img_t, [lang], audio_l)
        action = out[0].detach().cpu().numpy().astype(np.float32)
        if temperature > 0:
            # stochastic rollout: add Gaussian noise (scaled by temperature)
            action = action + self.rng.normal(0.0, 0.1 * temperature,
                                              size=action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
class VLLMEngine(RolloutEngine):
    """OpenVLA rollout via vLLM offline batch + native action-token decoding.

    Implementation notes (verify in Phase 1 after GPU mount):
      * vLLM 0.6.6.post1 supports OpenVLA-7B; LoRA via enable_lora=True.
      * Action tokens are decoded back to the 7-D continuous action with OpenVLA's
        own bin centers (identical to the OpenVLA eval pipeline).
    """

    BIN_CENTERS = None  # filled from model config on first use

    def __init__(self, model_id: str, lora_path: Optional[str] = None,
                 gpu_memory_utilization: float = 0.85, dtype: str = "bfloat16"):
        from vllm import LLM, SamplingParams  # lazy import (GPU only)

        self.model_id = model_id
        self.lora_path = lora_path
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            enable_lora=lora_path is not None,
            max_lora_rank=64,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        # LoRA requests are passed per-generate call via LoRARequest (see act());
        # the adapter must be served from a shared directory vLLM can read.
        self.lora_path = lora_path
        self.sampling = SamplingParams(temperature=1.0, max_tokens=7, top_p=0.95)
        # OpenVLA discretizes each action dim over 256 uniform bins in [-1, 1]
        self.BIN_CENTERS = np.linspace(-1, 1, 256, dtype=np.float32)

    def act(self, obs, lang, audio_wav=None, temperature=1.0) -> np.ndarray:
        img = obs["pixel_values"]
        if img.max() > 1.5:
            img = (img.astype(np.float32) / 255.0)
        prompt = {"prompt": lang, "multi_modal_data": {"image": img}}
        self.sampling.temperature = temperature if temperature > 0 else 1e-6
        out = self.llm.generate([prompt], sampling_params=self.sampling,
                                lora_request=self._lora_req())[0]
        tokens = out.outputs[0].token_ids[:7]  # 7 action tokens
        action = np.array([self.BIN_CENTERS[t] for t in tokens], dtype=np.float32)
        return action

    def _lora_req(self):
        if self.lora_path is None:
            return None
        from vllm.lora.request import LoRARequest
        return LoRARequest("policy", 1, self.lora_path)


# --------------------------------------------------------------------------- #
def make_engine(cfg, policy=None, processor=None, lora_path: Optional[str] = None,
                device="cuda") -> RolloutEngine:
    """Build the configured engine for a group.

    B3 (audio) and MiniCPM-o groups always use HFEngine (see module docstring).
    OpenVLA groups use VLLMEngine when `rollout.engine == "vllm"`.
    """
    engine = cfg.group.rollout_engine_override or cfg.rollout.engine
    if engine == "hf" or cfg.group.family == "omni" or cfg.group.audio:
        assert policy is not None, "HFEngine needs the policy model"
        return HFEngine(policy, processor, device=device,
                        image_size=cfg.environment.image_size)
    if engine == "vllm":
        return VLLMEngine(cfg.group.model_id, lora_path=lora_path,
                          gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
                          dtype=cfg.rollout.dtype)
    raise ValueError(f"unknown engine {engine}")
