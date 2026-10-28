"""src/models/factory.py — build the policy model for a group + stage."""
from __future__ import annotations

from typing import Optional

import torch

from ..config import Config
from .openvla import OpenVLAWrapper
from .minicpmo import MiniCPMoWrapper


def build_model(cfg: Config, stage: str = "sft",
                dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    """Build the policy for `cfg.group.id`.

    stage: "sft" (trainable policy), "grpo" (same), "ref" (frozen SFT reference for KL).
    """
    if dtype is None:
        dtype = torch.bfloat16 if cfg.sft.bf16 else torch.float32

    t = cfg.group.train
    kwargs = dict(
        model_id=cfg.group.model_id,
        use_lora=t.lora,
        lora_r=t.lora_r,
        lora_alpha=t.lora_alpha,
        lora_dropout=t.lora_dropout,
        hf_cache_dir=cfg.experiment.hf_cache_dir,
        dtype=dtype,
    )
    if cfg.group.family == "vla":
        model = OpenVLAWrapper(freeze_vision=t.freeze_vision, **kwargs)
    elif cfg.group.family == "omni":
        model = MiniCPMoWrapper(
            enable_audio=cfg.group.audio,
            audio_sample_rate=cfg.group.audio_sample_rate,
            audio_only=cfg.group.audio_only,
            action_head_type=cfg.model.action_head.type,
            num_bins=cfg.model.action_head.num_bins,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown family {cfg.group.family}")

    if stage in ("sft", "grpo"):
        model.prepare_for_sft()
    elif stage == "ref":
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"unknown stage {stage}")
    return model


def build_processor(cfg: Config):
    """Load only the tokenizer/image processor (no weights) — for dataset prep."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(cfg.group.model_id, trust_remote_code=True,
                                         cache_dir=cfg.experiment.hf_cache_dir)
