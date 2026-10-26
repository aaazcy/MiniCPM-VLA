#!/usr/bin/env python3
"""scripts/verify_models.py — Phase-1 GPU validation: load wrappers, run forward
passes on synthetic data, measure per-path VRAM peaks, report trainable params.

Each path is measured separately with caches emptied (avoids accumulation
artifacts). Usage:
    python scripts/verify_models.py --group a1 --batch 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.models.factory import build_model  # noqa: E402


def report(name: str, device: str = "cuda"):
    peak = torch.cuda.max_memory_allocated(device=device) / 1e9
    print(f"  [{name}] peak VRAM: {peak:.2f} GB")
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.empty_cache()


def make_fake_batch(batch: int, image_size: int, audio: bool):
    images = torch.randint(0, 255, (batch, 3, image_size, image_size),
                           dtype=torch.uint8).float() / 255.0
    langs = ["open the drawer and pick up the blue block"] * batch
    audio_wavs = None
    if audio:
        audio_wavs = [torch.randn(16000) for _ in range(batch)]
    return images, langs, audio_wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["a0", "a1", "b0", "b1", "b2", "b3"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.group)
    torch.manual_seed(0)
    model = build_model(cfg, stage="sft").to(args.device)

    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{cfg.group.id}] trainable {n_train/1e6:.2f}M / total {n_total/1e9:.2f}B")
    gc = getattr(model.base, "is_gradient_checkpointing", False)
    gc_llm = getattr(model.llm, "is_gradient_checkpointing", False)
    print(f"grad checkpointing: base={gc} llm={gc_llm}")
    report("load")

    images, langs, audio = make_fake_batch(args.batch, cfg.environment.image_size,
                                           cfg.group.audio)
    images = images.to(args.device)
    audio = [w.to(args.device) for w in audio] if audio else None

    # --- path 1: SFT (teacher-forced) ---
    demo = torch.zeros(args.batch, 7, device=args.device)
    out = model.forward_sft(images, langs, audio, actions=demo)
    print(f"  SFT out: {tuple(out.shape)} range "
          f"[{out.min().item():.3f}, {out.max().item():.3f}]")
    loss = (out ** 2).mean()
    loss.backward()
    report(f"SFT bwd (batch={args.batch})")
    del loss, out
    model.zero_grad(set_to_none=True)

    # --- path 2: GRPO logp ---
    logp = model.logp_actions(images, langs, torch.zeros_like(demo), audio)
    (logp.mean()).backward()
    report(f"GRPO logp bwd (batch={args.batch})")
    del logp
    model.zero_grad(set_to_none=True)

    # --- path 3: rollout/eval generate (batch 1) ---
    with torch.no_grad():
        out1 = model.forward_sft(images[:1], langs[:1], audio[:1] if audio else None)
    print(f"  generate out: {tuple(out1.shape)} | {out1[0].detach().cpu().tolist()}")
    report("generate (batch 1)")
    del out1

    print("OK")


if __name__ == "__main__":
    main()
