#!/usr/bin/env python3
"""scripts/train_sft.py — Stage 1 entry point (all groups).

Usage:
    python scripts/train_sft.py --group a0
    python scripts/train_sft.py --group b2 --resume latest
Saves LoRA + action-head checkpoints under runs/<group>/sft/best|last.
The `best` checkpoint is the GRPO starting point (and the frozen KL reference).
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.loader import build_sft_loaders  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.sft import SFTTrainer  # noqa: E402
from src.utils.ckpt import resolve_resume  # noqa: E402
from src.utils.logger import Logger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["a0", "a1", "b0", "b1", "b2", "b3"])
    ap.add_argument("--resume", default=None, help="checkpoint dir | 'latest'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None, help="override sft.max_epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override sft.batch_size")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N training batches (pipeline validation)")
    args = ap.parse_args()

    cfg = load_config(args.group)
    # quick overrides for cold-start probes / hyperparameter sweeps
    if args.epochs is not None:
        cfg.sft.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.sft.batch_size = args.batch_size
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)

    logger = Logger(cfg, cfg.group.id, tags=["sft", cfg.group.family])

    # B3 uses the CSI speech dataset; the other groups use CALVIN + Bridge
    if cfg.group.audio:
        from src.data.csi import CSIDataset
        train_ds = CSIDataset(Path(cfg.dataset.csi_root), Path(cfg.dataset.calvin_root),
                              "sft_train")
        val_ds = CSIDataset(Path(cfg.dataset.csi_root), Path(cfg.dataset.calvin_root),
                            "sft_val")
        from torch.utils.data import DataLoader

        def csi_collate(batch):
            # drop samples whose TTS wav is missing (graceful degradation)
            batch = [b for b in batch if b["audio_wav"] is not None]
            if not batch:
                return None
            px, acts, langs, auds = [], [], [], []
            for b in batch:
                T = b["pixel_values"].shape[0]
                px.append(b["pixel_values"])
                acts.append(b["actions"])
                langs.extend([b["lang"]] * T)
                auds.extend([b["audio_wav"]] * T)
            return {"pixel_values": torch.cat(px), "actions": torch.cat(acts),
                    "lang_tokens": None, "attention_mask": None, "langs": langs,
                    "audio_wavs": auds}
        train_loader = DataLoader(train_ds, batch_size=cfg.sft.batch_size, shuffle=True,
                                  collate_fn=csi_collate, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg.sft.batch_size, shuffle=False,
                                collate_fn=csi_collate, num_workers=args.num_workers)
    else:
        from src.models.factory import build_processor
        processor = build_processor(cfg)
        train_loader = build_sft_loaders(cfg, processor, "sft_train", cfg.sft.batch_size,
                                         args.num_workers)
        val_loader = build_sft_loaders(cfg, processor, "sft_val", cfg.sft.batch_size,
                                       args.num_workers)

    model = build_model(cfg, stage="sft")
    trainer = SFTTrainer(model, cfg, device=args.device)

    resume = resolve_resume(args.resume, cfg.sft_ckpt_dir, "sft")
    best_ckpt = trainer.train(train_loader, val_loader, logger, resume_from=resume,
                              smoke_steps=args.smoke)
    logger.log({"sft/best_ckpt": str(best_ckpt)})
    logger.finish()
    print(f"SFT done. GRPO starting checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
