"""src/training/sft.py — Stage 1: supervised fine-tuning (identical for all groups).

Loss: MSE(predicted_action, demo_action).
Optimizer: AdamW with two parameter groups:
    action head  lr = 1e-4
    LoRA         lr = 2e-5
Batch 8, weight decay 1e-4, gradient checkpointing on, BF16 autocast.
Early stop on SFT-val MSE plateau; the best checkpoint becomes the GRPO starting point.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.logger import Logger

log = logging.getLogger(__name__)


def make_optimizer(model: nn.Module, lr_head: float, lr_lora: float,
                   weight_decay: float) -> torch.optim.AdamW:
    head_params = []
    lora_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "action_head" in name or "native" in name:
            head_params.append(p)
        else:
            lora_params.append(p)
    groups = [
        {"params": head_params, "lr": lr_head},
        {"params": lora_params, "lr": lr_lora},
    ]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


class SFTTrainer:
    def __init__(self, model, cfg, device="cuda", is_main: bool = True):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.is_main = is_main
        # modern DDP does NOT forward arbitrary attrs (dtype, custom methods):
        # keep a handle on the raw module for those
        self.raw = model.module if hasattr(model, "module") else model
        self.model_dtype = getattr(self.raw, "dtype", torch.bfloat16)
        self.model.to(device)
        self.opt = make_optimizer(model, cfg.sft.lr_action_head, cfg.sft.lr_lora,
                                  cfg.sft.weight_decay)
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_left = cfg.sft.early_stop_patience
        self.epoch = 0
        self.global_step = 0
        if cfg.sft.gradient_checkpointing:
            for target in (self.raw, getattr(self.raw, "llm", None),
                           getattr(getattr(self.raw, "llm", None), "base_model", None)):
                if target is not None and hasattr(target, "gradient_checkpointing_enable"):
                    try:
                        target.gradient_checkpointing_enable()
                    except Exception as e:
                        log.warning("gradient checkpointing failed on %s: %s",
                                    type(target).__name__, e)

    # ------------------------------------------------------------------ #
    def run_epoch(self, loader, logger: Logger, train: bool = True,
                  max_batches: int = None):
        """Run one epoch -> (mean_loss, mean_mse).

        mean_loss is the optimization objective (CE for discrete heads, MSE for A);
        mean_mse is the expected-action MSE (diagnostic, cross-family comparable).
        """
        self.model.train(train)
        total_loss, total_mse, n = 0.0, 0.0, 0
        torch.set_grad_enabled(train)
        for bi, batch in enumerate(loader):
            if batch is None:  # collate can drop empty batches (B3 wav missing)
                continue
            if max_batches is not None and bi >= max_batches:
                break
            px = batch["pixel_values"].to(self.device, torch.float32)
            act = batch["actions"].to(self.device, self.model_dtype)

            with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=self.cfg.sft.bf16):
                audio = batch.get("audio_wavs")
                # Always go through self.model(...) so DDP's forward hooks fire.
                # MiniCPM-o discrete head: forward now returns raw logits (B,7,num_bins)
                # -> cross-entropy on quantized bins (same objective the old non-DDP
                # sft_loss path used). OpenVLA native head / continuous head: forward
                # returns the expected action -> MSE (unchanged behaviour).
                out = self.model(px, batch["langs"], audio, act)
                if getattr(self.raw, "action_head_type", None) == "discrete":
                    logits = out
                    q = self.raw.quantize(act.to(logits.device))
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           q.reshape(-1))
                    with torch.no_grad():
                        mse = float(F.mse_loss(
                            self.raw.expected_action(logits),
                            act.to(logits.device)))
                else:
                    loss = self.loss_fn(out, act)
                    mse = float(loss.detach())
            loss_val = float(loss.detach())

            if train:
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.raw.trainable_parameters(), 1.0)
                self.opt.step()
                if self.is_main and self.global_step % self.cfg.wandb.log_every_steps == 0:
                    logger.log({"sft/train_mse": mse, "sft/train_loss": loss_val,
                                "sft/step": self.global_step}, step=self.global_step)
            total_loss += loss_val * len(act)
            total_mse += mse * len(act)
            n += len(act)
            self.global_step += 1
        return total_loss / max(n, 1), total_mse / max(n, 1)

    # ------------------------------------------------------------------ #
    def train(self, train_loader, val_loader, logger: Logger,
              resume_from: Optional[Path] = None, smoke_steps: int = 0) -> Path:
        """Run SFT with early stopping; returns path of the best checkpoint.

        smoke_steps > 0: run only N training batches (pipeline validation), no
        epochs / early stopping.
        """
        if resume_from is not None:
            self._load_state(resume_from)

        ckpt_dir = self.cfg.sft_ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if smoke_steps:
            train_loss, train_mse = self.run_epoch(train_loader, logger, train=True,
                                                   max_batches=smoke_steps)
            # ALL ranks run val (no collectives, but keeps ranks in lockstep so
            # nobody destroys the process group while others still sync)
            val_loss, val_mse = self.run_epoch(val_loader, logger, train=False,
                                               max_batches=smoke_steps)
            if self.is_main:
                logger.log({"sft/smoke_train_mse": train_mse, "sft/smoke_val_mse": val_mse,
                            "sft/smoke_train_loss": train_loss,
                            "sft/smoke_val_loss": val_loss})
                log.info("[SFT smoke] train loss %.4f / mse %.4f | val loss %.4f / mse %.4f",
                         train_loss, train_mse, val_loss, val_mse)
                self.save(ckpt_dir / "smoke")
            return ckpt_dir / "smoke"
        history: Dict[str, list] = {"train_loss": [], "val_loss": [],
                                    "train_mse": [], "val_mse": []}

        for epoch in range(self.epoch, self.cfg.sft.max_epochs):
            self.epoch = epoch
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_loss, train_mse = self.run_epoch(train_loader, logger, train=True)
            # ALL ranks run val: keeps ranks in lockstep (main-only val lets
            # non-main ranks exit/destroy the group while main still syncs)
            val_loss, val_mse = self.run_epoch(val_loader, logger, train=False)
            if self.is_main:
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                logger.log({"sft/epoch": epoch, "sft/train_loss_epoch": train_loss,
                            "sft/val_loss": val_loss, "sft/val_mse": val_mse}, step=epoch)
                log.info("[SFT] epoch %d | train loss %.4f / val loss %.4f | val mse %.4f",
                         epoch, train_loss, val_loss, val_mse)

            if val_loss < self.best_val_loss - 1e-7:
                self.best_val_loss = val_loss
                self.patience_left = self.cfg.sft.early_stop_patience
                if self.is_main:
                    self.save(ckpt_dir / "best")
            else:
                self.patience_left -= 1
                if self.patience_left <= 0:
                    log.info("[SFT] early stop at epoch %d (val loss plateau)", epoch)
                    break
            if self.is_main:
                self.save(ckpt_dir / "last")

        if self.is_main:
            with open(ckpt_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
        return ckpt_dir / "best"

    # ------------------------------------------------------------------ #
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.raw.save_adapter(str(path))
        torch.save({"opt": self.opt.state_dict(), "epoch": self.epoch,
                    "best_val_loss": self.best_val_loss, "cfg": self.cfg.to_dict()},
                   path / "trainer_state.pt")

    def _load_state(self, path: Path):
        state = torch.load(path / "trainer_state.pt", map_location="cpu")
        self.opt.load_state_dict(state["opt"])
        self.epoch = state["epoch"] + 1
        self.best_val_loss = state.get("best_val_loss", state.get("best_val_mse", float("inf")))
        self.model.load_adapter(str(path))
        log.info("[SFT] resumed from %s (next epoch %d)", path, self.epoch)
