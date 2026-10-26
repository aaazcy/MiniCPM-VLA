"""src/utils/logger.py — wandb metric logging with offline fallback."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class Logger:
    """Thin wandb wrapper. Falls back to CSV/JSON when wandb is unavailable."""

    def __init__(self, cfg, group: str, tags: Optional[list] = None):
        self.cfg = cfg
        self.group = group
        self._wandb = None
        try:
            import wandb
            tags = tags or []
            if cfg.group.is_ablation:
                tags.append("ablation")  # B3 never enters head-to-head comparisons
            self._wandb = wandb.init(
                project=cfg.experiment.wandb_project,
                entity=cfg.experiment.wandb_entity,
                name=f"{group}-{wandb.util.generate_id()[:6]}",
                config=cfg.to_dict(),
                tags=tags,
                mode=cfg.wandb.mode,
                dir=str(cfg.run_dir),
            )
        except Exception as e:  # pragma: no cover
            log.warning("wandb unavailable (%s); logging to stdout only", e)
            self._wandb = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = False):
        if self._wandb is not None:
            self._wandb.log(metrics, step=step, commit=commit)
        else:
            log.info("METRIC %s: %s", step, metrics)

    def finish(self):
        if self._wandb is not None:
            self._wandb.finish()
