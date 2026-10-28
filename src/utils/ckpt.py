"""src/utils/ckpt.py — checkpoint discovery + compatibility guards for runs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def find_latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    """Return the most recent checkpoint dir (by mtime) with an adapter saved."""
    if not ckpt_dir.exists():
        return None
    candidates = [p for p in ckpt_dir.iterdir()
                  if p.is_dir() and (p / "trainer_state.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "trainer_state.pt").stat().st_mtime)


def resolve_resume(path: Optional[str], ckpt_dir: Path, stage: str) -> Optional[Path]:
    """--resume flag: explicit path | 'latest' | None."""
    if path is None:
        return None
    if path == "latest":
        p = find_latest_ckpt(ckpt_dir)
        if p is None:
            raise FileNotFoundError(f"no resumable checkpoint in {ckpt_dir}")
        return p
    p = Path(path)
    if not (p / "trainer_state.pt").exists():
        raise FileNotFoundError(f"{p} is not a valid checkpoint ({stage})")
    return p


def ckpt_action_head(ckpt_dir: Path) -> Optional[dict]:
    """Action-head config stored in a checkpoint's trainer_state.pt, or None.

    Checkpoints save the FULL resolved cfg (see SFTTrainer.save / GRPOTrainer.save);
    this lets us detect stale checkpoints from a previous architecture (e.g. the
    continuous MLP head vs the current discretized 7x255-bin head) BEFORE loading
    and failing with an opaque shape mismatch.
    """
    import torch
    st = ckpt_dir / "trainer_state.pt"
    if not st.exists():
        return None
    try:
        state = torch.load(st, map_location="cpu", weights_only=False)
    except Exception as e:  # corrupt / legacy pickle
        log.warning("could not read %s: %s", st, e)
        return None
    cfg = state.get("cfg") or {}
    model = cfg.get("model") or {}
    return model.get("action_head") or None


def head_compatible(ckpt_dir: Path, cfg) -> bool:
    """True if the checkpoint's action-head type/num_bins match the current cfg."""
    saved = ckpt_action_head(ckpt_dir)
    if saved is None:
        # no stored metadata (very old ckpt): assume the historical default
        saved = {"type": "continuous", "num_bins": 0}
    cur = cfg.model.action_head
    return (str(saved.get("type")) == str(cur.type)
            and int(saved.get("num_bins", 0) or 0) == int(cur.num_bins or 0))


def pick_compatible_ckpt(candidates, cfg) -> Optional[Path]:
    """First candidate that exists AND matches the current action-head config."""
    for c in candidates:
        if not (c / "trainer_state.pt").exists():
            continue
        if head_compatible(c, cfg):
            return c
        log.warning("[ckpt] skipping %s: action-head config does not match current "
                    "cfg (%s)", c, cfg.model.action_head)
    return None
