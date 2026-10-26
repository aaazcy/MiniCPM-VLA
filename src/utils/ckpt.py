"""src/utils/ckpt.py — checkpoint discovery for resuming runs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


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
