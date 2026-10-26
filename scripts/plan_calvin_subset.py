#!/usr/bin/env python3
"""scripts/plan_calvin_subset.py — window-level selection of env-A steps.

CALVIN structure (verified): 147 long "episodes" (avg 12.2k steps) in the training
split; env A = 46 episodes / 603,660 steps (scene_info range [1191339, 1795044]).
Language is annotated per 64-step WINDOW (lang_annotations info.indx). We select
windows (not whole episodes) to spread the subset across all env-A episodes while
keeping the download at post-training scale (~10-15 GB).

Outputs <root>/plan.json with the exact step-id lists for selective extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_episodes(split_dir: Path) -> np.ndarray:
    return np.load(split_dir / "ep_start_end_ids.npy", allow_pickle=True).reshape(-1, 2)


def load_scene_info(split_dir: Path) -> dict:
    p = split_dir / "scene_info.npy"
    if not p.exists():
        return {}
    return np.load(p, allow_pickle=True).item()


def load_windows(lang_ann: Path) -> np.ndarray:
    """(K,2) 64-step window ranges from auto_lang_ann.npy."""
    d = np.load(lang_ann, allow_pickle=True).item()
    return np.asarray(d["info"]["indx"], dtype=np.int64).reshape(-1, 2)


def expand(steps_iterable) -> list:
    out = []
    for s, e in steps_iterable:
        out.extend(range(int(s), int(e) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-meta", required=True)
    ap.add_argument("--validation-meta", required=True)
    ap.add_argument("--env", default="A", choices=list("ABCD"))
    ap.add_argument("--train-window-stride", type=int, default=12,
                    help="keep every Nth env-A window (64 steps each)")
    ap.add_argument("--val-window-stride", type=int, default=32,
                    help="keep every Nth validation window")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "calvin" / "plan.json"))
    args = ap.parse_args()

    tr = Path(args.training_meta)
    val = Path(args.validation_meta)

    tr_eps = load_episodes(tr)
    scene = load_scene_info(tr)
    if f"calvin_scene_{args.env}" in scene:
        lo, hi = scene[f"calvin_scene_{args.env}"]
    else:
        lo, hi = tr_eps[0, 0], tr_eps[-1, 1]
        print("WARNING: no scene_info; using the whole training split")

    windows = load_windows(tr / "lang_annotations" / "auto_lang_ann.npy")
    a_windows = windows[(windows[:, 0] >= lo) & (windows[:, 1] <= hi)]
    sel_train = a_windows[:: args.train_window_stride]
    train_steps = expand(sel_train)
    print(f"env {args.env}: {len(a_windows)} windows -> keep {len(sel_train)} "
          f"({len(train_steps)} steps, ~{len(train_steps) * 260e3 / 1e9:.1f} GB)")

    val_windows = load_windows(val / "lang_annotations" / "auto_lang_ann.npy")
    sel_val = val_windows[:: args.val_window_stride]
    id_test_steps = expand(sel_val)
    print(f"validation: {len(val_windows)} windows -> keep {len(sel_val)} "
          f"({len(id_test_steps)} steps, ~{len(id_test_steps) * 260e3 / 1e9:.2f} GB)")

    plan = {
        "env": args.env, "train_window_stride": args.train_window_stride,
        "val_window_stride": args.val_window_stride,
        "train_windows": sel_train.tolist(), "val_windows": sel_val.tolist(),
        "train_steps": train_steps, "id_test_steps": id_test_steps,
        "scene_ranges": {k: [int(v[0]), int(v[1])] for k, v in scene.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(plan, f)
    print(f"plan -> {out}")


if __name__ == "__main__":
    main()
