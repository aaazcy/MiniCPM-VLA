#!/usr/bin/env python3
"""One-time migration: add `task` (CALVIN subtask name) to existing trajectory npz.

For each <root>/<split>/<traj_id>.npz, parse [w0, w1] from the traj_id (format
`{prefix}_{w0:07d}_{w1:07d}`), look up the subtask name from the language annotation,
and re-save the npz with the `task` field. Needed for CALVIN success detection.

Usage:
    python scripts/add_task_field.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_TRAIN = "data/calvin/meta/task_ABC_D/training/lang_annotations/auto_lang_ann.npy"
DEFAULT_VAL = "data/calvin/meta/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy"


def load_task_map(lang_path: Path) -> dict:
    d = np.load(lang_path, allow_pickle=True).item()
    indx = np.asarray(d["info"]["indx"], dtype=np.int64).reshape(-1, 2)
    task = [str(x) for x in d["language"]["task"]]
    return {tuple(i): t for i, t in zip(indx, task)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/calvin")
    ap.add_argument("--lang-train", default=DEFAULT_TRAIN)
    ap.add_argument("--lang-val", default=DEFAULT_VAL)
    args = ap.parse_args()

    root = Path(args.root)
    task_map = {**load_task_map(Path(args.lang_train)),
                **load_task_map(Path(args.lang_val))}

    for split in ["sft_train", "sft_val", "id_test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        files = sorted(f for f in split_dir.glob("*.npz") if ".tmp" not in f.name)
        n_updated = n_skip = n_miss = 0
        for f in files:
            parts = f.stem.split("_")            # {prefix}_{w0}_{w1}
            w0, w1 = int(parts[-2]), int(parts[-1])
            task = task_map.get((w0, w1))
            if task is None:
                print(f"  [warn] {f.name}: no task for window ({w0},{w1})")
                n_miss += 1
                continue
            d = dict(np.load(f, allow_pickle=True))
            if "task" in d and str(d["task"]) == task:
                n_skip += 1
                continue
            d["task"] = task
            # atomic write (temp + rename) so an interrupted run can't corrupt a file.
            # numpy appends ".npz" unless the name already ends in it, so name the temp
            # file `*.tmp.npz`.
            tmp = f.with_name(f.stem + ".tmp.npz")
            np.savez_compressed(tmp, **d)
            tmp.replace(f)
            n_updated += 1
        print(f"==> {split}: {len(files)} files, updated={n_updated}, "
              f"already-ok={n_skip}, missing={n_miss}")


if __name__ == "__main__":
    main()
