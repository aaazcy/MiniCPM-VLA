#!/usr/bin/env python3
"""scripts/extend_calvin_data.py — expand env-A TRAINING windows (smaller stride),
download the missing raw steps via the full zip index, then re-run preprocessing
and split carving. val_windows / id_test stay LOCKED (eval-set consistency).

Pipeline:
    1. lang_annotations (auto_lang_ann.npy) for training+validation from the full index
    2. select every Nth env-A training window -> new plan.json (val_windows unchanged)
    3. download missing episode_*.npz steps (threaded, resume-safe)
    4. preprocess_data.py (rebuild sft_train/sft_val/id_test + leak-guarded manifests)
    5. make_grpo_eval_split.py (carve the GRPO eval pool from the new sft_train)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts.selective_extract import fetch_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    args = ap.parse_args()

    data = PROJECT_ROOT / "data" / "calvin"
    full_index_path = data / "task_ABC_D.index.json"
    if not full_index_path.exists():
        raise SystemExit("full index missing — run fetch_zip_index.py first")
    idx = json.load(open(full_index_path))
    entries, url = idx["entries"], idx["url"]

    # 1. lang annotations (windows + instruction texts) — original layout nests
    # them under meta/task_ABC_D/ (downloaded during initial setup)
    lang_train = data / "meta" / "task_ABC_D" / "training" / "lang_annotations" / "auto_lang_ann.npy"
    lang_val = data / "meta" / "task_ABC_D" / "validation" / "lang_annotations" / "auto_lang_ann.npy"
    for p in (lang_train, lang_val):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    lt = np.load(lang_train, allow_pickle=True).item()
    scene = np.load(data / "meta" / "training" / "scene_info.npy",
                    allow_pickle=True).item()
    lo, hi = scene["calvin_scene_A"]
    wins = [(int(a), int(b)) for a, b in lt["info"]["indx"]
            if int(a) >= lo and int(b) <= hi]

    old_plan = json.load(open(data / "plan.json"))
    # val_windows LOCKED: the id_test / ood eval set must not change
    train = [w for i, w in enumerate(wins) if i % args.stride == 0]
    plan = {"env": "A", "train_window_stride": args.stride,
            "val_window_stride": old_plan.get("val_window_stride", 32),
            "train_windows": train, "val_windows": old_plan["val_windows"]}
    with open(data / "plan.json", "w") as f:
        json.dump(plan, f, indent=1)
    print(f"windows: train {len(old_plan['train_windows'])} -> {len(train)} "
          f"(stride {args.stride}); val locked {len(old_plan['val_windows'])}",
          flush=True)

    # 2. download missing raw steps (train split only; val steps already local)
    raw_root = data / "raw"           # fetch_one nests under out_dir/name
    needed, have = 0, 0
    missing = {}
    for w0, w1 in train:
        for s in range(w0, w1 + 1):
            name = f"task_ABC_D/training/episode_{s:07d}.npz"
            needed += 1
            if (raw_root / name).exists():
                have += 1
            else:
                missing[name] = entries[name]
    print(f"steps needed {needed}, present {have}, to download {len(missing)}",
          flush=True)
    if missing and not args.skip_download:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_one, url, n, m, raw_root): n
                    for n, m in missing.items()}
            for fut in as_completed(futs):
                done += 1
                if done % 2000 == 0:
                    print(f"  {done}/{len(missing)}", flush=True)
        print("download done", flush=True)

    # 3. preprocess + carve GRPO eval pool
    if not args.skip_preprocess:
        py = "/root/miniconda3/bin/python"
        cmd = [py, str(PROJECT_ROOT / "scripts" / "preprocess_data.py"),
               "--dataset", "calvin",
               "--raw", str(data / "raw" / "task_ABC_D"),
               "--lang-train", str(lang_train), "--lang-val", str(lang_val),
               "--plan", str(data / "plan.json")]
        print("running preprocess:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        subprocess.run([py, str(PROJECT_ROOT / "scripts" / "make_grpo_eval_split.py")],
                       check=True)
    print("EXTEND DONE", flush=True)


if __name__ == "__main__":
    main()
