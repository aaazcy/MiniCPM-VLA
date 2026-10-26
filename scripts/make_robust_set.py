#!/usr/bin/env python3
"""scripts/make_robust_set.py — build the robustness test set from CALVIN's official
paraphrased language annotations (lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy).

For every id_test trajectory, replace the original instruction with the official
paraphrase (same 64-step window), keeping the same episode/initial state. Outputs
<calvin_root>/splits/robust_instructions.json used by scripts/evaluate.py --split robustness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_lang(path: Path) -> dict:
    d = np.load(path, allow_pickle=True).item()
    return {"ann": [str(x) for x in d["language"]["ann"]],
            "indx": np.asarray(d["info"]["indx"], dtype=np.int64).reshape(-1, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calvin-root", default=str(PROJECT_ROOT / "data" / "calvin"))
    ap.add_argument("--paraphrase", required=True,
                    help="validation lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy")
    ap.add_argument("--original", required=True,
                    help="validation lang_annotations/auto_lang_ann.npy")
    args = ap.parse_args()

    root = Path(args.calvin_root)
    para = load_lang(Path(args.paraphrase))
    orig = load_lang(Path(args.original))

    # window (w0,w1) -> paraphrase text
    para_by_window = {tuple(indx): ann for ann, indx in zip(para["ann"], para["indx"])}
    # fallback: nearest window start
    para_starts = np.array([p[0] for p in para["indx"]])

    with open(root / "splits" / "calvin_id_test.json") as f:
        ids = json.load(f)["ids"]

    instructions, init_states, ids_out = [], [], []
    skipped = 0
    for tid in ids:
        # id format: id_test_<w0:07d>_<w1:07d> (64-step window range)
        parts = tid.rsplit("_", 2)
        if len(parts) != 3:
            skipped += 1
            continue
        w0, w1 = int(parts[1]), int(parts[2])
        text = para_by_window.get((w0, w1))
        if text is None:
            j = int(np.argmin(np.abs(para_starts - w0)))
            text = para["ann"][j]
        npz_path = root / "id_test" / f"{tid}.npz"
        if not npz_path.exists():
            skipped += 1
            continue
        with np.load(npz_path) as f_:
            init_states.append((f_["init_robot_obs"].astype(np.float32).tolist(),
                                f_["init_scene_obs"].astype(np.float32).tolist()))
        instructions.append(text)
        ids_out.append(tid)

    out = {"instructions": instructions, "init_states": init_states, "ids": ids_out}
    with open(root / "splits" / "robust_instructions.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"robust set: {len(instructions)} instructions ({skipped} skipped) -> "
          f"{root / 'splits' / 'robust_instructions.json'}")


if __name__ == "__main__":
    main()
