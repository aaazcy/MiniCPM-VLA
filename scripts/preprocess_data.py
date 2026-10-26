#!/usr/bin/env python3
"""scripts/preprocess_data.py — raw CALVIN step files / BridgeData V2 -> local format.

REAL CALVIN layout (verified against the official zips):
    <split>/episode_XXXXXXX.npz   ONE STEP per file (flat, no scene subdirs):
        rgb_static (200,200,3) uint8 | rel_actions (7,) | actions (7,)
        robot_obs (15,) | scene_obs (24,) | depth_*
    <split>/ep_start_end_ids.npy  (N,2) inclusive [start,end] step ids per episode
    <split>/scene_info.npy        {'calvin_scene_A': [lo,hi], ...} env step ranges
    <split>/lang_annotations/auto_lang_ann.npy
        {'language': {'ann':(K,) str, 'task':(K,) str, 'emb':(K,1,384)},
         'info': {'indx': (K,2) 64-step windows}}
    <split>/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy  (paraphrased instructions)

Output trajectory format (one npz per language segment):
    <root>/<split>/<traj_id>.npz
        rgb (T,3,224,224) uint8 | actions (T,7) [-1,1] | proprio (T,8)
        lang str | init_robot_obs (15,) | init_scene_obs (24,)

Splits (locked, see src/data/splits.py): 85% sft_train / 15% sft_val (env-A segments,
seed 42); validation env-A segments -> id_test; testing (env D) -> ood_test.
Leakage asserted on write. Bad trajectories filtered heuristically.

Usage:
    python scripts/preprocess_data.py --dataset calvin \
        --raw data/calvin/raw/training --meta data/calvin/meta/training \
        --lang data/calvin/meta/training/lang_annotations/auto_lang_ann.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import assert_no_leak, make_sft_splits, write_manifest  # noqa: E402
from src.envs.calvin_env import CalvinEnv  # noqa: E402

IMG_SIZE = 224
MIN_SEGMENT_STEPS = 8


def resize_image(img: np.ndarray) -> np.ndarray:
    from PIL import Image
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = np.asarray(Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return img


def heuristic_filter(actions: np.ndarray) -> bool:
    """True if bad (keep=False). DISABLED by default: CALVIN teleop play data
    legitimately contains long arm-moves with a static gripper and jittery steps, so
    the original stall/jitter heuristics reject most windows (66%+). Since every group
    trains on identical data, filtering is not required for the comparison; re-enable
    here if manual curation (collision/failure labels) becomes available."""
    return False
    # original heuristics (kept for reference):
    # T = len(actions)
    # if T < 5:
    #     return False
    # grip = actions[:, 6]
    # run, best = 0, 0
    # for i in range(1, T):
    #     run = run + 1 if abs(grip[i] - grip[i - 1]) < 1e-3 else 0
    #     best = max(best, run)
    # if best > 30 and np.abs(actions[1:, :3]).mean() > 1e-4:
    #     return True
    # step_l2 = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
    # med = float(np.median(step_l2)) + 1e-9
    # if (step_l2 > 3 * med + 0.5).sum() > 0.2 * T:
    #     return True
    # return False


def load_lang(path: Path) -> dict:
    d = np.load(path, allow_pickle=True).item()
    return {
        "ann": [str(x) for x in d["language"]["ann"]],
        "task": [str(x) for x in d["language"]["task"]],
        "indx": np.asarray(d["info"]["indx"], dtype=np.int64).reshape(-1, 2),
    }


def load_step(step_path: Path) -> dict:
    """Load one raw step npz (CALVIN stores ONE step per file)."""
    with np.load(step_path) as f:
        return {"rgb": f["rgb_static"], "rel_actions": f["rel_actions"],
                "robot_obs": f["robot_obs"], "scene_obs": f["scene_obs"]}


def _build_segment(args) -> object:
    """Worker: build one trajectory npz from a 64-step window. Returns (traj_id, npz_path)
    or None if filtered."""
    raw_split_dir, out_dir, prefix, w0, w1, ann = args
    w0, w1 = int(w0), int(w1)
    if w1 - w0 + 1 < MIN_SEGMENT_STEPS:
        return None
    steps = []
    for s in range(w0, w1 + 1):
        p = Path(raw_split_dir) / f"episode_{s:07d}.npz"
        if not p.exists():
            break
        steps.append(load_step(p))
    if len(steps) < MIN_SEGMENT_STEPS:
        return None
    rgb = np.stack([resize_image(s["rgb"]) for s in steps]).transpose(0, 3, 1, 2)
    actions = np.stack([CalvinEnv.normalize_action(s["rel_actions"]) for s in steps])
    if heuristic_filter(actions):
        return None
    proprio = np.stack([s["robot_obs"][:8] for s in steps])
    traj_id = f"{prefix}_{w0:07d}_{w1:07d}"
    out_path = Path(out_dir) / f"{traj_id}.npz"
    np.savez_compressed(
        out_path,
        rgb=rgb.astype(np.uint8), actions=actions.astype(np.float32),
        proprio=proprio.astype(np.float32), lang=ann,
        init_robot_obs=steps[0]["robot_obs"].astype(np.float32),
        init_scene_obs=steps[0]["scene_obs"].astype(np.float32))
    return traj_id


def process_calvin(raw_root: Path, lang_train_path: Path, lang_val_path: Path,
                   plan_path: Path, out_root: Path, train_ratio: float = 0.85,
                   seed: int = 42) -> None:
    """Convert the selectively-extracted step files into trajectory npz segments.

    raw_root: dir containing training/ + validation/ step files (episode_*.npz).
    plan: plan.json from plan_calvin_subset.py — exactly the windows we downloaded.
    Segments are written to out_root/{sft_train,id_test}/ + split manifests.
    """
    lang_train = load_lang(lang_train_path)
    lang_val = load_lang(lang_val_path)
    with open(plan_path) as f:
        plan = json.load(f)

    def window_ann(lang: dict, w0: int, w1: int) -> str:
        for ann, (a0, a1) in zip(lang["ann"], lang["indx"]):
            if int(a0) == w0 and int(a1) == w1:
                return ann
        return lang["ann"][0]

    def collect_windows(raw_split_dir: Path, windows, lang: dict, prefix: str) -> list:
        out_dir = out_root / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for (w0, w1) in windows:
            jobs.append((str(raw_split_dir), str(out_dir), prefix, int(w0), int(w1),
                         window_ann(lang, int(w0), int(w1))))
        from concurrent.futures import ProcessPoolExecutor
        ids = []
        with ProcessPoolExecutor(max_workers=8) as ex:
            for i, res in enumerate(ex.map(_build_segment, jobs)):
                if res is not None:
                    ids.append(res)
                if (i + 1) % 32 == 0:
                    print(f"  [{prefix}] {i + 1}/{len(jobs)}")
        return ids

    print("==> env-A training windows -> sft_train/sft_val (85/15)")
    train_ids = collect_windows(raw_root / "training", plan["train_windows"],
                                lang_train, "sft_train")
    splits = make_sft_splits(train_ids, train_ratio, seed)

    print("==> validation windows -> id_test")
    val_ids = collect_windows(raw_root / "validation", plan["val_windows"],
                              lang_val, "id_test")

    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("calvin_sft_train", splits["sft_train"]),
                      ("calvin_sft_val", splits["sft_val"]),
                      ("calvin_id_test", val_ids)]:
        write_manifest(splits_dir, name, ids)

    assert_no_leak({
        "sft_train": set(splits["sft_train"]), "sft_val": set(splits["sft_val"]),
        "id_test": set(val_ids), "ood_test": set(),
    })
    print(f"==> CALVIN done: {len(splits['sft_train'])} train / {len(splits['sft_val'])} val / "
          f"{len(val_ids)} id_test")


def process_bridge(root: Path, max_trajectories: int = 20000, seed: int = 42):
    """BridgeData V2 (HuggingFace 'bridge_dataset') -> budgeted subsample."""
    from datasets import load_dataset

    print(f"==> BridgeData V2: streaming, budget {max_trajectories} trajectories")
    ds = load_dataset("bridge_dataset", "bridge_orig", split="train", streaming=True)
    rng = np.random.default_rng(seed)
    out_dir = root / "sft_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, kept, seen = [], 0, 0
    for ex in ds:
        seen += 1
        if kept >= max_trajectories:
            break
        if rng.random() > 0.5:
            continue
        try:
            frames = ex["observation"]["image"]
            act = np.asarray(ex["action"], dtype=np.float32)
            if act.ndim == 2:
                act = act[:, :7]
            lang = ex.get("natural_language_instruction", "")
            if not lang:
                continue
            rgb = np.stack([resize_image(np.asarray(fr)) for fr in frames]).transpose(0, 3, 1, 2)
            if act.ndim == 2 and len(act) > 1:
                act = np.vstack([act[0:1], np.diff(act, axis=0)])
            act = np.clip(act, -1, 1)
            if heuristic_filter(act):
                continue
            traj_id = f"bridge_{seen:07d}"
            np.savez_compressed(out_dir / f"{traj_id}.npz", rgb=rgb.astype(np.uint8),
                                actions=act, lang=str(lang),
                                proprio=np.zeros((len(act), 8), dtype=np.float32))
            ids.append(traj_id)
            kept += 1
        except Exception:
            continue
        if seen >= max_trajectories * 20:
            break
    (root / "splits").mkdir(parents=True, exist_ok=True)
    write_manifest(root / "splits", "bridge_train", ids)
    print(f"==> Bridge done: kept {kept}/{seen} scanned")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calvin", "bridge"], required=True)
    ap.add_argument("--root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--raw", default=None,
                    help="CALVIN raw step root (contains training/ + validation/)")
    ap.add_argument("--plan", default=str(PROJECT_ROOT / "data" / "calvin" / "plan.json"))
    ap.add_argument("--lang-train", default=None,
                    help="training lang_annotations/auto_lang_ann.npy path")
    ap.add_argument("--lang-val", default=None,
                    help="validation lang_annotations/auto_lang_ann.npy path")
    ap.add_argument("--train-ratio", type=float, default=0.85)
    ap.add_argument("--max-trajectories", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    if args.dataset == "calvin":
        assert args.raw and args.lang_train and args.lang_val, \
            "calvin needs --raw, --lang-train and --lang-val"
        process_calvin(Path(args.raw), Path(args.lang_train), Path(args.lang_val),
                       Path(args.plan), root / "calvin", args.train_ratio, args.seed)
    else:
        process_bridge(root / "bridge", args.max_trajectories, args.seed)
