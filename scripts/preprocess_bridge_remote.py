#!/usr/bin/env python3
"""scripts/preprocess_bridge_remote.py — BridgeData V2 (LeRobot mirror) -> local npz.

Source: IPEC-COMMUNITY/bridge_orig_lerobot on HF (downloaded via hf-mirror.com since
huggingface.co is unreachable from this host):
    meta/episodes.jsonl   {"episode_index": i, "tasks": [lang], "length": T}
    meta/tasks.jsonl      {"task_index": i, "task": lang}
    data/chunk-XXX/episode_XXXXXX.parquet   per-episode metadata (action, state, ...)
    videos/chunk-XXX/observation.images.image_0/episode_XXXXXX.mp4

Output (identical trajectory format to CALVIN):
    <root>/sft_train/bridge_<ep>.npz
        rgb (T,3,224,224) uint8 | actions (T,7) [-1,1] | proprio (T,8) | lang

Bridge 7-D action normalization (documented in limitations): pos deltas / 0.5,
rot deltas / pi (clip), gripper {0,1} -> {-1,1}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import write_manifest  # noqa: E402

BASE = "https://hf-mirror.com/datasets/IPEC-COMMUNITY/bridge_orig_lerobot/resolve/main"
IMG_SIZE = 224


def http_get(url: str, dst: Path, retries: int = 5) -> bool:
    import time
    import urllib.request
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (vla-omni-bench)"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as f:
                f.write(r.read())
            if dst.stat().st_size > 0:
                return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  FAIL {url}: {e}")
                return False
            time.sleep(2 * (attempt + 1))
    return False


def normalize_action(a: np.ndarray) -> np.ndarray:
    out = a.copy().astype(np.float32)
    out[:3] = np.clip(out[:3] / 0.5, -1, 1)
    out[3:6] = np.clip(out[3:6] / np.pi, -1, 1)
    out[6] = 2.0 * out[6] - 1.0   # {0,1} -> {-1,1}
    return out


def decode_video(path: Path, expect_frames: int) -> np.ndarray:
    """Decode mp4 -> (T, 3, 224, 224) uint8, resizing on the fly (low memory)."""
    import av
    from PIL import Image
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")          # (H, W, 3)
            small = np.asarray(Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE)))
            out.append(small)
            if len(out) >= expect_frames:
                break
    return np.stack(out[:expect_frames]).transpose(0, 3, 1, 2)


def process_episode(ep: dict, out_dir: Path) -> bool:
    import pyarrow.parquet as pq
    ep_idx = ep["episode_index"]
    chunk = f"chunk-{ep_idx // 1000:03d}"
    name = f"episode_{ep_idx:06d}"

    parquet_url = f"{BASE}/data/{chunk}/{name}.parquet"
    video_url = f"{BASE}/videos/{chunk}/observation.images.image_0/{name}.mp4"
    pq_path = out_dir / f"{name}.parquet"
    mp4_path = out_dir / f"{name}.mp4"

    if not http_get(parquet_url, pq_path) or not http_get(video_url, mp4_path):
        return False
    try:
        tbl = pq.read_table(pq_path)
        actions = np.stack(tbl.column("action").to_pylist())
        T = len(actions)
        rgb = decode_video(mp4_path, T)
        if len(rgb) < T:
            T = len(rgb)
            actions = actions[:T]
        actions_n = np.stack([normalize_action(a) for a in actions])
        lang = ep["tasks"][0] if ep["tasks"] else ""
        np.savez_compressed(
            out_dir / f"bridge_{ep_idx:06d}.npz",
            rgb=rgb.astype(np.uint8), actions=actions_n,
            proprio=np.zeros((T, 8), dtype=np.float32), lang=lang)
        return True
    except Exception as e:
        print(f"  skip episode {ep_idx}: {e}")
        return False
    finally:
        pq_path.unlink(missing_ok=True)
        mp4_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "data" / "bridge"))
    ap.add_argument("--max-trajectories", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    tmp = root / "tmp"
    out_dir = root / "sft_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # episode metadata
    ep_jsonl = root / "episodes.jsonl"
    if not ep_jsonl.exists():
        assert http_get(f"{BASE}/meta/episodes.jsonl?download=true", ep_jsonl), \
            "failed to download episodes.jsonl"
    episodes = [json.loads(line) for line in open(ep_jsonl)]
    print(f"total episodes: {len(episodes)}")

    # deterministic subsample (same subset for every group)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(episodes), size=args.max_trajectories, replace=False).tolist()
    sel = [episodes[i] for i in idx]
    print(f"selected {len(sel)} episodes")

    from concurrent.futures import ThreadPoolExecutor
    ids = [p.stem for p in out_dir.glob("bridge_*.npz")]  # resume: already done
    print(f"resume: {len(ids)} episodes already processed")
    todo = []
    for ep in sel:
        if (out_dir / f"bridge_{ep['episode_index']:06d}.npz").exists():
            continue
        todo.append(ep)
    print(f"to process: {len(todo)}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_episode, ep, tmp): ep["episode_index"]
                for ep in todo}
        import concurrent.futures as cf
        for fut in cf.as_completed(futs):
            ep_idx = futs[fut]
            try:
                if fut.result():
                    # move npz from tmp to final dir
                    src = tmp / f"bridge_{ep_idx:06d}.npz"
                    dst = out_dir / f"bridge_{ep_idx:06d}.npz"
                    src.rename(dst)
                    ids.append(dst.stem)
            except Exception as e:
                print(f"  episode {ep_idx} failed: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}")

    (root / "splits").mkdir(parents=True, exist_ok=True)
    write_manifest(root / "splits", "bridge_train", ids)
    print(f"==> Bridge done: {len(ids)}/{len(sel)} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
