#!/usr/bin/env python3
"""scripts/preprocess_csi_tts.py — B3 audio: text instruction -> SYNTHESIZED speech.

For every CALVIN trajectory in the B3 pool, synthesize the instruction to a 16 kHz
mono wav (edge-tts -> espeak-ng fallback) and store it frame-aligned:
the wav is split into ceil(T/8) segments, one per CALVIN 8-frame window (CSI alignment
convention). NOTE: this is synthesized speech, not human natural speech.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.csi import SAMPLE_RATE, WINDOW, align_wav_to_windows, synthesize  # noqa: E402


def main(calvin_root: Path, csi_root: Path, split: str = "sft_train",
         overwrite: bool = False, limit: int = 0):
    calvin_root = Path(calvin_root)
    csi_root = Path(csi_root)
    with open(calvin_root / "splits" / f"calvin_{split}.json") as f:
        ids = json.load(f)["ids"]
    if limit:
        ids = ids[:limit]

    wav_dir = csi_root / "wav"
    seg_dir = csi_root / "wav_segments"
    wav_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for traj_id in ids:
        # instruction text lives in the preprocessed CALVIN trajectory npz
        with np.load(calvin_root / split / f"{traj_id}.npz") as f:
            lang = str(f["lang"])
            T = len(f["actions"])

        # wav keyed by sha256(instruction) — same convention as rollout & CSIDataset
        name = hashlib.sha256(lang.encode()).hexdigest()[:16]
        wav_path = wav_dir / f"{name}.wav"
        seg_path = seg_dir / f"{name}.npy"
        if wav_path.exists() and seg_path.exists() and not overwrite:
            ok += 1
            continue

        if not synthesize(lang, wav_path):
            print(f"[FAIL] TTS for {traj_id}: {lang}")
            fail += 1
            continue

        import soundfile as sf
        wav, sr = sf.read(str(wav_path), dtype="float32")
        assert sr == SAMPLE_RATE
        segs = align_wav_to_windows(wav, T, WINDOW)
        np.save(seg_path, segs, allow_pickle=True)
        ok += 1
        if ok % 50 == 0:
            print(f"  {ok}/{len(ids)} synthesized")

    print(f"==> CSI TTS done: {ok} ok, {fail} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calvin-root", default=str(PROJECT_ROOT / "data" / "calvin"))
    ap.add_argument("--csi-root", default=str(PROJECT_ROOT / "data" / "csi"))
    ap.add_argument("--split", default="sft_train")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    main(Path(args.calvin_root), Path(args.csi_root), args.split,
         args.overwrite, args.limit)
