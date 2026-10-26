#!/usr/bin/env python3
"""scripts/make_grpo_eval_split.py — carve a GRPO eval pool out of sft_train.

Protocol fix: GRPO's periodic deterministic evaluation must NOT use id_test
instructions (that would leak the test set during training). Instead we hold out a
small pool from sft_train:
    calvin_grpo_eval = every Nth window of calvin_sft_train (stride 7 -> ~15 windows)
These windows are excluded from GRPO rollout sampling and used ONLY for the
periodic deterministic evaluation (checkpoint selection rule: best grpo_eval).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import assert_no_leak, load_manifest, write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calvin-root", default=str(PROJECT_ROOT / "data" / "calvin"))
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.calvin_root)
    splits_dir = root / "splits"

    sft_train = sorted(load_manifest(splits_dir / "calvin_sft_train.json")["ids"])
    grpo_eval = sft_train[args.seed_offset::args.stride]
    sft_train_new = [i for i in sft_train if i not in set(grpo_eval)]

    print(f"sft_train: {len(sft_train)} -> {len(sft_train_new)} (removed "
          f"{len(grpo_eval)} as grpo_eval)")

    # overwrite sft_train manifest WITHOUT grpo_eval (manifest write refuses to
    # overwrite existing files -> delete first, deliberate action)
    (splits_dir / "calvin_sft_train.json").unlink()
    write_manifest(splits_dir, "calvin_sft_train", sft_train_new)
    write_manifest(splits_dir, "calvin_grpo_eval", grpo_eval)

    id_test = set(load_manifest(splits_dir / "calvin_id_test.json")["ids"])
    sft_val = set(load_manifest(splits_dir / "calvin_sft_val.json")["ids"])
    assert_no_leak({"sft_train": set(sft_train_new), "sft_val": sft_val,
                    "id_test": id_test, "ood_test": set()})
    assert set(grpo_eval).isdisjoint(id_test), "LEAK: grpo_eval overlaps id_test"
    print("OK: grpo_eval disjoint from id_test; manifests updated")


if __name__ == "__main__":
    main()
