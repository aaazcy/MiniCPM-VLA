"""src/data/splits.py — experiment guard: train / val / test must never leak.

Split policy (locked):
  * calvin_A training trajectories -> 85% SFT-train / 15% SFT-val (seed 42)
      - SFT-val is used ONLY for SFT early stopping, NEVER in RL or eval.
  * calvin_A official validation split (calvin/validation) -> ID test (unseen in training)
  * calvin_D -> OOD test (unseen in training)
  * robustness set (paraphrased / reordered / long instructions) -> unseen, provided as a
    separate manifest with the SAME trajectory ids as ID test.
Every manifest is written once with sha256 hashes; loaders assert no overlap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

SPLIT_NAMES = ["sft_train", "sft_val", "id_test", "ood_test"]


def _hash_ids(ids: List[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]


def write_manifest(root: Path, name: str, ids: List[str]) -> Path:
    """Write a split manifest and its hash. Refuses to overwrite an existing manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"split manifest {path} already exists; delete it only if you "
                              "intend to regenerate ALL splits")
    payload = {"name": name, "ids": sorted(ids), "sha256": _hash_ids(ids)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_manifest(path: Path) -> Dict[str, object]:
    with open(path) as f:
        data = json.load(f)
    assert _hash_ids(data["ids"]) == data["sha256"], f"manifest {path} corrupted (hash mismatch)"
    return data


def assert_no_leak(split_sets: Dict[str, Set[str]]) -> None:
    """Guarantee: test sets are disjoint from train/val; val disjoint from train."""
    assert SPLIT_NAMES[0] in split_sets and SPLIT_NAMES[1] in split_sets
    train, val = split_sets[SPLIT_NAMES[0]], split_sets[SPLIT_NAMES[1]]
    assert train.isdisjoint(val), "LEAK: SFT train ∩ SFT val is non-empty"
    for test_name in ("id_test", "ood_test"):
        if test_name in split_sets:
            t = split_sets[test_name]
            assert t.isdisjoint(train), f"LEAK: {test_name} ∩ sft_train non-empty"
            assert t.isdisjoint(val), f"LEAK: {test_name} ∩ sft_val non-empty"
    if "id_test" in split_sets and "ood_test" in split_sets:
        assert split_sets["id_test"].isdisjoint(split_sets["ood_test"]), \
            "LEAK: id_test ∩ ood_test non-empty"


def make_sft_splits(all_ids: List[str], train_ratio: float, seed: int) -> Dict[str, List[str]]:
    """85/15 split of CALVIN training trajectories (deterministic)."""
    import random
    rng = random.Random(seed)
    ids = sorted(all_ids)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return {"sft_train": ids[:n_train], "sft_val": ids[n_train:]}


def write_all_manifests(root: Path, sft_train: List[str], sft_val: List[str],
                        id_test: List[str], ood_test: List[str],
                        bridge_train: List[str]) -> List[Path]:
    paths = []
    for name, ids in [("calvin_sft_train", sft_train), ("calvin_sft_val", sft_val),
                      ("calvin_id_test", id_test), ("calvin_ood_test", ood_test),
                      ("bridge_train", bridge_train)]:
        paths.append(write_manifest(root, name, ids))
    return paths
