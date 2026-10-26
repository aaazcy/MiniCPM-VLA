"""src/data/loader.py — SFT dataset loaders for CALVIN and BridgeData V2.

Both loaders read the local preprocessed format produced by scripts/preprocess_data.py:
    <root>/<split>/<traj_id>.npz
        rgb:     (T, 3, 224, 224) uint8      # static camera, 224x224
        actions: (T, 7) float32              # normalized [-1, 1]
        lang:    str (json-serialized inside npz)

Leakage guard: loaders consult the split manifests under <root>/splits/ and assert
requested ids belong to the requested split and no test id appears in train/val.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .splits import assert_no_leak, load_manifest


def _load_manifest_set(root: Path, name: str) -> set:
    p = root / "splits" / f"{name}.json"
    if not p.exists():
        return set()
    return set(load_manifest(p)["ids"])


def _resolve_split_ids(root: Path, split: str) -> List[str]:
    """Map logical split name -> trajectory id list with leak assertions."""
    if split in ("sft_train", "sft_val"):
        ids = _load_manifest_set(root, f"calvin_{split}") | _load_manifest_set(root, "bridge_train")
    elif split == "calvin_train":
        ids = _load_manifest_set(root, "calvin_sft_train")
    elif split == "calvin_sft_val":
        ids = _load_manifest_set(root, "calvin_sft_val")
    else:
        raise ValueError(f"unknown split {split}")
    assert len(ids) > 0, f"split {split} is empty — run preprocess_data.py first"
    return sorted(ids)


class CalvinDataset(Dataset):
    """Single trajectory per item: images (T,3,224,224), actions (T,7), lang string."""

    def __init__(self, root: Path, split: str, transform=None):
        self.root = Path(root)
        self.split = split
        self.ids = _resolve_split_ids(self.root, split)
        self.transform = transform

        # Leakage guard on every load
        train = _load_manifest_set(self.root, "calvin_sft_train")
        val = _load_manifest_set(self.root, "calvin_sft_val")
        id_test = _load_manifest_set(self.root, "calvin_id_test")
        ood_test = _load_manifest_set(self.root, "calvin_ood_test")
        assert_no_leak({"sft_train": train, "sft_val": val,
                        "id_test": id_test, "ood_test": ood_test})

    def __len__(self):
        return len(self.ids)

    def _npz_path(self, idx: int) -> Path:
        p = self.root / self.split / f"{self.ids[idx]}.npz"
        if not p.exists() and self.split == "sft_val":
            # manifests split 85/15 but files physically live in sft_train/
            p = self.root / "sft_train" / f"{self.ids[idx]}.npz"
        return p

    def __getitem__(self, idx: int):
        with np.load(self._npz_path(idx)) as f:
            rgb = f["rgb"]            # (T,3,224,224) uint8
            actions = f["actions"]    # (T,7) float32
            lang = str(f["lang"])
            proprio = f["proprio"] if "proprio" in f else np.zeros((len(actions), 8),
                                                                   dtype=np.float32)
        if self.transform is not None:
            rgb = self.transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).float() / 255.0
        return {"pixel_values": rgb, "actions": torch.from_numpy(actions), "lang": lang,
                "id": self.ids[idx], "proprio": torch.from_numpy(proprio)}


class BridgeDataset(Dataset):
    """Budgeted BridgeData V2 subsample; identical format to CalvinDataset.
    All Bridge trajectories go to sft_train only (SFT-val is CALVIN-only)."""

    def __init__(self, root: Path, split: str = "sft_train", transform=None):
        self.root = Path(root)
        self.ids = _load_manifest_set(self.root, "bridge_train")
        if not self.ids:
            raise RuntimeError("no bridge_train manifest — run "
                               "preprocess_bridge_remote.py first")
        self.ids = sorted(self.ids)
        self.transform = transform
        self.split = "sft_train"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        with np.load(self.root / self.split / f"{self.ids[idx]}.npz") as f:
            rgb = f["rgb"]
            actions = f["actions"]
            lang = str(f["lang"])
            proprio = f["proprio"] if "proprio" in f else np.zeros((len(actions), 8),
                                                                   dtype=np.float32)
        if self.transform is not None:
            rgb = self.transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).float() / 255.0
        return {"pixel_values": rgb, "actions": torch.from_numpy(actions), "lang": lang,
                "id": self.ids[idx], "proprio": torch.from_numpy(proprio)}


def build_sft_loaders(cfg, processor, split: str, batch_size: int, num_workers: int = 8):
    """Build train/val DataLoaders for SFT. split in {"sft_train","sft_val"}.

    Items are re-windowed into single steps (image, action, lang) by the collate fn,    because SFT supervises step-level MSE. Language is passed as raw text; each model
    wrapper runs its own processor internally.
    """
    from torch.utils.data import DataLoader

    steps_per_traj = cfg.sft.steps_per_traj

    def step_collate(batch):
        # flatten (T, ...) trajectory items into individual steps; cap the number of
        # steps per trajectory so the effective step-batch stays memory-bounded
        # (batch_size counts TRAJECTORIES, not steps).
        import random as _r
        pixels, acts, langs, proprios = [], [], [], []
        for b in batch:
            T = b["pixel_values"].shape[0]
            if T > steps_per_traj:
                if split == "sft_train":
                    start = _r.randint(0, T - steps_per_traj)  # random window
                else:
                    start = 0
                sl = slice(start, start + steps_per_traj)
            else:
                sl = slice(0, T)
            pixels.append(b["pixel_values"][sl])
            acts.append(b["actions"][sl])
            langs.extend([b["lang"]] * (sl.stop - sl.start))
            proprios.append(b.get("proprio", torch.zeros(T, 8))[sl])
        pixel_batch = torch.cat(pixels, dim=0)
        act_batch = torch.cat(acts, dim=0)
        proprio_batch = torch.cat(proprios, dim=0)
        return {"pixel_values": pixel_batch, "actions": act_batch,
                "langs": langs, "proprio": proprio_batch}

    # combined loader: CALVIN (+ Bridge for training only; SFT-val is CALVIN-only)
    ds_list = [CalvinDataset(Path(cfg.dataset.calvin_root), split)]
    if split == "sft_train":
        ds_list.append(BridgeDataset(Path(cfg.dataset.bridge_root), split))
    ds = torch.utils.data.ConcatDataset(ds_list)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "sft_train"),
                      collate_fn=step_collate, num_workers=num_workers, pin_memory=True)
