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
    """Map logical split name -> CALVIN trajectory id list with leak assertions.

    NOTE: CALVIN-only by default. Bridge (if requested) is added separately by
    build_sft_loaders, never unioned into the CALVIN split manifests.
    """
    if split in ("sft_train", "sft_val"):
        ids = _load_manifest_set(root, f"calvin_{split}")
    elif split == "calvin_train":
        ids = _load_manifest_set(root, "calvin_sft_train")
    elif split == "calvin_sft_val":
        ids = _load_manifest_set(root, "calvin_sft_val")
    else:
        raise ValueError(f"unknown split {split}")
    assert len(ids) > 0, f"split {split} is empty — run preprocess_data.py first"
    return sorted(ids)


class CalvinDataset(Dataset):
    """Single trajectory per item: images (T,3,224,224), actions (T,7), lang string.

    With `chunk_size` set, yields consecutive chunks of steps (used for full-coverage
    validation); otherwise yields the whole trajectory and the collate samples a
    random chunk per epoch (training).
    """

    def __init__(self, root: Path, split: str, transform=None, chunk_size: int = None):
        self.root = Path(root)
        self.split = split
        self.ids = _resolve_split_ids(self.root, split)
        self.transform = transform
        self.chunk_size = chunk_size
        self._index = None
        if chunk_size and chunk_size > 0:
            self._index = []
            for i in range(len(self.ids)):
                with np.load(self._npz_path(i)) as f:
                    t = int(f["actions"].shape[0])
                for s in range(0, t, chunk_size):
                    self._index.append((i, s, min(s + chunk_size, t)))

        # Leakage guard on every load
        train = _load_manifest_set(self.root, "calvin_sft_train")
        val = _load_manifest_set(self.root, "calvin_sft_val")
        id_test = _load_manifest_set(self.root, "calvin_id_test")
        ood_test = _load_manifest_set(self.root, "calvin_ood_test")
        assert_no_leak({"sft_train": train, "sft_val": val,
                        "id_test": id_test, "ood_test": ood_test})

    def __len__(self):
        return len(self._index) if self._index is not None else len(self.ids)

    def _npz_path(self, idx: int) -> Path:
        p = self.root / self.split / f"{self.ids[idx]}.npz"
        if not p.exists() and self.split == "sft_val":
            # manifests split 85/15 but files physically live in sft_train/
            p = self.root / "sft_train" / f"{self.ids[idx]}.npz"
        return p

    def __getitem__(self, idx: int):
        if self._index is not None:
            i, s, e = self._index[idx]
            sl = slice(s, e)
        else:
            i, sl = idx, slice(0, None)
        with np.load(self._npz_path(i)) as f:
            rgb = f["rgb"][sl]            # (K,3,224,224) uint8
            actions = f["actions"][sl]    # (K,7) float32
            lang = str(f["lang"])
            proprio = f["proprio"][sl] if "proprio" in f else np.zeros(
                (actions.shape[0], 8), dtype=np.float32)
        if self.transform is not None:
            rgb = self.transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).float() / 255.0
        return {"pixel_values": rgb, "actions": torch.from_numpy(actions), "lang": lang,
                "id": self.ids[i], "proprio": torch.from_numpy(proprio)}


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

    Training: CALVIN trajectories are pre-chunked into `steps_per_traj`-long items
    so every frame is seen each epoch (full coverage) while per-forward memory stays
    at batch_size x steps_per_traj samples. Bridge is added only if "bridge" is in
    cfg.group.dataset.

    Validation: CALVIN trajectories are pre-chunked (val_chunk_size) so early
    stopping sees every frame, not just the first few.
    """
    from torch.utils.data import DataLoader

    steps_per_traj = cfg.sft.steps_per_traj
    val_chunk_size = getattr(cfg.sft, "val_chunk_size", 8)

    # training caps each item to a random chunk of `cap` steps; val items are already
    # chunks and are passed through untouched.
    cap = val_chunk_size if split == "sft_val" else steps_per_traj

    def step_collate(batch):
        # flatten (T, ...) trajectory/chunk items into individual steps
        import random as _r
        pixels, acts, langs, proprios = [], [], [], []
        for b in batch:
            T = b["pixel_values"].shape[0]
            if T > cap:
                if split == "sft_train":
                    start = _r.randint(0, T - cap)  # random window
                else:
                    start = 0
                sl = slice(start, start + cap)
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

    # combined loader: CALVIN (+ Bridge for training only if requested)
    if split == "sft_val":
        ds_list = [CalvinDataset(Path(cfg.dataset.calvin_root), split,
                                 chunk_size=val_chunk_size)]
    else:
        # full-coverage chunking: each trajectory is split into steps_per_traj-long
        # chunks; one chunk per batch => every frame is seen each epoch, while the
        # per-forward memory stays at batch_size x steps_per_traj samples. Bridge
        # (if requested) stays whole and is sliced by the collate.
        ds_list = [CalvinDataset(Path(cfg.dataset.calvin_root), split,
                                 chunk_size=steps_per_traj)]
        if "bridge" in cfg.group.dataset:
            ds_list.append(BridgeDataset(Path(cfg.dataset.bridge_root), split))
    ds = torch.utils.data.ConcatDataset(ds_list)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "sft_train"),
                      collate_fn=step_collate, num_workers=num_workers, pin_memory=True)
