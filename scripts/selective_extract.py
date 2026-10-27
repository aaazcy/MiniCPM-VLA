#!/usr/bin/env python3
"""scripts/selective_extract.py — fetch specific files from a REMOTE zip via range requests.

Uses the index built by fetch_zip_index.py to download individual entries of
task_ABC_D.zip (517 GB) without storing the zip locally. Works with ZIP64 archives.

Usage examples:
    # metadata files needed for episode grouping / language annotations
    python scripts/selective_extract.py --index data/calvin/task_ABC_D.index.json \
        --pattern 'task_ABC_D/training/ep_start_end_ids.npy' \
        --pattern 'task_ABC_D/training/lang_annotations/auto_lang_ann.npy' \
        --out-dir data/calvin/partial

    # subsample N training episodes (step files), e.g. env-A episodes by id range
    python scripts/selective_extract.py --index data/calvin/task_ABC_D.index.json \
        --pattern 'task_ABC_D/training/episode_*.npz' --episode-range 0 2000 \
        --stride 5 --out-dir data/calvin/partial   # every 5th episode
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import struct
import urllib.request
from pathlib import Path


def unpack_exact(fmt: str, buf: bytes):
    return struct.unpack(fmt, buf[:struct.calcsize(fmt)])


def http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def select_entries(index: dict, patterns, episode_range, stride) -> dict:
    ep_re = re.compile(r"episode_(\d+)\.npz$")
    out = {}
    for name, meta in index.items():
        if not name.endswith(".npz"):
            # metadata files are matched by exact pattern
            if any(name == p for p in patterns):
                out[name] = meta
            continue
        if not any(re.match(p.replace("\\*", ".*"), name) for p in patterns):
            continue
        m = ep_re.search(name)
        if m and episode_range is not None:
            ep = int(m.group(1))
            if ep < episode_range[0] or ep > episode_range[1]:
                continue
        out[name] = meta
    if stride and stride > 1:
        keep = {}
        for i, (name, meta) in enumerate(sorted(out.items())):
            if i % stride == 0:
                keep[name] = meta
        out = keep
    return out


def fetch_one(url: str, name: str, meta: dict, out_dir: Path, retries: int = 6) -> Path:
    import time
    import zlib
    dst = out_dir / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected = meta.get("usize", meta["csize"])
    if dst.exists() and dst.stat().st_size == expected:
        return dst  # already downloaded

    loffset = meta["offset"]
    csize = meta["csize"]
    last_err = None
    for attempt in range(retries):
        try:
            # ONE HTTP Range request covering the local header + compressed data.
            # The header is 30 + nlen + elen bytes; CALVIN filenames are short so
            # 512 bytes of headroom is more than enough. Halves the round-trips vs
            # the previous header-then-data two-request flow.
            head_guess = 512
            buf = http_range(url, loffset, loffset + 30 + head_guess + csize - 1)
            if unpack_exact("<I", buf[0:4])[0] != 0x04034B50:
                raise RuntimeError(f"bad local header for {name}")
            nlen, elen = unpack_exact("<HH", buf[26:30])
            header_len = 30 + nlen + elen
            if header_len > 30 + head_guess:
                raise RuntimeError(f"header too long for {name}: {header_len}")
            data = buf[header_len:header_len + csize]
            if len(data) != csize:
                raise RuntimeError(f"short read {len(data)}/{csize}")
            if meta.get("method") == 8:      # deflate-compressed in the outer zip
                data = zlib.decompress(data, -15)  # raw deflate
            if len(data) != expected:
                raise RuntimeError(f"decompressed size {len(data)} != {expected}")
            dst.write_bytes(data)
            return dst
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{name}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--pattern", action="append", default=[],
                    help="file pattern(s); '\\*' acts as wildcard. Repeatable.")
    ap.add_argument("--episode-range", nargs=2, type=int, default=None,
                    help="episode id range [lo hi] (inclusive) for episode_*.npz")
    ap.add_argument("--stride", type=int, default=0,
                    help="keep every Nth matching episode (0 = keep all)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--all", action="store_true",
                    help="download every entry in the (slim) index")
    args = ap.parse_args()

    with open(args.index) as f:
        data = json.load(f)
    url = data.get("url") or "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
    index = data["entries"]

    if args.all:
        selected = index
    else:
        selected = select_entries(index, args.pattern or ["\\*"], args.episode_range, args.stride)
    print(f"selected {len(selected)} files; downloading with {args.workers} workers ...")

    out_dir = Path(args.out_dir)
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, url, n, m, out_dir): n for n, m in selected.items()}
        for i, fut in enumerate(cf.as_completed(futs)):
            name = futs[fut]
            try:
                dst = fut.result()
                results.append(dst)
            except Exception as e:
                print(f"  FAIL {name}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(selected)} done")
    total = sum(p.stat().st_size for p in results)
    print(f"done: {len(results)} files, {total/1e9:.2f} GB -> {out_dir}")


if __name__ == "__main__":
    main()
