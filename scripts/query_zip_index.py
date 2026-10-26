#!/usr/bin/env python3
"""scripts/query_zip_index.py — stream-filter the big zip index into a slim JSON.

The full index (1.9M entries, 219 MB JSON) cannot be loaded into memory under this
container's 2 GB cgroup limit. This script makes ONE streaming pass with ijson.parse
(one entry in memory at a time) and keeps only matching entries (metadata files +
episode files in a range/stride), writing a tiny index for selective_extract.py.

Index layout: {"url": ..., "zip_size": ..., "entries": {"<name>": {"offset":..,
               "csize":.., "usize":.., "method":..}, ...}}

Usage:
    python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
        --include 'task_ABC_D/training/ep_start_end_ids.npy' \
        --include 'task_ABC_D/training/lang_annotations/auto_lang_ann.npy' \
        --episode-dir 'task_ABC_D/training' --episode-range 1191339 1241339 \
        --stride 40 --out data/calvin/subset.index.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import ijson

META_FIELDS = {"offset", "csize", "usize", "method"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--include", action="append", default=[],
                    help="exact entry name(s) to keep (repeatable)")
    ap.add_argument("--episode-dir", default=None,
                    help="keep episode_*.npz under this zip dir (e.g. task_ABC_D/training)")
    ap.add_argument("--episode-range", nargs=2, type=int, default=None,
                    help="episode id range [lo hi] inclusive")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth matching episode")
    ap.add_argument("--episode-list", default=None,
                    help="JSON with lists of step ids ('train_steps'/'id_test_steps') "
                         "or a plain list; keep episode_<id>.npz for those ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    include = set(args.include)
    step_ids = set()
    if args.episode_list:
        data = json.load(open(args.episode_list))
        if isinstance(data, list):
            step_ids = {int(x) for x in data}
        else:
            for key in ("train_steps", "id_test_steps"):
                step_ids |= {int(x) for x in data.get(key, [])}
    ep_re = re.compile(r"^episode_(\d+)\.npz$")
    kept = {}
    n_meta = n_ep = 0

    cur_name, cur_meta = None, {}

    def finalize():
        nonlocal cur_name, cur_meta, n_meta, n_ep
        if not cur_name:
            return
        name, meta = cur_name, cur_meta
        cur_name, cur_meta = None, {}
        if name in include:
            kept[name] = dict(meta)
            n_meta += 1
        elif step_ids or (args.episode_dir and args.episode_range):
            m = ep_re.match(name.rsplit("/", 1)[-1])
            if m:
                ep = int(m.group(1))
                if step_ids:
                    if ep in step_ids:
                        kept[name] = dict(meta)
                        n_ep += 1
                elif args.episode_dir and name.startswith(args.episode_dir + "/") \
                        and args.episode_range[0] <= ep <= args.episode_range[1]:
                    n_ep += 1
                    if n_ep % args.stride == 1:
                        kept[name] = dict(meta)

    # ijson event layout for {"entries": {"<name>": {"offset":..,"csize":..,...}}}:
    #   ("entries", "map_key")              -> entry name
    #   ("entries.<name>", "start_map")     -> meta map begins
    #   ("entries.<name>.<field>", "number")-> meta value (field = last prefix segment)
    #   ("entries", "end_map")              -> entries container ends
    with open(args.index, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "entries" and event == "map_key":
                finalize()
                cur_name = value
                cur_meta = {}
            elif prefix == "entries" and event == "end_map":
                finalize()
            elif event == "number" and prefix.startswith("entries.") and cur_name:
                field = prefix.rsplit(".", 1)[-1]
                cur_meta[field] = value

    print(f"kept {len(kept)} entries ({n_meta} metadata, "
          f"{n_ep} episodes in range; stride {args.stride})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"url": "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip",
                   "entries": kept}, f)
    print(f"slim index -> {out} ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
