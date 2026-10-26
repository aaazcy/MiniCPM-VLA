#!/usr/bin/env python3
"""scripts/fetch_zip_index.py — build a local index of a REMOTE ZIP64 file's entries.

The CALVIN task_ABC_D.zip is 517 GB; we cannot store it locally. This script downloads
only the central directory via HTTP range requests and produces a JSON index:
    {name: {"offset": local_file_header_offset, "csize": compressed_size,
            "usize": uncompressed_size, "method": compression_method}}
so that scripts/selective_extract.py can fetch individual files (e.g. a subsample of
episode npz files, ep_start_end_ids.npy, lang_annotations) without downloading the zip.

Usage:
    python scripts/fetch_zip_index.py --url http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip \
        --out data/calvin/task_ABC_D.index.json
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

# NOTE: this container's struct.unpack rejects buffers LONGER than the format size.
def unpack_exact(fmt: str, buf: bytes):
    return struct.unpack(fmt, buf[:struct.calcsize(fmt)])


def http_range(url: str, start: int, end: int, retries: int = 4) -> bytes:
    import time
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  [retry {attempt + 1}] range {start}-{end}: {e}")
            time.sleep(2 * (attempt + 1))


def http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def parse_eocd_tail(url: str, size: int):
    """Fetch the last 128 KiB, parse (ZIP64) EOCD records -> (cd_offset, cd_size, n_entries)."""
    tail = http_range(url, max(0, size - 131072), size - 1)
    i = tail.rfind(b"\x50\x4b\x05\x06")
    assert i >= 0, "EOCD not found"
    eocd = tail[i:i + 22]
    (_sig, _disk, _cd_disk, n_disk, n_total, cd_size, cd_offset, _clen) = \
        unpack_exact("<IHHHHIIH", eocd)
    if n_total == 0xFFFF or cd_offset == 0xFFFFFFFF:  # ZIP64
        k = tail.rfind(b"\x50\x4b\x06\x06")
        rec = tail[k:k + 56]
        (_sz, _vm, _vn, _d1, _d2, n_disk64, n_total64, cd_size64, cd_offset64) = \
            unpack_exact("<QHHIIQQQQ", rec[4:56])
        return cd_offset64, cd_size64, n_total64
    return cd_offset, cd_size, n_total


def fetch_central_directory(url: str, cd_offset: int, cd_size: int,
                            part_file: Path = None, chunk: int = 4 * 1024 * 1024):
    """Yield central-directory bytes in order (memory-friendly, resumable via .part).

    On resume, replays the existing .part bytes first, then continues the download."""
    done = 0
    if part_file and part_file.exists():
        done = part_file.stat().st_size
        print(f"resuming central directory from {done/1e6:.1f} MB")
        with open(part_file, "rb") as fh:
            while True:
                blob = fh.read(chunk)
                if not blob:
                    break
                yield blob
    fh = open(part_file, "ab") if part_file else None
    pos = done
    while pos < cd_size:
        end = min(cd_offset + pos + chunk - 1, cd_offset + cd_size - 1)
        blob = http_range(url, cd_offset + pos, end)
        if fh:
            fh.write(blob)
        yield blob
        pos += len(blob)
        if pos % (64 * 1024 * 1024) < chunk:
            print(f"  {pos/1e6:.0f}/{cd_size/1e6:.0f} MB")
    if fh:
        fh.close()


def parse_entries(cd_bytes_iter) -> dict:
    index = {}
    buf = b""
    for chunk in cd_bytes_iter:
        buf += chunk
        pos = 0
        while pos + 46 <= len(buf):
            if unpack_exact("<I", buf[pos:pos + 4])[0] != 0x02014B50:
                pos += 1
                continue
            (vmade, vneed, flags, method, mtime, mdate, crc, csize, usize,
             nlen, elen, clen) = unpack_exact("<HHHHHHIIIHHH", buf[pos + 4:pos + 34])
            (disk, intattr, extattr, loffset) = unpack_exact("<HHII", buf[pos + 34:pos + 46])
            need = 46 + nlen + elen + clen
            if pos + need > len(buf):      # entry spans chunk boundary
                break
            # ZIP64 extra (id 0x0001): overflowed fields appear in order
            if 0xFFFFFFFF in (usize, csize, loffset):
                extra = buf[pos + 46 + nlen:pos + 46 + nlen + elen]
                p = 0
                zvals = []
                while p + 4 <= len(extra):
                    eid, esz = unpack_exact("<HH", extra[p:p + 4])
                    if eid == 0x0001:
                        zvals = list(unpack_exact("<" + "Q" * (esz // 8), extra[p + 4:p + 4 + esz]))
                        break
                    p += 4 + esz
                if usize == 0xFFFFFFFF and zvals:
                    usize = zvals.pop(0)
                if csize == 0xFFFFFFFF and zvals:
                    csize = zvals.pop(0)
                if loffset == 0xFFFFFFFF and zvals:
                    loffset = zvals.pop(0)
            name = buf[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
            index[name] = {"offset": loffset, "csize": csize, "usize": usize,
                           "method": method}
            pos += need
        buf = buf[pos:]  # keep the trailing partial entry
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"HEAD {args.url} ...")
    size = http_size(args.url)
    print(f"zip size: {size/1e9:.2f} GB")

    cd_offset, cd_size, n_total = parse_eocd_tail(args.url, size)
    print(f"central directory: {cd_size/1e6:.1f} MB at offset {cd_offset}, "
          f"{n_total} entries")

    print("downloading central directory ...")
    part = Path(args.out).with_suffix(".part")
    index = parse_entries(fetch_central_directory(args.url, cd_offset, cd_size, part))
    print(f"parsed {len(index)} entries")
    if part.exists():
        part.unlink()  # cleanup partial file after successful parse

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"url": args.url, "zip_size": size, "entries": index}, f)
    print(f"saved index -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
