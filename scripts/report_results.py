#!/usr/bin/env python3
"""scripts/report_results.py — aggregate the results ledger into readable tables.

Reads runs/results.jsonl (append-only ledger written by evaluate.py) and prints:
  1. aggregate metrics per group (success_rate, mean_steps, action_smoothness)
  2. per-task success-rate table (task | group | n_instructions | success_rate)

Handles sharded evals (the orchestrator runs `--shard K 3`); shard entries are
re-aggregated by instruction count, so no separate merge step is required.

Usage:
    python scripts/report_results.py                 # id/native
    python scripts/report_results.py --split ood
    python scripts/report_results.py --modality audio
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "runs" / "results.jsonl"


def load_ledger() -> list:
    if not LEDGER.exists():
        return []
    entries = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def aggregate_entries(es: list):
    """Aggregate one eval run's entries (1 non-sharded, or N shards) -> (metrics, per_instr)."""
    if len(es) == 1 and es[0].get("shard") is None:
        return es[0]["metrics"], es[0].get("per_instruction", [])
    acc, wsum, total = {}, {}, 0
    per_instr = []
    for e in es:
        n = len(e.get("per_instruction", []))
        if n == 0:
            n = e.get("metrics", {}).get("n_instructions", 0)
        total += n
        for k, v in e.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + v * n
                wsum[k] = wsum.get(k, 0.0) + n
        per_instr.extend(e.get("per_instruction", []))
    merged = {k: acc[k] / wsum[k] for k in acc if wsum.get(k)}
    merged["n_instructions"] = total
    return merged, per_instr


def latest_runs(entries: list, split: str, modality: str) -> dict:
    """group -> latest eval run {metrics, per_instruction, ckpt, ts}."""
    runs = defaultdict(list)
    for e in entries:
        if e.get("split") != split or e.get("modality") != modality:
            continue
        runs[(e["group"], e.get("ckpt"))].append(e)

    latest = {}
    for (group, ckpt), es in runs.items():
        metrics, per_instr = aggregate_entries(es)
        ts = max(e.get("ts", "") for e in es)
        cur = latest.get(group)
        if cur is None or ts >= cur["ts"]:
            latest[group] = {"group": group, "ckpt": ckpt, "ts": ts,
                             "metrics": metrics, "per_instruction": per_instr}
    return latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="id")
    ap.add_argument("--modality", default="native")
    args = ap.parse_args()

    entries = load_ledger()
    if not entries:
        print(f"no results in {LEDGER} yet — run evaluate.py first")
        return

    latest = latest_runs(entries, args.split, args.modality)
    if not latest:
        print(f"no {args.split}/{args.modality} results yet")
        return

    groups = sorted(latest)
    print(f"\n=== aggregate metrics ({args.split}/{args.modality}) ===")
    hdr = f"{'group':6s} {'success':>9s} {'mean_steps':>11s} {'smoothness':>11s} {'n_instr':>8s} {'ckpt'}"
    print(hdr)
    print("-" * len(hdr))
    for g in groups:
        r = latest[g]
        m = r["metrics"]
        ckpt = Path(r["ckpt"]).name if r.get("ckpt") else "-"
        print(f"{g:6s} {m.get('success_rate', 0):9.3f} {m.get('mean_steps', 0):11.1f} "
              f"{m.get('action_smoothness', 0):11.3f} {m.get('n_instructions', 0):8d} {ckpt}")

    print(f"\n=== per-task success rate ({args.split}/{args.modality}) ===")
    task_rates = defaultdict(lambda: defaultdict(list))
    task_n = defaultdict(lambda: defaultdict(int))
    for g in groups:
        for pi in latest[g]["per_instruction"]:
            t = pi.get("task") or "(none)"
            task_rates[t][g].append(pi["metrics"]["success_rate"])
            task_n[t][g] += 1

    col_w = 34
    print(f"{'task':<{col_w}s} " + "".join(f"{g:>10s}" for g in groups) + f"{'n':>6s}")
    print("-" * (col_w + 1 + 10 * len(groups) + 6))
    for t in sorted(task_rates):
        row = f"{t:<{col_w}s} "
        for g in groups:
            rates = task_rates[t].get(g, [])
            if rates:
                row += f"{sum(rates) / len(rates):10.2f}"
            else:
                row += f"{'-':>10s}"
        counts = "+".join(str(task_n[t][g]) for g in groups)
        print(f"{row} {counts:>6s}")


if __name__ == "__main__":
    main()
