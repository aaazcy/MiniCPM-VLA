#!/usr/bin/env python3
"""scripts/plot_results.py — success curves + failure pie charts.

Inputs (produced by training/eval):
  * runs/<group>/grpo/history.json      — GRPO metrics over env steps
  * runs/<group>/eval_*.json            — final ID/OOD/robustness metrics
  * runs/<group>/failures.json          — failed trajectories with labels

Plots:
  1. ID / OOD success-rate curves vs env steps (per group, and A-vs-B comparison)
  2. Sample-efficiency lines at 50% / 60% success thresholds
  3. Failure-category pie chart per group
  B3 (ablation) is plotted separately and NEVER mixed into the A-vs-B comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_history(run_dir: Path) -> list:
    p = run_dir / "grpo" / "history.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # SFT-only groups: flat line at their eval success
    return []


def plot_success_curves(groups: dict, outdir: Path, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for gid, run_dir in groups.items():
        hist = load_history(run_dir)
        if not hist:
            continue
        steps = [h["env_steps"] for h in hist]
        succ = [h.get("grpo/success_rate", np.nan) for h in hist]
        # sample-efficiency: steps at first crossing of 50%/60% success
        for thresh in (0.5, 0.6):
            idx = next((i for i, s in enumerate(succ) if s >= thresh), None)
            if idx is not None:
                axes[0].axvline(steps[idx], color="gray", lw=0.6, alpha=0.5)
                axes[1].annotate(f"{gid}@{int(thresh*100)}%: {steps[idx]}",
                                 (0.05, 0.9 - 0.08 * len(axes[1].texts)), xycoords="axes fraction",
                                 fontsize=8)
        axes[0].plot(steps, succ, marker=".", label=gid)
    axes[0].axhline(0.5, ls="--", c="k", lw=0.8)
    axes[0].axhline(0.6, ls="--", c="k", lw=0.8)
    axes[0].set_xlabel("env steps")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("GRPO success curves (rollout success)")
    axes[0].legend()

    # final eval metrics bar chart (ID vs OOD)
    import matplotlib.pyplot as plt2
    final = {}
    for gid, run_dir in groups.items():
        for split, label in [("eval_id.json", "id"), ("eval_ood.json", "ood")]:
            p = run_dir / split
            if p.exists():
                with open(p) as f:
                    final.setdefault(gid, {})[label] = json.load(f)["metrics"].get(
                        "success_rate", np.nan)
    if final:
        gids = list(final)
        x = np.arange(len(gids))
        w = 0.35
        axes[1].bar(x - w / 2, [final[g].get("id", np.nan) for g in gids], w, label="ID")
        axes[1].bar(x + w / 2, [final[g].get("ood", np.nan) for g in gids], w, label="OOD")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(gids)
        axes[1].set_ylabel("success rate")
        axes[1].set_title("Final ID / OOD full-task success")
        axes[1].legend()

    out = outdir / "success_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_failure_pie(run_dir: Path, outdir: Path, show: bool = False):
    """Failure-category pie chart from runs/<group>/failures.json (labels filled by
    the labeling step: manual or heuristic, see docs)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    p = run_dir / "failures.json"
    if not p.exists():
        return
    with open(p) as f:
        failures = json.load(f)
    labels = [f.get("label") or "unlabeled" for f in failures]
    counts = Counter(labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values(), labels=[f"{k}\n({v})" for k, v in counts.items()],
           autopct="%1.1f%%", startangle=90)
    ax.set_title(f"{run_dir.parent.name} failure categories (n={len(failures)})")
    out = outdir / f"failure_pie_{run_dir.parent.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "viz"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    runs = {p.name.upper(): p for p in sorted(root.iterdir()) if p.is_dir()}
    # head-to-head comparison: exclude the B3 ablation (never mixed with A groups)
    compare = {g: p for g, p in runs.items() if g != "B3"}
    plot_success_curves(compare, Path(args.outdir), args.show)
    for gid, run_dir in runs.items():
        plot_failure_pie(run_dir, Path(args.outdir), args.show)
    # B3 ablation gets its own curve plot
    if "B3" in runs:
        plot_success_curves({"B3 (ablation)": runs["B3"]}, Path(args.outdir), args.show)


if __name__ == "__main__":
    main()
