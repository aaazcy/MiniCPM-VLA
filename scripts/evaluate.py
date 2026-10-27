#!/usr/bin/env python3
"""scripts/evaluate.py — full evaluation protocol (ID / OOD / robustness).

Usage:
    python scripts/evaluate.py --group a1 --split id
    python scripts/evaluate.py --group b2 --split ood
    python scripts/evaluate.py --group b2 --split robustness
    python scripts/evaluate.py --group b2 --qualitative-failures   # 50 failed trajs

Loads the best GRPO checkpoint (or best SFT checkpoint for SFT-only groups),
runs 20 seeded repeats per episode, logs to wandb and writes JSON to runs/<group>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.eval.evaluator import Evaluator  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.rollout.engine import make_engine  # noqa: E402
from src.utils.logger import Logger  # noqa: E402


def load_instructions(manifest: Path):
    """Collect (instruction, task, init_robot_obs, init_scene_obs) from a split manifest."""
    import glob
    root = manifest.parent.parent
    split = manifest.stem.replace("calvin_", "")
    with open(manifest) as f:
        ids = json.load(f)["ids"]
    instrs, tasks, init_states = [], [], []
    for tid in ids:
        candidates = glob.glob(str(root / split / f"{tid}.npz")) or \
            glob.glob(str(root / "sft_train" / f"{tid}.npz")) or \
            glob.glob(str(root / "id_test" / f"{tid}.npz")) or \
            glob.glob(str(root / "ood_test" / f"{tid}.npz"))
        if not candidates:
            continue
        with np.load(candidates[0]) as f_:
            instrs.append(str(f_["lang"]))
            tasks.append(str(f_["task"]) if "task" in f_ else None)
            if "init_robot_obs" in f_ and "init_scene_obs" in f_:
                init_states.append((f_["init_robot_obs"].astype(np.float32),
                                    f_["init_scene_obs"].astype(np.float32)))
            else:
                init_states.append(None)
    return instrs, tasks, init_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["a0", "a1", "b0", "b1", "b2", "b3"])
    ap.add_argument("--split", choices=["id", "ood", "robustness", "csi"], required=True)
    ap.add_argument("--modality", choices=["native", "audio", "text"], default="native",
                    help="cross-modal 2x2 matrix: 'audio' = speech-only input, "
                         "'text' = text-only input, 'native' = the group's own "
                         "trained modality (B2->text, B3->speech)")
    ap.add_argument("--qualitative-failures", action="store_true",
                    help="collect 50 failed trajectories for failure labeling")
    ap.add_argument("--ckpt", default="auto", help="checkpoint dir | 'auto'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true",
                    help="cold-start probe: 5 instructions x 3 seeds (~15 min)")
    ap.add_argument("--shard", type=int, nargs=2, default=None, metavar=("K", "N"),
                    help="evaluate instructions[K::N] (split the eval across N GPUs)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override eval temperature (e.g. 1.0 for stochastic-rollout probe)")
    args = ap.parse_args()

    cfg = load_config(args.group)
    if args.quick:
        cfg.eval.max_episodes = 5
        cfg.eval.seeds = 3
    if args.temperature is not None:
        cfg.eval.temperature = args.temperature
    # Cross-modal override: rebuild the SAME model with the other input channel.
    # Base weights are identical across B groups, so B2-ckpt-on-audio and
    # B3-ckpt-on-text are valid zero-shot transfer tests.
    if args.modality == "audio":
        cfg.group.audio = True
        cfg.group.audio_only = True
    elif args.modality == "text":
        cfg.group.audio = False
        cfg.group.audio_only = False
    torch.manual_seed(cfg.experiment.seed)
    logger = Logger(cfg, cfg.group.id, tags=["eval"])

    # checkpoint selection: GRPO best > SFT best (SFT-only groups) — but ONLY
    # checkpoints whose action-head config matches the current one (stale
    # checkpoints from a previous head architecture are skipped loudly)
    if args.ckpt == "auto":
        from src.utils.ckpt import pick_compatible_ckpt
        grpo_best = Path(cfg.grpo_ckpt_dir) / "last"
        sft_best = Path(cfg.sft_ckpt_dir) / "best"
        ckpt = pick_compatible_ckpt([grpo_best, sft_best], cfg)
        if ckpt is None:
            raise FileNotFoundError(
                f"no compatible checkpoint in {cfg.grpo_ckpt_dir} or "
                f"{cfg.sft_ckpt_dir} (action-head config: {cfg.model.action_head})")
    else:
        ckpt = Path(args.ckpt)
    if not (ckpt / "trainer_state.pt").exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")

    policy = build_model(cfg, stage="ref")  # eval mode, frozen
    policy.load_adapter(str(ckpt))
    policy.to(args.device)
    policy.eval()

    engine = make_engine(cfg, policy=policy, processor=policy.processor,
                         lora_path=str(ckpt), device=args.device)
    evaluator = Evaluator(cfg, policy, engine=engine, device=args.device)

    calvin_root = Path(cfg.dataset.calvin_root)
    if args.split == "id":
        instrs, tasks, init_states = load_instructions(calvin_root / "splits" / "calvin_id_test.json")
        scenes = ["calvin_scene_A"] * len(instrs)
    elif args.split == "ood":
        # official ABC->D protocol: same validation instructions rendered in env D
        instrs, tasks, init_states = load_instructions(calvin_root / "splits" / "calvin_id_test.json")
        scenes = ["calvin_scene_D"] * len(instrs)
    elif args.split == "csi":
        # cross-modal 2x2 matrix: grpo_eval windows (16 instructions, ALL have
        # synthesized wavs) — the same fixed set is used for audio AND text
        # evals so the two channels are directly comparable.
        instrs, tasks, init_states = load_instructions(calvin_root / "splits" / "calvin_grpo_eval.json")
        scenes = ["calvin_scene_A"] * len(instrs)
    else:
        # robustness: paraphrased / reordered / long instructions (same episodes, new text)
        rob_path = calvin_root / "splits" / "robust_instructions.json"
        if not rob_path.exists():
            raise FileNotFoundError(
                f"{rob_path} missing — create it (paraphrased instructions keyed by "
                "the same trajectory ids as calvin_id_test) or run "
                "scripts/make_robust_set.py from the official lang_paraphrase file")
        with open(rob_path) as f:
            rob = json.load(f)
        instrs, init_states = rob["instructions"], rob["init_states"] \
            if "init_states" in rob else [None] * len(rob["instructions"])
        # robustness keeps the SAME trajectory ids as id_test -> reuse their task names
        tasks = rob.get("tasks", [None] * len(instrs))
        scenes = ["calvin_scene_A"] * len(instrs)

    if args.shard is not None:
        k, n = args.shard
        instrs = instrs[k::n]
        tasks = tasks[k::n]
        init_states = init_states[k::n]
        scenes = scenes[k::n]

    if args.qualitative_failures:
        failures = evaluator.collect_failures(
            instrs, n_failed=cfg.qualitative.sample_failed_trajectories
            if hasattr(cfg, "qualitative") else 20,
            out_path=cfg.run_dir / "failures.json", tasks=tasks)
        print(f"collected {len(failures)} failed trajectories -> {cfg.run_dir}/failures.json")
        logger.finish()
        return

    metrics = evaluator.evaluate_subset(instrs, cfg.eval.seeds, cfg.eval.temperature,
                                        scenes=scenes, init_states=init_states, tasks=tasks)
    logger.log({f"eval/{args.split}/{k}": v for k, v in metrics.items()}, commit=True)
    out = cfg.run_dir / f"eval_{args.split}_{args.modality}.json"
    shard_tag = f"_shard{args.shard[0]}of{args.shard[1]}" if args.shard else ""
    out = cfg.run_dir / f"eval_{args.split}_{args.modality}{shard_tag}.json"
    with open(out, "w") as f:
        json.dump({"group": cfg.group.id, "split": args.split,
                   "modality": args.modality, "shard": args.shard,
                   "ckpt": str(ckpt), "n_instructions": len(instrs),
                   "metrics": metrics}, f, indent=2)
    print(f"[{cfg.group.id}/{args.split}/{args.modality}] "
          f"{json.dumps(metrics, indent=2)}\nSaved to {out}")
    logger.finish()


if __name__ == "__main__":
    main()
