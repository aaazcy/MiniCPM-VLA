# VLA vs Omni: controlled post-training comparison — OpenVLA-OFT-7B vs MiniCPM-o 4.5

Controlled comparison of two paradigms for robot manipulation, focused on **post-training**
(SFT warmstart + GRPO RL):

1. Baseline A: native pre-trained VLA — **OpenVLA-OFT-7B** (enhanced recipe) + SFT + GRPO
2. Experimental B: general Omni foundation model — **MiniCPM-o 4.5 (9B)** + added action
   head + SFT + GRPO
3. Ablation B3: MiniCPM-o with synchronized **synthesized speech** (CSI), independent
   ablation, never compared head-to-head vs A.

The core question: can a general Omni vision-audio-language foundation model with an added
action head match a native VLA on robot manipulation, and what does synchronized speech add?

---

## 1. Server profile & plan adjustments (checked on this machine)

### Observed hardware (AutoDL container, GPU NOT mounted yet)

| Resource | Value | Verdict |
|---|---|---|
| CPU | 2× Intel Xeon Platinum 8352V, 128 threads, AVX-512 | ✅ preprocessing trivial |
| RAM | 1.0 TiB (934 GiB available) | ✅ far above the 64 GB requirement |
| GPU | **none mounted** (`nvidia-smi` empty, `torch.cuda.is_available()=False`) | ⚠️ mount before Phase 1 (L40S 48 G recommended) |
| CUDA toolkit | 12.4.131, driver SDK 550.120 | ✅ supports CUDA ≤ 12.4 |
| Python | 3.12.3 (miniconda), torch 2.5.1+cu124 preinstalled | ✅ base env |
| System disk | overlay 30 G, ~25 G free | ⚠️ keep envs/weights off it |
| Data disk `/root/autodl-tmp` | **quota 50 G** (underlying 2×7.6 T NVMe RAID1) | ⚠️ see budget below |
| Quirk | this Python's `struct.unpack` requires exactly-sized buffers | worked around in all parsers |

### Key findings that shaped the plan

1. **CALVIN is much bigger than the spec assumed.** The official `task_ABC_D.zip` is
   **517 GB** (1.89 M step-level npz files). There is **no published "calvin_A-only" zip**.
   Since this experiment only compares post-training, we do NOT download the full zip:
   * `scene_info.npy` (a 525-byte official fix) maps step ranges per environment:
     `A = [1191339, 1795044]`, `B = [0, 598909]`, `C = [598910, 1191338]`.
   * We parse the zip's central directory over HTTP range requests
     (`scripts/fetch_zip_index.py`), then selectively download only **~200–300 env-A
     episodes** for SFT (~4–8 GB raw), env-A validation episodes for the ID test, and
     env-D testing episodes for the OOD test (`scripts/selective_extract.py`).
   * The 1.3 G `calvin_debug_dataset` (already downloaded) is used for pipeline smoke
     tests only — it contains a single env-D episode per split, too small for training.
2. **Baseline choice (per user decision): OpenVLA-OFT-7B only.** π0.5-3B was evaluated
   and dropped: flow-matching head conflicts with the shared MSE/GRPO spec, requires
   proprio, and 3B vs 9B size asymmetry. OFT keeps the same code path as the original
   OpenVLA while using the improved training recipe.
3. **Post-training data scale (per user decision).** No full-scale CALVIN/Bridge.
   SFT data ≈ 200–300 env-A episodes (~45–90 k steps) + ≤5 k Bridge subsample.
4. **B3 speech cannot use stock vLLM** (no Whisper-audio path for MiniCPM-o): B3 rolls
   out with the HF engine; vision-only groups use vLLM. See `src/rollout/engine.py`.
5. **Pinned versions** from the preinstalled env: torch 2.5.1+cu124 → vLLM `0.6.6.post1`.
   The GRPO loop is custom (TRL cannot run CALVIN env rollouts).

### Disk budget (50 G quota — sufficient with staged weights, no expansion required)

| Item | Size |
|---|---|
| CALVIN env-A subset (raw → preprocessed 224px) | ~4–8 G |
| Bridge V2 subsample (5 k trajs) | ~2 G |
| CSI wavs (TTS, 16 kHz mono) | <0.5 G |
| OpenVLA-OFT-7B weights (BF16, staged) | ~15 G |
| MiniCPM-o 4.5 weights (BF16, staged) | ~19 G |
| Checkpoints (LoRA + head only) | ~1 G |
| conda env (torch + vLLM + deps) | ~12 G |
| HF cache (staged, purged between phases) | 0 (purged) |

Phase A peak ≈ 15 + 8 + 2 + 12 ≈ 37 G ✓ — Phase B peak ≈ 19 + 8 + 2 + 12 ≈ 41 G ✓
(Expanding the disk is optional; if you do expand, both models can stay resident.)

---

## 2. Groups (strict control: same env, action space, reward, GRPO hyperparams, eval)

| Group | Model | Backbone | Stage | Note |
|---|---|---|---|---|
| A0 | OpenVLA-OFT-7B | LoRA LLM + native head | SFT only | no-RL sanity |
| A1 | OpenVLA-OFT-7B | LoRA LLM + native head | SFT → GRPO | **main baseline** |
| B0 | MiniCPM-o 4.5 | frozen, head only | SFT only | head-only sanity |
| B1 | MiniCPM-o 4.5 | frozen, head only | SFT → GRPO | |
| B2 | MiniCPM-o 4.5 | LoRA(r=8,α=16) + head | SFT → GRPO | **main experiment** |
| B3 | MiniCPM-o 4.5 | = B2 + Whisper audio on | SFT → GRPO | CSI speech ablation — **never compared head-to-head vs A** |

Action space: 7-D continuous `[dx, dy, dz, rx, ry, rz, gripper]`, normalized [-1, 1].
Action head (B groups, last-token hidden): `Linear(h,512) → LN → GELU → Linear(512,256) → LN → GELU → Linear(256,7) → Tanh`.
OpenVLA uses its native discretized action head (trained with the same trainer code path).

### Data (post-training scale)
- **SFT (A0,A1,B0,B1,B2):** ~200–300 env-A CALVIN episodes (85% train / 15% SFT-val,
  early stop only, never in RL) + ≤5 k Bridge V2 subsample; images 224×224; heuristic
  collision/failure filtering; identical data for every group.
- **B3:** CSI — same CALVIN episodes, text → synthesized 16 kHz TTS wav, frame-aligned
  (8-frame windows). *Synthesized, not human speech — limitation.*
- **Tests (never seen in training):** ID = env-A validation episodes (held-out);
  OOD = env-D episodes (cross-scene: new camera/texture/object layout); robustness =
  official paraphrased instructions (`lang_paraphrase-MiniLM-L3-v2`). 20 seeds per episode.

### Pipeline (identical for every group)
1. **SFT:** MSE(pred, demo); AdamW; LR head=1e-4, LoRA=2e-5; batch 8; wd 1e-4; grad ckpt;
   early stop on SFT-val MSE plateau; best ckpt = GRPO start.
2. **GRPO:** `group_size=4, rollout_per_group=4, clip_eps=0.2, beta_kl=0.05`;
   LR head=1e-4, LoRA=2e-5; max_grad_norm=1.0; AdamW; group-relative advantage (no
   critic); clipped policy loss + KL vs frozen SFT reference; backprop only LoRA + head;
   deterministic eval (temp 0) every 200 episodes + checkpoint.
3. **Reward:** `R = R_success + 0.5·R_dist + 1.0·R_collide + 0.1·R_action`
   (success +10; dist = −L2 ee→target, progress proxy when unavailable; collide −2;
   action = jitter penalty).

### Metrics
Full-task success (ID / OOD, primary), sample-efficiency steps to 50%/60%, mean return,
mean steps, action smoothness, ee positional error, robustness degradation, failure-type
pie chart, auxiliary VLM checks (target recall / instruction understanding).

---

## 3. Project layout

```
vla-omni-bench/
├── README.md  docs/limitations.md  requirements.txt
├── configs/
│   ├── base.yaml                  # shared locked hyperparameters
│   ├── group/{a0,a1,b0,b1,b2,b3}.yaml
│   └── eval.yaml
├── scripts/
│   ├── check_env.py               # GPU/CUDA/vLLM/disk sanity check
│   ├── fetch_zip_index.py         # parse remote CALVIN zip central dir (HTTP ranges)
│   ├── plan_calvin_subset.py      # pick env-A episodes from scene_info + metadata
│   ├── selective_extract.py       # download chosen files from the 517G zip
│   ├── download_data.sh           # debug dataset + pointers (no GPU needed)
│   ├── preprocess_data.py         # raw steps -> 224px trajectory npz + splits (leak-guarded)
│   ├── preprocess_csi_tts.py      # B3 synthesized 16kHz speech
│   ├── train_sft.py / train_grpo.py / evaluate.py
│   ├── run_group.sh               # sequential / multi-GPU runner
│   ├── plot_results.py            # success curves + failure pie charts
│   └── smoke_test.py              # CPU-only smoke tests
└── src/
    ├── config.py                  # pydantic config, locked & validated
    ├── data/{splits,loader,csi}.py
    ├── envs/calvin_env.py         # CALVIN wrapper (scene cfg A/D, 100-step episodes)
    ├── models/{action_head,openvla,minicpmo,factory}.py
    ├── rollout/engine.py          # vLLM / HF rollout abstraction
    ├── training/{rewards,sft,grpo}.py
    ├── eval/evaluator.py
    └── utils/{logger,ckpt,instruction_sampler}.py
```

## 4. Quickstart

```bash
# Phase 0 — CPU (current state)
python scripts/check_env.py                     # reports "GPU missing" until you mount one
bash scripts/download_data.sh                   # debug dataset + scene_info fix
python scripts/fetch_zip_index.py --url http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip \
    --out data/calvin/task_ABC_D.index.json     # ~230 MB, resumable (chunked fetch script included)
python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
    --include 'task_ABC_D/training/ep_start_end_ids.npy' ... \
    --out data/calvin/meta.index.json           # slim metadata index (streaming, 2GB-RAM-safe)
python scripts/selective_extract.py --index data/calvin/meta.index.json --all \
    --out-dir data/calvin/meta
cp data/calvin/meta/training/scene_info.npy data/calvin/meta/task_ABC_D/training/
python scripts/plan_calvin_subset.py --training-meta data/calvin/meta/task_ABC_D/training \
    --validation-meta data/calvin/meta/task_ABC_D/validation --train-window-stride 12
python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
    --episode-list data/calvin/plan.json --out data/calvin/steps.index.json
python scripts/selective_extract.py --index data/calvin/steps.index.json --all \
    --out-dir data/calvin/raw                     # ~8 GB, resumable
python scripts/preprocess_data.py --dataset calvin --raw data/calvin/raw/task_ABC_D \
    --lang-ann data/calvin/meta/task_ABC_D/training/lang_annotations/auto_lang_ann.npy
python scripts/smoke_test.py

# Phase 1 — after mounting an L40S 48G
python scripts/check_env.py                    # must pass before training
bash scripts/run_group.sh                      # a0 a1 (Phase A) then b0..b3 (Phase B)
python scripts/plot_results.py --outdir viz
```

## 5. Reproducibility & experiment guards

- All hyperparameters locked in `configs/*.yaml`, validated by pydantic; each run logs the
  resolved config to wandb.
- Split guard (`src/data/splits.py`): manifests carry sha256 hashes; loaders assert test
  ids never appear in train/val.
- Same reward code path, GRPO loop and evaluator for all groups — differences confined to
  model/backbone/LoRA/audio flags in group configs.
- B3 is tagged `is_ablation` in wandb and excluded from all head-to-head plotting.
- All random seeds fixed; eval repeats 20× per episode.

## 6. Limitations (full text in docs/limitations.md)

Different vision backbones (ViT-L vs SigLip2), synthesized speech (B3), heuristic
trajectory filtering, single static-camera view, sim2real gap, simulation stochasticity,
env-A episode range from scene_info (verify in Phase 1), OFT HF id to verify, small-data
post-training regime (results are about post-training comparison, not full-scale VLA SOTA).
