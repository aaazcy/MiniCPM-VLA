# Experiment Plan — VLA vs Omni post-training comparison

> Status: Phase 0 in progress (CPU-only data prep) · GPU not yet mounted
> Last updated: 2026-10-26

## 0. One-line goal

Can a general Omni foundation model (**MiniCPM-o 4.5, 9B**) with a newly added action head
match a native VLA (**OpenVLA-OFT-7B**) after identical post-training (SFT warmstart + GRPO
RL), and what does synchronized synthesized speech (B3) add?

## 1. Experiment matrix

### 1.1 Groups (6 arms)

| Group | Model | Trainable | Stage | Purpose |
|---|---|---|---|---|
| A0 | OpenVLA-OFT-7B | LoRA LLM + native head | SFT only | no-RL sanity |
| A1 | OpenVLA-OFT-7B | LoRA LLM + native head | SFT → GRPO | **main baseline** |
| B0 | MiniCPM-o 4.5 | frozen backbone + new head | SFT only | no-RL sanity |
| B1 | MiniCPM-o 4.5 | frozen backbone + new head | SFT → GRPO | head-only RL |
| B2 | MiniCPM-o 4.5 | LoRA(r=8,α=16) + new head | SFT → GRPO | **main experiment** |
| B3 | MiniCPM-o 4.5 | = B2 + Whisper audio | SFT → GRPO | speech ablation (independent) |

### 1.2 Comparisons (7 research questions)

**Cross-paradigm (core)**
1. **A1 vs B2** — main: native VLA vs Omni+action head (both fully post-trained)
2. **A0 vs B0** — SFT-only sanity: paradigm gap without RL
3. **A1 vs B1** — LoRA-controlled: RL effect with frozen backbones on both sides

**Within-family (ablations)**
4. A0 vs A1 — GRPO gain (OpenVLA-OFT)
5. B0 vs B1 — GRPO gain (MiniCPM-o head-only)
6. B1 vs B2 — LoRA backbone vs head-only (MiniCPM-o)
7. B2 vs B3 — synchronized speech benefit (**B-family internal only, never vs A**)

### 1.3 Runs

| Kind | Count | Detail |
|---|---|---|
| SFT training | 6 | one per group |
| GRPO training | 4 | A1, B1, B2, B3 |
| Evaluation | 18 | 6 groups × {ID, OOD, robustness}, 20 seeds each |
| Qualitative | 6 | 50 failed trajectories per group → failure pie |
| Auxiliary VLM | 2 | target recall / instruction understanding (per family) |

## 2. Locked protocol (identical for all groups)

- **Action space**: 7-D continuous `[dx,dy,dz,rx,ry,rz,gripper]` ∈ [-1,1]
- **Env**: CALVIN (Franka, long-horizon chains), max 100 steps/episode, single static cam 224×224
- **SFT**: MSE; AdamW; lr head=1e-4, LoRA=2e-5; batch 8; wd 1e-4; grad ckpt; early stop on
  SFT-val MSE (val never used in RL)
- **GRPO**: group_size=4, rollout_per_group=4, clip_eps=0.2, beta_kl=0.05; lr head=1e-4,
  LoRA=2e-5; max_grad_norm=1.0; group-relative advantage (no critic); KL vs frozen SFT
  reference; deterministic eval (temp=0) every 200 episodes + checkpoint
- **Reward**: R = R_success + 0.5·R_dist + 1.0·R_collide + 0.1·R_action
  (success +10; dist −L2 ee→target with progress proxy; collide −2; action jitter penalty)
- **Splits guard**: 85/15 train/SFT-val; ID test = env-A validation (held-out); OOD = env-D;
  robustness = official paraphrased instructions; sha256 manifests, leak assertions on load

## 3. Data plan (post-training scale)

- CALVIN full zip = 517 GB (1.89M step files) → **NOT downloaded**. Remote zip
  central-directory parsing + HTTP-range selective extraction
  (`fetch_zip_index.py` → `query_zip_index.py` → `selective_extract.py`).
- **Structure (verified)**: 147 long episodes (avg 12.2k steps) in training; env A =
  46 episodes / 603,660 steps = 6,089 64-step windows (scene_info fix: A = steps
  [1191339, 1795044]). Language is annotated per 64-step window. **No testing split**
  in task_ABC_D.zip → OOD follows the official ABC→D protocol: validation episodes
  rendered with `calvin_scene_D`.
- **SFT subset**: 127 env-A windows / 7,720 steps (~2.0 GB download), window-level
  stride 48 across all env-A episodes. 85/15 train/SFT-val.
- **ID/OOD test**: 34 validation windows / 1,982 steps (~0.5 GB).
- **Bridge V2**: ≤2000 trajectories via HF streaming (same seed for all groups).
- **Actions**: CALVIN `rel_actions` are ALREADY normalized to [-1,1] (pos factor 50,
  rot factor 20, gripper ±1) and the env consumes them directly — no rescaling.
- **B3**: same CALVIN windows + synthesized 16 kHz TTS wav (edge-tts → espeak-ng),
  frame-aligned to 8-step sub-windows. Synthesized speech, not human — disclosed.
- Debug dataset (1.3 GB, downloaded): pipeline smoke tests only (1 episode/split).

### Minimal-training budget (applied — see configs/base.yaml)

| Knob | Original spec | Minimal | Rationale |
|---|---|---|---|
| CALVIN SFT data | calvin_A split (603k steps) | 127 windows / 7.7k steps | LoRA few-demo scale; controlled comparison needs consistency, not volume |
| Bridge | 20k+ trajs | 2k trajs | embodiment diversity with ~1/10 cost |
| SFT epochs | 20 | 6 (+early stop) | LoRA converges fast on small data |
| GRPO env steps | 400k | **40k** | ~60 GRPO updates — enough for a stable trend at 16 trajs/update |
| Rollout episode cap | 100 | 60 | shorter rollouts, same action space |
| Eval during GRPO | every 200 eps × 8 | every 2000 eps × 5 | 10× fewer eval env steps |
| Qualitative failures | 50/group | 20/group | pie chart still interpretable |

Per-group estimate (L40S 48G): SFT ≈ 10–20 min; GRPO ≈ 1–3 h (vLLM for A, HF engine
for B — B3 audio slower); eval ≈ 0.5–1 h. All 6 groups ≈ 1–2 days wall-clock.
Each knob is a config value — raise later if a trend needs more signal.

**Container constraint found**: cgroup memory limit = 2 GB (cardless AutoDL mode).
All Phase-0 tooling is streaming/low-memory (ijson single-pass); re-check RAM after
mounting the GPU (the limit is expected to follow the instance spec).

## 4. Execution phases

### Phase 0 — CPU now (in progress)
- [x] Full codebase: configs, SFT/GRPO trainers, rollouts (vLLM+HF), evaluator, plotting
- [x] 19 CPU smoke checks green; pydantic-locked hyperparameters; leak-guarded splits
- [x] scene_info fix (env-A mapping) downloaded; debug dataset downloaded
- [x] zip central directory (229 MB) parsed → 1.89M-entry index (JSON on disk)
- [x] Metadata extracted (ep_start_end_ids, lang_annotations, paraphrase; 8 files)
- [x] Window-level plan: 127 env-A windows / 7,720 steps (~2.0 GB) + 34 val windows
- [x] Downloaded 11,046 step files (~3.3 GB)
- [x] CALVIN preprocessed: 107 train / 20 SFT-val / 34 id_test segments (708 MB)
- [x] Robustness set: 34 paraphrased instructions (official lang_paraphrase)
- [x] B3 TTS: 107/107 synthesized 16 kHz wavs (edge-tts)
- [x] Bridge V2: 1,995 episodes via hf-mirror
- [x] grpo_eval split: 16 windows held out from sft_train (leak-free GRPO eval)
- [x] OpenVLA-7B GPU-verified (load 15.1G / SFT bwd batch4 23.1G / generate 15.4G)
- [x] A0 SFT smoke + full training started (paused for disk expansion, ~step 250)
- [~] MiniCPM-o-4_5 download (18/20 GB, RESUMABLE via snapshot_download)

### PAUSED — user expanding data disk (resume steps below)
```bash
# 1. resume MiniCPM-o download (auto-resumes from .incomplete)
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM-o-4_5', local_dir='/root/autodl-tmp/models/MiniCPM-o-4_5', max_workers=2)"
# 2. verify B wrapper on GPU1 (incl. B3 audio path + VRAM baseline)
python scripts/verify_models.py --group b2 --batch 2
# 3. relaunch training
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_group.sh a0 > runs/log_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_group.sh b0 b1 b2 b3 > runs/log_b.log 2>&1 &
```

### Phase 1 — after GPU mount (L40S 48 G recommended)
- [ ] Mount GPU → `check_env.py` green (torch 2.5.1+cu124, BF16)
- [ ] `pip install vllm==0.6.6.post1` (+ calvin_env, openvla packages)
- [ ] Verify weights: `openvla/openvla-7b-oft` (fallback openvla-7b) — Phase A
- [ ] MiniCPM-o 4.5 checkpoint id/loc — Phase B (adapt `encode()` to real forward path)
- [ ] Train A0 → A1 (SFT → GRPO), eval id/ood/robustness
- [ ] Train B0 → B1 → B2 → B3, eval all splits
- [ ] Qualitative failure labeling (50 trajs/group) + auxiliary VLM checks
- [ ] plot_results (B3 excluded from A-vs-B charts)

## 5. Risks & mitigations (tracked in docs/limitations.md)

| Risk | Mitigation |
|---|---|
| OFT HF id not published | fallback `openvla/openvla-7b`, same code path |
| MiniCPM-o 4.5 weights not public | generic wrapper + verified fallback `MiniCPM-o-2_6`; replace id when available |
| MiniCPM-o forward signature differs | Phase-1 TODO in `src/models/minicpmo.py` |
| Freiburg server unstable | 16 MB chunked downloader with retry + resume; index cached |
| env-A range from scene_info unverified | render 3 sampled episodes in `calvin_scene_A` before training |
| HF download slow | `HF_ENDPOINT=https://hf-mirror.com` |
| vLLM/cu124 mismatch | pin `vllm==0.6.6.post1`; HF engine fallback for all groups |

## 6. Success criteria

- A1 vs B2 both run to completion under identical protocol; results reproducible from
  `configs/` + fixed seeds
- Primary metric: ID success rate (mean ± std over 20 seeds), OOD success rate
- Sample-efficiency: env steps to 50%/60% success
- All 7 comparisons answered with statistical spread reported; B3 isolated as ablation
