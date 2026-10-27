# Experiment Plan — VLA vs Omni post-training comparison

> Status: **smoke-gate phase**(冒烟门禁,3×32G vGPU 已挂载,数据/模型全就绪)
> Last updated: 2026-10-27

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
| B3 | MiniCPM-o 4.5 | = B2 + Whisper audio, **speech-only** | SFT → GRPO | speech ablation (independent) |

> **B3 input modality (2026-10-27, revised):** B3 is AUDIO-ONLY — the text sequence
> carries only `(<image>)(<audio>)` placeholders, NO instruction words. Speech is the
> sole language channel. This makes B2 vs B3 a clean single-variable comparison
> (vision constant; language via text vs via sound) for the robot voice-control goal.
> TTS wavs are pre-synthesized from the instruction text (edge-tts/espeak-ng),
> so no new data is required.
>
> **Cross-modal 2×2 eval matrix (added):** besides the native comparison, BOTH models
> are tested on BOTH input channels on the same fixed instruction set (grpo_eval,
> 16 instructions, all with wavs):
>
> | | text input | audio input |
> |---|---|---|
> | B2 model | in-distribution | zero-shot transfer |
> | B3 model | zero-shot transfer | in-distribution |
>
> `scripts/evaluate.py --split csi --modality {native,audio,text}` implements the
> matrix (audio = rebuild same weights with audio tower ON + speech-only; text =
> audio tower OFF). Known caveat: 4/16 grpo_eval instruction TEXTS also appear in
> training windows (different states); strict subset = remaining 12 (report both).

### 1.2 Comparisons (7 research questions)

**Cross-paradigm (core)**
1. **A1 vs B2** — main: native VLA vs Omni+action head (both fully post-trained)
2. **A0 vs B0** — SFT-only sanity: paradigm gap without RL
3. **A1 vs B1** — LoRA-controlled: RL effect with frozen backbones on both sides

**Within-family (ablations)**
4. A0 vs A1 — GRPO gain (OpenVLA-OFT)
5. B0 vs B1 — GRPO gain (MiniCPM-o head-only)
6. B1 vs B2 — LoRA backbone vs head-only (MiniCPM-o)
7. B2 vs B3 — text vs speech language channel (**B-family internal only, never vs A**)
8. 2×2 cross-modal transfer — B2-on-audio & B3-on-text zero-shot (same 16-instr set)

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
- **Action head**: MiniCPM-o groups use a discretized 7×256-bin head (cross-entropy),
  matching OpenVLA's native bins. This replaces the continuous MLP head that collapsed
  to the dataset mean in the first smoke runs.
- **Env**: CALVIN (Franka, long-horizon chains), max 100 steps/episode, single static cam 224×224
- **SFT**: discrete-head CE (MiniCPM-o) / native-bin CE (OpenVLA); AdamW; lr head=1e-4,
  LoRA=2e-5; batch 1 chunk × 4 steps (full-coverage chunked sampling); wd 1e-4; grad ckpt;
  early stop on SFT-val loss (val never used in RL)
- **GRPO**: group_size=4, rollout_per_group=4, clip_eps=0.2, beta_kl=0.05; lr head=1e-4,
  LoRA=2e-5; max_grad_norm=1.0; group-relative advantage (no critic); KL vs frozen SFT
  reference; deterministic eval (temp=0) every 2000 episodes + checkpoint
- **Reward**: R = R_success + 0.5·R_dist + 0.1·R_action
  (success +10; dist −L2 ee→target with progress proxy; action jitter penalty)
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
- **SFT subset (CALVIN-only)**: ~500 env-A windows / ~32k steps, window-level stride 12
  across all env-A episodes (6,089 windows). 85/15 train/SFT-val, then a 16-window
  GRPO-eval pool is carved out of train (id_test stays locked).
- **ID/OOD test**: 34 validation windows / 1,982 steps (~0.5 GB).
- **Bridge V2**: **removed from the main comparison** — CALVIN-only for all groups.
  The loader still supports `dataset: [calvin, bridge]` as an optional ablation, but no
  group config uses it.
- **Actions**: CALVIN `rel_actions` are ALREADY normalized to [-1,1] (pos factor 50,
  rot factor 20, gripper ±1) and the env consumes them directly — no rescaling.
- **B3**: same CALVIN windows + synthesized 16 kHz TTS wav (edge-tts → espeak-ng),
  frame-aligned to 8-step sub-windows. Synthesized speech, not human — disclosed.
- Debug dataset (1.3 GB, downloaded): pipeline smoke tests only (1 episode/split).

### Minimal-training budget (applied — see configs/base.yaml)

| Knob | Original spec | Minimal | Rationale |
|---|---|---|---|
| CALVIN SFT data | calvin_A split (603k steps) | ~500 windows / ~32k steps | LoRA few-demo scale; controlled comparison needs consistency, not volume |
| Bridge | 20k+ trajs | **0 (removed)** | CALVIN-only main experiment; Bridge optional ablation only |
| SFT epochs | 20 | 2 (full-coverage chunked) | LoRA converges fast on small data |
| GRPO env steps | 400k | **20k** | ~21 GRPO updates (16 trajs/update) — minimal trend; raise if needed |
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

### Phase 0 — 数据与代码(✅ 全部完成)
- [x] Full codebase: configs, SFT/GRPO trainers, rollouts (HF engine), evaluator, plotting
- [x] 19 CPU smoke checks green; pydantic-locked hyperparameters; leak-guarded splits
- [x] CALVIN env-A 窗口抽取与预处理:stride 48→12(~500 窗口;~416 sft_train + ~76 sft_val + 16 grpo_eval + 34 id_test)
- [x] Bridge V2 从主实验移除(loader 仍保留支持);CSI 语音;鲁棒指令 34 条
- [x] 模型本地化:OpenVLA-7B(15G)+ MiniCPM-o-4_5(19G)
- [x] Git 基建:独立仓库 + 自动提交守护进程 + `/commit` 命令 + commit-writer agent + skill

### Phase 1 — GPU 验证(✅ 大部分完成)
- [x] 挂卡(3×32G vGPU)后 check_env 全绿
- [x] OpenVLA-7B 验证:加载 15.1G、SFT bwd batch4 23.1G、生成 15.4G
- [x] MiniCPM-o-4_5 验证(B0/B2 通过):PIL 修复(1-tile 1024 tokens)+ sdpa + gc + batch1
- [x] CALVIN env 渲染链路:show_gui 修正 + 软渲染回退 + int64/占位图防御
- [~] 冒烟门禁(见 docs/smoke_tests.md):A0/A1 SFT ✅、B0/B2 模型 ✅,其余进行中

### Phase 2 — 统一开训(冒烟全过后启动)
```
GPU0: A0(SFT) → A1(SFT→GRPO) → eval
GPU1: B2(SFT→GRPO) → eval                # 主实验,最早出结果
GPU2: B0(SFT) → B1(SFT→GRPO) → B3(SFT→GRPO 音频) → eval
```
- 启动:`VLA_DEVICE=cuda:N bash scripts/run_group.sh <groups>`
- 参数:SFT 2 epoch / patience 1;batch 2×2(A)/1×1(B);GRPO 20k 步;eval 10 seed × 20 窗口
- 预估:主对比 A1 vs B2 ≈ 8-10h;全部 6 组 ≈ 12-16h

### Phase 3 — 评估与分析
- [ ] 每组 ID/OOD/鲁棒评估(mean±std)+ 失败轨迹分类 + 绘图
- [ ] 7 个对比结论 + 论文图表

## 5. Risks & mitigations (tracked in docs/limitations.md)

| Risk | Mitigation |
|---|---|
| OFT HF id not published | 已回退 `openvla/openvla-7b`(本地快照) |
| MiniCPM-o 4.5 weights not public | 已确认存在并本地化(openbmb/MiniCPM-o-4_5) |
| MiniCPM-o forward signature differs | 已按真实源码适配(get_vllm_embedding/get_omni_embedding 路径) |
| vGPU 无 EGL | 软渲染回退 + 图像防御转换(见 smoke_tests.md 坑记录) |
| 32G 卡 9B 压线 | 1-tile + sdpa + gc + batch 1 已通过 |
| GRPO 20k 步趋势弱 | 先跑主对比;不足则 A1/B2 单独加预算 |
| 容器 bash 数组损坏 | 全部脚本用位置参数,禁用数组/("$@") |

## 6. Success criteria

- A1 vs B2 both run to completion under identical protocol; results reproducible from
  `configs/` + fixed seeds
- Primary metric: ID success rate (mean ± std over 20 seeds), OOD success rate
- Sample-efficiency: env steps to 50%/60% success
- All 7 comparisons answered with statistical spread reported; B3 isolated as ablation
