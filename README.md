# MiniCPM-VLA

Post-training the multimodal speech model **MiniCPM-o 4.5** into a robot-manipulation policy (VLA): attach a 7-DoF action head, run two-stage post-training (**SFT behavior cloning + GRPO reinforcement learning**), and evaluate on the **CALVIN** benchmark against a native VLA baseline (**OpenVLA-7B**) to verify **"directly controlling a robot arm from spoken instructions"**.

> **Status**: end-to-end pipeline verified (CE SFT → GRPO → CALVIN success detection). Training and evaluation in progress.

## Highlights

- **Omnimodal VLA**: image + text + speech (Whisper audio tower) inputs → 7-DoF action output (end-effector pose + gripper, [−1,1]).
- **Discretized action head**: both families output a categorical distribution over 7×256 uniform bins in [−1,1], trained with **cross-entropy** (matching OpenVLA's native bin parameterization — avoids the mean-collapse of direct MSE regression).
- **Two-stage post-training**: SFT (cross-entropy behavior cloning) → GRPO (group-relative policy optimization with KL constraint).
- **Parameter-efficient**: LoRA adapters on the LLM + a **fully fine-tuned action head**. OpenVLA's native `lm_head` is unfrozen so it can adapt to CALVIN (a frozen pre-trained head collapses to a constant action on out-of-domain data).
- **CALVIN success detection implemented**: the raw CALVIN env only reports `{robot_info, scene_info}`, so success was previously *always False*. We compute it via the official `Tasks` oracle and store the subtask name in the data.
- **3-GPU parallel**: SFT uses DDP; GRPO uses rollout-parallel sampling; evaluation shards across GPUs.

---

## Experiment Design (3 arms)

| Arm | Backbone | Trainable | Pipeline | Input |
|---|---|---|---|---|
| A | OpenVLA-7B | LoRA (LLM, r=16) + full-fine-tuned native head | SFT → GRPO | image + text |
| B2 ⭐ | MiniCPM-o 4.5 | LoRA (r=8, α=16) + discrete action head | SFT → GRPO | image + text |
| B3 | MiniCPM-o 4.5 | warm-start from B2 SFT | speech SFT → GRPO | image + **speech-only** |

**Research questions**:
- **A vs B2** — headline: native VLA vs Omni + action head (both fully post-trained, text input).
- **B2 vs B3** — text vs. speech language channel (B3 warm-started from B2's text SFT, speech-only input).

**Evaluation**: ID / OOD success rate (10 seeds × 34 episodes) + a 2×2 cross-modal matrix (B2-on-speech / B3-on-text zero-shot) + robustness instructions.

---

## Method

### Action head (shared by both families)

- **Discrete, 256 bins** in [−1,1] per action dimension → 7×256-way categorical.
- **A (OpenVLA)**: the native head = the **last 256 columns of the `lm_head`** logits (bin order reversed vs vocab → flipped in `_head_logits`). Fully fine-tuned; only those 256 columns receive gradients (the CE loss uses just them), so the pre-trained text-vocab columns are untouched.
- **B (MiniCPM-o)**: a from-scratch `DiscreteActionHead` `Linear(4096→512)→LN→GELU→…→Linear(256→7×256)`.
- Rollout decodes `argmax` (eval, temp=0) or categorical sampling (training, temp=1.0); GRPO uses the categorical log-probability, matching OpenVLA.

### Stage 1: SFT (Supervised Fine-Tuning)

Loss = **cross-entropy** on the quantized demonstration action (7×256 bins), identical for both families. (The `sft.loss: mse` config field is legacy — the trainer uses CE whenever the head is discrete, which is always the case here.)

| Component | Setting |
|---|---|
| Loss | cross-entropy on 7×256 bins (`expected_action` MSE logged as a diagnostic) |
| Optimizer | AdamW, two param groups: **action head 1e-4**, **LoRA 2e-5**, weight decay 1e-4 |
| Trainable | LoRA (A: q/k/v/o_proj; B: LLM LoRA) + **full action head** |
| Chunking | full-coverage: each trajectory split into `steps_per_traj=4` chunks → every frame seen each epoch |
| Validation | `val_chunk_size=8` chunks (no gradients) |
| Precision | bf16 autocast + gradient checkpointing |
| Epochs / early stop | `max_epochs` (default 2, current runs 4); patience-1 early stop on val loss |
| Checkpoints | `best` (lowest val loss → GRPO start), `last`, `epoch_N` (versioned, never overwritten) |

**3-GPU DDP** (see [Multi-GPU](#multi-gpu)): `DistributedDataParallel` + `DistributedSampler`, with `_set_static_graph()` (gradient checkpointing triggers reentrant backward; the SFT graph is static across iterations).

### Stage 2: GRPO (Group-Relative Policy Optimization)

```
loop until total_env_steps:
  1. rollout  rollout_per_group × group_size trajectories (16) in CALVIN —
              each group rolls out the SAME instruction group_size times
              (textbook GRPO; the relative baseline compares responses to one task)
  2. reward   per-trajectory reward (see below)
  3. advantage  A = (R - mean(group)) / (std(group) + ε)   # no separate critic
  4. update   clipped policy loss + KL penalty vs frozen SFT reference
  5. every eval_every_episodes env steps: deterministic eval + checkpoint
```

| Hyperparameter (locked across all groups) | Value |
|---|---|
| group_size / rollout_per_group | 8 / 2 (16 trajectories per update; one instruction per group) |
| clip ε / KL β | 0.2 / 0.05 |
| lr action head / LoRA | 1e-4 / 2e-5 |
| max grad norm | 1.0 |
| total_env_steps / max_traj_len | 20000 / 60 |
| eval_every_episodes | 2000 |

**Objective** (per-chunk, immediate backward so the autograd graph is freed — the 32 GB card OOMs otherwise):

```
ratio   = exp(logp_policy - logp_old)
loss    = -min(ratio·A, clip(ratio, 1±0.2)·A) + 0.05·KL(policy ‖ reference)
```

**Reward** (fixed for all groups):

```
R = 10·success + 0.5·R_dist + 0.1·R_action
  R_success = +10 on full task success
  R_dist    = -L2(ee → target); CALVIN exposes no per-step target distance, so
              a subtask-progress proxy -(1-progress)·10 is used (identical across
              groups → relative comparisons stay valid)
  R_action  = -mean L2 jitter between consecutive actions
```

**Dual-phase offload**: policy (~15–19 GB) and frozen reference (~15–19 GB) can't coexist on a 32 GB card. Reference log-probs are computed first (no-grad) with the reference on GPU / policy on CPU, then swapped for the policy gradient phase.

**3-GPU rollout-parallel** (`train_grpo_dist.py`): each rank holds a policy replica and samples ~6 trajectories in parallel; rank 0 gathers them, computes advantages + the update, and broadcasts the trainable weights back. (The parallelized stage is **env sampling** — the GRPO bottleneck — not gradient computation; the update stays on rank 0.)

### Multi-GPU

| Stage | Command | Parallelism |
|---|---|---|
| SFT | `torchrun --nproc_per_node=3 scripts/train_sft.py --group X --epochs N` | DDP (gradient) |
| GRPO | `torchrun --nproc_per_node=3 scripts/train_grpo_dist.py --group X` | rollout-parallel (sampling) |
| Eval | `python scripts/evaluate.py … --shard K 3` (×3) | shard the episode list |

---

## Data Preparation

Full reproducible pipeline in [`docs/data_preparation.md`](docs/data_preparation.md). Summary:

**CALVIN selective extraction** (the official `task_ABC_D.zip` is 517 GB / 1.9 M step files — never downloaded whole):

```bash
# 1. index the zip central directory (229 MB, resumable)
python scripts/fetch_zip_index.py \
    --url http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip \
    --out data/calvin/task_ABC_D.index.json
# 2. stream-filter metadata (episode ids, lang annotations, scene mapping)
python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
    --include 'task_ABC_D/training/ep_start_end_ids.npy' \
    --include 'task_ABC_D/training/lang_annotations/auto_lang_ann.npy' ... \
    --out data/calvin/meta.index.json
python scripts/selective_extract.py --index data/calvin/meta.index.json --all --out-dir data/calvin/meta
# 3. window-level sampling plan (env A, 64-step windows, stride 12)
python scripts/plan_calvin_subset.py \
    --training-meta data/calvin/meta/task_ABC_D/training \
    --validation-meta data/calvin/meta/task_ABC_D/validation \
    --env A --train-window-stride 12 --val-window-stride 32
# 4. extract only the chosen step files (HTTP Range, deflate-decompress, resumable)
python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
    --episode-list data/calvin/plan.json --out data/calvin/steps.index.json
python scripts/selective_extract.py --index data/calvin/steps.index.json --all --out-dir data/calvin/raw
```

**Preprocess → unified per-window npz** (`<root>/<split>/<traj_id>.npz`):

```python
rgb: (T,3,224,224) uint8   # static camera
actions: (T,7) float32 [-1,1]   # already normalized in the dataset (pos ÷50, rot ÷20, gripper ±1)
lang: str                   # instruction for the window
init_robot_obs / init_scene_obs   # recorded start state (official reset protocol)
```

```bash
python scripts/preprocess_data.py --dataset calvin --raw data/calvin/raw/task_ABC_D \
    --plan data/calvin/plan.json --lang-train .../training/lang_annotations/auto_lang_ann.npy \
    --lang-val .../validation/lang_annotations/auto_lang_ann.npy
# → sft_train(85%) / sft_val(15%) / id_test, with leak-proof manifests
```

**B3 speech (CSI)**: synthesize TTS for each instruction (`edge-tts` → resample 16 kHz), keyed by `sha256(instruction)[:16]`:

```bash
python scripts/preprocess_csi_tts.py --calvin-root data/calvin --csi-root data/csi --split sft_train
```

**Leak-proof splits** (`src/data/splits.py`, hard guarantee): `sft_train ∩ sft_val = ∅`, `id_test ∩ (sft_train ∪ sft_val) = ∅`; re-validated on every loader instantiation.

```
calvin_sft_train(~416)  calvin_sft_val(~76)  calvin_grpo_eval(16)  calvin_id_test(34)  robust_instructions(34)
```

---

## Evaluation Mechanism

### Success detection

CALVIN's raw env reports only `{robot_info, scene_info}` — no success signal. The wrapper (`src/envs/calvin_env.py`) computes success by comparing the **episode-start** `scene_info` against the **current** `scene_info` via the official `Tasks` oracle (`new_playtable_tasks.yaml`): the episode succeeds iff the instruction's `task` subtask is completed.

### Metrics

`success_rate` (primary), `mean_return`, `mean_steps`, `action_smoothness` (mean L2 between consecutive actions), `mean_ee_error`.

### Splits

| Split | Episodes | Scene | Note |
|---|---|---|---|
| `id` | 34 | `calvin_scene_A` | official held-out val windows (never trained on) |
| `ood` | 34 | `calvin_scene_D` | same instructions, rendered in env D (ABC→D protocol) |
| `robustness` | 34 | `calvin_scene_A` | paraphrased/reordered/long instructions |
| `csi` | 16 | `calvin_scene_A` | cross-modal 2×2 matrix (`--modality native/audio/text`) |

### Protocol

- Each episode is run **10 seeds** (temperature 0, deterministic) → mean ± std.
- Episodes reset to their **recorded start state** (`init_robot_obs`/`init_scene_obs`).
- `--shard K N` slices the episode list `instructions[K::N]` so one eval can be spread across N GPUs; merge shards by weighting each metric by `n_instructions`.

### Results ledger

Every eval appends to `runs/results.jsonl` (append-only, never overwritten): aggregate metrics **plus** the per-instruction breakdown (`task`, `instruction`, per-seed metrics). `scripts/report_results.py` reads the ledger and prints a per-group aggregate table and a per-task success-rate table:

```bash
python scripts/report_results.py                 # id/native
python scripts/report_results.py --split ood
python scripts/report_results.py --modality audio
```

---

## Quick Start

```bash
# 0. smoke-test gate (18 checks, 3 GPUs in parallel)
bash scripts/run_smokes.sh

# 1. SFT (3-GPU DDP, 4 epochs; early-stop on val loss)
torchrun --nproc_per_node=3 scripts/train_sft.py --group a1 --epochs 4   # A
torchrun --nproc_per_node=3 scripts/train_sft.py --group b2 --epochs 4   # B2
# B3 warm-start from B2's text SFT:
python scripts/train_sft.py --group b3 --init-from runs/B2/sft/best

# 2. evaluate the lowest-val-loss SFT checkpoint (3-card shard)
python scripts/evaluate.py --group b2 --split id --ckpt runs/B2/sft/best --shard 0 3 --device cuda:0 &
python scripts/evaluate.py --group b2 --split id --ckpt runs/B2/sft/best --shard 1 3 --device cuda:1 &
python scripts/evaluate.py --group b2 --split id --ckpt runs/B2/sft/best --shard 2 3 --device cuda:2 &

# 3. GRPO (3-GPU rollout-parallel)
torchrun --nproc_per_node=3 scripts/train_grpo_dist.py --group b2

# 4. final evaluation (auto-picks GRPO best > SFT best)
python scripts/evaluate.py --group b2 --split id
python scripts/evaluate.py --group b2 --split ood
python scripts/evaluate.py --group b2 --split csi --modality audio   # B2 zero-shot on speech
python scripts/evaluate.py --group b3 --split csi --modality text    # B3 zero-shot on text
```

The end-to-end sequence is automated by `scripts/orchestrator.py` (A SFT → A best eval → B2 SFT → B2 best eval → B2 GRPO), restart-safe via `runs/orchestrator_state.json`.

---

## Repository Layout

```
configs/           # base.yaml (shared, locked) + group/<arm>.yaml overrides
src/
  models/          # OpenVLA / MiniCPM-o wrappers + action heads
  training/        # SFT / GRPO trainers, reward function
  envs/            # CALVIN wrapper + Tasks-oracle success detection
  rollout/         # HF / vLLM inference engines
  eval/            # evaluation protocol (ID/OOD/robustness/2×2 matrix)
  data/            # loaders, leak-proof splits, CSI speech dataset
  utils/           # ckpt (head-compat), instruction sampler, logger
scripts/           # train/eval/data-prep/diagnostic/orchestration
docs/              # experiment plan, data preparation, limitations, smoke tests
```

## Environment

- torch 2.5.1+cu124, transformers (hf-mirror), `calvin_env` (pybullet, headless EGL)
- 3×32 GB vGPUs; HF engine for all groups (vLLM optional for OpenVLA)
- CALVIN env-A 64-step language windows (stride-12, ~500 windows); B3 uses CSI (synthesized TTS speech). BridgeData V2 is **not** part of the main comparison (CALVIN-only).

## References

[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) · [OpenVLA](https://github.com/openvla/openvla) · [CALVIN](https://github.com/mees/calvin) · [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
