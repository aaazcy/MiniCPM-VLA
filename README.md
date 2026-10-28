# MiniCPM-VLA

Post-training the multimodal speech model **MiniCPM-o 4.5** into a robot-manipulation policy (VLA): attach a 7-DoF action head, run two-stage post-training (**SFT behavior cloning + GRPO reinforcement learning**), and evaluate on the **CALVIN** benchmark against a native VLA baseline (**OpenVLA-7B**) to verify **"directly controlling a robot arm from spoken instructions"**.

> **Status**: end-to-end pipeline verified (CE SFT → GRPO → CALVIN success detection). Training and evaluation in progress.

## Highlights

- **Omnimodal VLA**: image + text + speech (Whisper audio tower) inputs → 7-DoF action output (end-effector pose + gripper, [−1,1]); both families use a discretized 256-bin action head trained with cross-entropy (matching OpenVLA's native bin parameterization).
- **Two-stage post-training**: SFT (cross-entropy behavior cloning) → GRPO (group-relative policy optimization with KL constraint).
- **Parameter-efficient**: LoRA adapters on the LLM + a **fully fine-tuned action head** (OpenVLA's native `lm_head` is unfrozen so it can adapt to CALVIN); fits a single 32 GB GPU.
- **CALVIN success detection implemented**: the raw CALVIN env only reports `{robot_info, scene_info}`, so success was previously *always False*. We compute it via the official `Tasks` oracle (compare start vs current `scene_info`) and store the subtask name in the data.

## Method

### Stage 1: SFT (Supervised Fine-Tuning)

Loss = cross-entropy on a discretized action head (7×256 uniform bins) for both MiniCPM-o and OpenVLA (OpenVLA's native bin head is also trained with CE on its action tokens, and is **fully fine-tuned** rather than frozen). AdamW (action head lr 1e-4 / LoRA lr 2e-5); 2–4 epochs with full-coverage chunked sampling + patience-1 early stopping on CALVIN SFT-val loss; bf16 with gradient checkpointing. Per-epoch checkpoints (`epoch_N`) are saved so the best-by-ID-success epoch can be picked for GRPO.

### Stage 2: GRPO (Reinforcement-Learning Post-Training)

Roll out the policy in CALVIN → group-relative advantages → clipped policy loss + KL penalty against the frozen SFT reference model.

| Hyperparameter (locked across all groups) | Value |
|---|---|
| Group structure / clip ε / KL β | 4 groups × 4 trajectories / 0.2 / 0.05 |
| Reward | R = R<sub>success</sub> + 0.5·R<sub>dist</sub> + 0.1·R<sub>action</sub> |
| Total steps / evaluation | 20k env steps / deterministic eval every 2000 episodes |
| Backprop scope | LoRA + action head only |

## Experiment Design (3 arms)

| Arm | Backbone | Trainable | Pipeline | Input |
|---|---|---|---|---|
| A | OpenVLA-7B | LoRA (LLM, r=16) + full-fine-tuned native head | SFT → GRPO | image + text |
| B2 ⭐ | MiniCPM-o 4.5 | LoRA (r=8, α=16) + action head | SFT → GRPO | image + text |
| B3 | MiniCPM-o 4.5 | warm-start from B2 SFT | speech SFT → GRPO | image + **speech-only** |

**Research questions**: **A vs B2** (headline — native VLA vs Omni, text); **B2 vs B3** (text vs. speech language channel, B3 warm-started from B2's text SFT).

**Evaluation**: ID / OOD success rate (10 seeds × 34 episodes, mean ± std) + the 2×2 cross-modal matrix (B2-on-speech / B3-on-text zero-shot).

## Quick Start

```bash
# 1. Smoke-test gate (3 GPUs in parallel)
bash scripts/run_smokes.sh

# 2. Full training
torchrun --nproc_per_node=3 scripts/train_sft.py --group a1 --epochs 4        # A (3-GPU DDP)
python scripts/train_sft.py --group b2 --device cuda:1                         # B2
python scripts/train_sft.py --group b3 --init-from runs/B2/sft/best --device cuda:2  # B3 warm-start

# 3. Evaluation (mainline + 2×2 cross-modal matrix)
python scripts/evaluate.py --group b2 --split id
python scripts/evaluate.py --group b2 --split csi --modality audio   # B2 zero-shot on speech
python scripts/evaluate.py --group b3 --split csi --modality text    # B3 zero-shot on text
```

## Repository Layout

```
configs/           # training hyperparameters (base.yaml locks shared config) + group configs
src/
  models/          # OpenVLA / MiniCPM-o wrappers + action heads
  training/        # SFT / GRPO trainers, reward functions
  envs/            # CALVIN environment (headless) + success detection (Tasks oracle)
  rollout/         # HF inference engine
  eval/            # evaluation protocol (ID/OOD/robustness/2×2 matrix)
scripts/           # training, eval, data-preprocessing, and diagnostic/orchestration scripts
docs/              # experiment plan, limitations
```

## Environment

- torch 2.5.1+cu124, transformers (hf-mirror)
- 3×32 GB vGPUs; HF engine only (no vLLM dependency)
- CALVIN env-A 64-step language windows (stride-12 subsample, ~500 windows); B3 uses CSI (synthesized TTS speech). BridgeData V2 is **not** part of the main comparison (CALVIN-only).

## References

[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) · [OpenVLA](https://github.com/openvla/openvla) · [CALVIN](https://github.com/mees/calvin) · [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
