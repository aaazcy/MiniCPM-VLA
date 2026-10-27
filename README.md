# MiniCPM-VLA

Post-training the multimodal speech model **MiniCPM-o 4.5** into a robot-manipulation policy (VLA): attach a 7-DoF action head, run two-stage post-training (**SFT behavior cloning + GRPO reinforcement learning**), and evaluate on the **CALVIN** benchmark with 6 controlled groups (OpenVLA baseline vs. Omni model; speech vs. text input ablation) to verify **"directly controlling a robot arm from spoken instructions"**.

> **Current release: Smoke-Test Release** — all 18 smoke-gate checks pass; training pipelines are launch-ready; experimental results pending.

## Highlights

- **Omnimodal VLA**: image + text + speech (Whisper audio tower) inputs → 7-DoF action output (end-effector pose + gripper, [−1,1]); MiniCPM-o uses a discretized 256-bin action head (same parameterization as OpenVLA)
- **Two-stage post-training**: SFT (MSE behavior cloning) → GRPO (group-relative policy optimization with KL constraint)
- **Parameter-efficient**: only LoRA adapters and the action head are trained — fits a single 32 GB GPU; policy/reference models swapped via dual-phase offload
- **Rigorous comparisons**: 6 experiment groups + a 2×2 cross-modal speech/text evaluation matrix; leak-guarded splits (sha256 manifests)

## Method

### Stage 1: SFT (Supervised Fine-Tuning)

Loss = cross-entropy on a discretized action head (7×256 uniform bins, matching OpenVLA) for both MiniCPM-o and OpenVLA groups (OpenVLA's native bin head is also trained with CE on its action tokens). AdamW (action head lr 1e-4 / LoRA lr 2e-5); 2 epochs with full-coverage chunked sampling (every frame seen each epoch) + patience-1 early stopping on CALVIN SFT-val loss; bf16 with gradient checkpointing.

### Stage 2: GRPO (Reinforcement-Learning Post-Training)

Roll out the policy in CALVIN → group-relative advantages → clipped policy loss + KL penalty against the frozen SFT reference model.

| Hyperparameter (locked across all groups) | Value |
|---|---|
| Group structure / clip ε / KL β | 4 groups × 4 trajectories / 0.2 / 0.05 |
| Reward | R = R<sub>success</sub> + 0.5·R<sub>dist</sub> + 0.1·R<sub>action</sub> |
| Total steps / evaluation | 20k env steps / deterministic eval every 200 episodes |
| Backprop scope | LoRA + action head only |

## Experiment Design (6 Groups)

| Group | Backbone | Trainable | Pipeline | Input |
|---|---|---|---|---|
| A0 | OpenVLA-7B | LoRA + native action head | SFT only | image + text |
| A1 | OpenVLA-7B | same | SFT → GRPO | image + text |
| B0 | MiniCPM-o 4.5 | action head only (frozen backbone) | SFT only | image + text |
| B1 | MiniCPM-o 4.5 | action head only | SFT → GRPO | image + text |
| B2 ⭐ | MiniCPM-o 4.5 | LoRA (r=8, α=16) + action head | SFT → GRPO | image + text |
| B3 | MiniCPM-o 4.5 | same as B2 | SFT → GRPO | image + **speech-only** |

**8 research questions**: A1 vs B2 (headline), A0 vs B0, A1 vs B1, A0 vs A1, B0 vs B1, B1 vs B2, B2 vs B3 (text vs. speech language channel), and the 2×2 cross-modal transfer matrix (B2 evaluated on speech / B3 evaluated on text, zero-shot).

**Evaluation**: ID / OOD success rate (10 seeds × 20 episodes, mean ± std) + robustness instructions + qualitative analysis of 50 failed trajectories.

## Quick Start

```bash
# 1. Smoke-test gate (3 GPUs in parallel)
bash scripts/run_smokes.sh

# 2. Single-group smokes
python scripts/train_sft.py   --group b2 --smoke 5 --device cuda:1
python scripts/smoke_grpo.py  --group b2 --device cuda:1

# 3. Full training (3-GPU schedule: GPU0 A0→A1 | GPU1 B2 | GPU2 B0→B1→B3)
VLA_DEVICE=cuda:0 bash scripts/run_group.sh a0 a1
VLA_DEVICE=cuda:1 bash scripts/run_group.sh b2
VLA_DEVICE=cuda:2 bash scripts/run_group.sh b0 b1 b3

# 4. Evaluation (mainline + 2×2 cross-modal matrix)
python scripts/evaluate.py --group b2 --split id
python scripts/evaluate.py --group b2 --split csi --modality audio   # B2 zero-shot on speech
python scripts/evaluate.py --group b3 --split csi --modality text    # B3 zero-shot on text
```

## Smoke-Test Status ✅

| Category | Status |
|---|---|
| 19 CPU smokes (config / leakage / reward / data) | ✅ |
| A0/A1 SFT, A1 GRPO | ✅ |
| B0/B1/B2/B3 model verification (fwd/bwd/logp/generate; B3 incl. audio) | ✅ |
| B0/B1/B2 SFT, B3 SFT (speech-only) | ✅ |
| B1/B2 GRPO, B3 GRPO (speech rollout) | ✅ |

## Repository Layout

```
configs/           # training hyperparameters (base.yaml locks shared config) + 6 group configs
src/
  models/          # OpenVLA / MiniCPM-o wrappers + action heads
  training/        # SFT / GRPO trainers, reward functions
  envs/            # CALVIN environment (headless)
  rollout/         # HF inference engine
  eval/            # evaluation protocol (ID/OOD/robustness/2×2 matrix)
scripts/           # training, smoke, evaluation, data-preprocessing scripts
docs/              # experiment plan, smoke matrix, data pipeline, limitations
```

## Environment

- torch 2.5.1+cu124, transformers (hf-mirror)
- 3×32 GB vGPUs; HF engine only (no vLLM dependency)
- CALVIN env-A 64-step language windows (stride-12 subsample, ~500 windows); B3 uses CSI (synthesized TTS speech). BridgeData V2 is **not** part of the main comparison (CALVIN-only).

## References

[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) · [OpenVLA](https://github.com/openvla/openvla) · [CALVIN](https://github.com/mees/calvin) · [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
