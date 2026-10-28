# Action-head design notes (MLP vs autoregressive vs flow matching)

Background for why B uses a single-pass discrete head, why A is autoregressive, and
what the field's current answer is. Kept here because it shapes how A-vs-B results
should be read (see limitations.md §16).

## 1. The three (really four) action-head families

| Head | Output | How | Speed | Notes |
|---|---|---|---|---|
| Continuous regression (CR) | ℝ^7 single vector | MLP + regression loss | 1 forward | prone to mean-collapse on multi-modal targets |
| Discrete regression (DR) | 7×256 categorical | MLP + CE | 1 forward | **B's current head**; avoids collapse, but independent-per-dim "factorization error" |
| Autoregressive (AR) | 7 tokens sequentially | next-token CE over action vocab | 7 forwards | **A's native head**; captures cross-dim dependence but <2 Hz control |
| Flow matching (continuous / discrete) | sampled from a distribution | iterative denoising/refinement | ~5–20 steps | SOTA direction; captures multi-modality + joint consistency |

## 2. Research findings (2025–2026)

- **Autoregressive is slow** — *Knowledge Insulating VLA* (NeurIPS 2025): AR VLAs run
  **<2 Hz**; π0-FAST takes ~750 ms for a 1 s chunk. OpenVLA-OFT (2025) reports
  *"parallel decoding + action chunking is strictly better than autoregressive
  generation"* (26–43× faster **and** higher success).
- **Naive binning is token-hungry** — *FAST* (2025): per-dim 256-bin tokenization makes
  "hundreds of tokens per chunk", slow training/inference on high-frequency data.
- **Discrete can beat continuous** — *When Do Discrete Actions Make Good Continuous
  Controllers?* (arXiv 2610.20429, 2026): **discrete regression consistently outperforms
  continuous regression**, contradicting the "quantization loses precision" belief.
- **But continuous regression can win on precise single tasks** — a RoboTwin/LIBERO
  ablation (CSCI699) found a Smooth-L1 regression head beats bin/token on a single
  place-shoe task, with lower variance and better robustness to visual blur. Task-
  dependent.
- **Single-pass DR's weakness is factorization error** — the 2026 paper's Finding 2:
  independently predicted tokens can combine into incompatible actions even when each
  marginal is correct. Iterative refinement (discrete flow) fixes this.

## 3. Continuous vs discrete flow matching

Both are *generative* (they sample from a multi-modal action distribution) and both use
*iterative refinement*, unlike single-pass regression. The difference is the state space
and the refinement mechanism.

### Continuous flow matching (π0, RDT)

- State space: **continuous** ℝ^d. Noise source: Gaussian.
- The model learns a **velocity field** v_θ(x_t, t) that transports the Gaussian cloud
  onto the action cloud. Training regresses the velocity; inference integrates the ODE
  `dx/dt = v_θ(x, t)` from t=0 (noise) to t=1 (a sampled action).
- Output is a continuous action directly; arbitrary precision; can represent multi-modal
  distributions (the flow splits into branches).
- Cost: usually a separate "action expert" (smaller than the VLM), decoupled from the
  autoregressive backbone.

### Discrete flow matching (Discrete Diffusion VLA, DFM-VLA)

- State space: **discrete** tokens (the 256 bins). Noise source: [MASK] (or random token).
- Based on discrete flow matching / discrete diffusion (Campbell et al. 2024; Gat et al.
  2024): a continuous-time Markov chain over token states. The model learns which tokens
  to "unmask"/jump at each step.
- Inference: start with all action tokens masked, iteratively predict + re-mask the
  low-confidence ones until convergence ("first easy, then hard"). The fixed-length token
  output fits the VLM's own token space, so it can stay *unified* with the language model.
- Output is discrete tokens → bins → action (quantized, but iterative refinement recovers
  cross-dimension consistency that single-pass DR lacks).

### Side-by-side

| | Continuous flow | Discrete flow |
|---|---|---|
| State | ℝ^d continuous | categorical tokens |
| Noise → data path | ODE velocity field (Gaussian→data) | CTMC transitions ([MASK]→tokens) |
| Output precision | arbitrary | bin-quantized |
| Unification with VLM | decoupled action expert | can share the token space |
| Examples | π0, RDT, SmolVLA | Discrete Diffusion VLA, DFM-VLA |
| Solves DR's factorization error | ✅ (joint denoising) | ✅ (iterative refinement) |

## 4. What this means for this project

1. **Keep B as single-pass DR.** Do not switch to autoregressive — it only adds 7× latency
   and is not clearly better.
2. **A's autoregression is its native design**, and A-vs-B already contrasts AR-token vs
   single-pass-DR action decoding; record this as a confound (limitations.md §16).
3. **If B's success needs a boost**, the principled upgrade is **discrete flow matching**
   (iterative refinement of the same 256-bin tokens) or a **continuous flow action
   expert** — both are drop-in replacements for the DR head, neither touches the backbone.
