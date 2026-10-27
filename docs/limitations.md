# Limitations & fairness notes (must-read before interpreting results)

1. **Different vision backbones.** A groups use OpenVLA-OFT's DINOv2+SigLIP vision
   encoder; B groups use MiniCPM-o's SigLip2. Visual representations are not matched;
   any gap cannot be attributed purely to "VLA vs Omni" architecture. Both receive the
   identical single static-camera 224×224 input.

2. **Synthesized speech for B3.** All B3 audio is TTS-generated from the text
   instructions (edge-tts primary, espeak-ng fallback), frame-aligned to the 8-frame
   CALVIN window. It is *not* human natural speech; conclusions about audio benefit do
   not transfer to real human commands without additional validation.

3. **Sim2real gap.** Everything is evaluated in the CALVIN PyBullet simulator. No claims
   are made about real-robot transfer.

4. **Simulation stochasticity.** PyBullet physics, gripper contact and object scattering
   are stochastic; we report mean ± std over 20 seeded repeats, but cross-run variance
   remains.

5. **Trajectory filtering is heuristic.** The "bad collision/failure" filter is
   disabled by default (the original stall/jitter heuristics rejected most CALVIN
   teleop windows). If manual curation (collision/failure labels) becomes available it
   can be re-enabled in `preprocess_data.py`; it is not human-labeled.

6. **Single-view input.** Only the CALVIN static camera is used (224×224) so both models
   see identical input; the gripper camera and proprio are dropped. OpenVLA's native head
   normally receives proprio — dropping it slightly handicaps A groups; deliberate
   controlled-comparison choice, disclosed here.

7. **Post-training data scale.** The CALVIN env-A training subset is ~500 64-step
   language windows (stride 12, a few percent of the full 517 GB dataset). BridgeData V2
   is **not** used in the main comparison. Results are about the *post-training
   comparison* between paradigms under identical data, not full-scale VLA performance.
   Absolute success rates should not be compared to papers trained on the full dataset.

8. **"calvin_A" identification.** There is no published A-only zip; env-A episodes are
   identified via the official `scene_info.npy` step ranges (`A = [1191339, 1795044]`,
   from `task_ABC_D_scene_info.zip`). The mapping must be sanity-checked in Phase 1
   (e.g., rendering a few sampled episodes in `calvin_scene_A`).

9. **MiniCPM-o 4.5 checkpoint availability.** The wrapper is written generically
   (trust_remote_code); set `model_id` in `configs/group/b*.yaml` to the exact HF id or
   a local path. Verified fallback id for the code path: `openbmb/MiniCPM-o-2_6`.

10. **OpenVLA-OFT checkpoint id.** `openvla/openvla-7b-oft` must be verified in Phase 1;
    fallback `openvla/openvla-7b` (same code path).

11. **B3 is an independent ablation only.** It is never pooled with or compared
    head-to-head against group A in any plot or table.

12. **vLLM coverage.** vLLM is used for vision-only rollout; B3 uses the HF engine (stock
    vLLM lacks the Whisper audio path for MiniCPM-o). Inference latencies differ between
    B3 and other groups; wall-clock comparisons should account for this.

13. **Reward observability.** CALVIN does not expose ground-truth end-effector→target
    distance for every subtask; `R_dist` uses the distance when the env provides it,
    otherwise a subtask-progress proxy. Identical for all groups, so relative
    comparisons hold.

14. **Language-model generation gap.** OpenVLA-OFT's LLM is Llama2-based (2023);
    MiniCPM-o uses Qwen3 (2025). This gap is part of what the experiment measures, but
    reviewers may attribute language-dependent failure differences to it (see the
    failure-type pie chart for evidence).

15. **Open-loop demo-replay fidelity is ~35%, and is NOT a model ceiling.** Replaying the
    recorded CALVIN demonstrations *open-loop* — reset to the recorded initial state and
    re-apply the recorded relative actions — succeeds on only ~35% of `id_test` windows
    (`scripts/oracle_replay.py`): coarse tasks (open/close drawer, stack/unstack,
    lightbulb) succeed, while contact-precise tasks (turn_on_led, rotate, lift, push) fail.
    This is **not** a success-detection bug: replaying a failed `turn_on_led` demo leaves the
    LED `logical_state` at 0 across all 65 steps (the gripper never presses the button),
    while an `sft_train` counterpart of the same task toggles it — so the oracle is correct.

    The cause is reset/execution drift. CALVIN stores only positions/orientations
    (`robot_obs` 15-d, `scene_obs` 24-d) with **no velocities and no contacts**, so a reset
    restores positions to ~0.1 mm but the physics re-settles; the actions are *relative*
    deltas, so open-loop replay compounds the per-step error (the classic compounding-error
    / DAgger problem, Ross et al. 2011). This is a known, benchmark-wide property: a recent
    CALVIN case study reports exact-reset teacher replay at 14/14 (100%) vs 0/14 under a
    neutral (symbolic) reset — i.e. demo-replay success is dominated by reset fidelity, not
    by the demonstrations themselves. See *"Benchmark-path fidelity in parallel
    robot-policy evaluation: a CALVIN case study", The Journal of Supercomputing (2026)*.

    Crucially, this does **not** cap our policy's success: the model is evaluated
    **closed-loop** (observes → acts → corrects drift each step), so it is unaffected by
    open-loop replay drift. We also verified the reset *mechanism* is not the cause: an A/B
    replay of a scene-A `turn_on_led` demo with `reset_from_storage` (full serialized state
    incl. joint velocities, reconstructed via `serialize()` after reset) still leaves the LED
    untoggled (0 transitions), identical to the plain `reset(robot_obs, scene_obs)` path
    (`scripts/diag_reset_ab.py`). The paper's 100% is therefore attributable to task/scene
    robustness (scene D), not to a reset source or mechanism we lack.
