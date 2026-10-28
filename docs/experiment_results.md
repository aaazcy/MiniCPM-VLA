# 实验结果与结论总结（当前进度）

> 更新日期：2026-10-28。涵盖 A1（OpenVLA）与 B2（MiniCPM-o）的 SFT/评测/GRPO 现状、关键 bug 修复与结论。

## TL;DR

- **A1 的塌缩 bug 已定位并修复**：根因是 wrapper 没给 `generate()` 传 `pixel_values`（视觉被静默绕过），另有 transformers ≥4.50 的 slice 次因。修复后 `action_smoothness` 0.0 → 1.261，模型会动了。
- **B2 SFT 真实成功率 11.76%**（4/34，集中在 unstack/close_drawer/rotate_blue 三个简单任务）。
- **B2 GRPO 暂无效果**：rollout 成功率 0%（温度 1.0 采样 + 60 步截断把 SFT 的脆弱成功打没了），21 次 update 没学到东西。
- **A1 完整评测尚未跑**（只做了快速探针），正式成功率待定。

---

## 1. A1（OpenVLA-7B）结果

### SFT
- 3 个 epoch（早停于 epoch 2 平台），best = epoch 1。
- 轨迹：val_loss 3.657 → **3.634** → 3.753；val_mse 0.062 → 0.057。
- 训练配置：`In: ... Out:` 官方 prompt 格式 + 255 中心 + 解冻 lm_head。

### 塌缩 bug（已修复，两层根因）
| 层级 | 根因 | 修复 |
|---|---|---|
| **主因** | wrapper `_generate_action`/`sample_action` 调 `generate()` 时**漏传 `pixel_values`** → 视觉从未被消费 → 恒定动作 | `ebd44fa`：给 generate() 传 pixel_values |
| 次因 | transformers ≥4.50 把空 DynamicCache 当非 None，`prepare_inputs_for_generation` 把 prompt 切成 1 token | 改本地 model + transformers 缓存 |

- 修复验证（黑图/白图回归测试）：视觉 backbone 调用 2 次、两图产生**不同动作** ✅。

### 快速探针（5 指令 × 3 seeds）
```
action_smoothness = 1.261（修复前 0.0，塌缩已解决）
success_rate      = 0.0（5 条任务 0 成功）
mean_steps        = 100.0
```
- 结论：视觉通了、模型会动了，但 **SFT-only + ~500 窗口**下成功率仍为 0（预期内）。

---

## 2. B2（MiniCPM-o 4.5）结果

### SFT
- 4 个 epoch（未早停），best = epoch 3。
- 轨迹：val_loss 3.931 → 3.887 → 3.866 → **3.865**；val_mse 0.152 → 0.128。
- 训练配置：chat template 格式 + 255 中心（与 OpenVLA 对齐）。

### 评测（ID，34 指令 × 10 seeds）
```
success_rate      = 11.76%（4/34）
action_smoothness = 0.354（非塌缩）
mean_steps        = 92.6
```
分任务成功（均 10/10 seeds，真实成功、非假阳性）：

| 任务 | 窗口 | 步数 | 平滑度 |
|---|---|---|---|
| unstack_block | 2/2 | 16 / 22 | 0.40 / 0.54 |
| close_drawer | 1/2 | 83 | 0.41 |
| rotate_blue_block_left | 1/2 | 27 | 0.41 |

- 抓取类（lift）、精细旋转（rotate_red/pink）、推块（push）、灯（led/lightbulb）全部 0%。

### GRPO（21 updates / 20160 env steps，约 2 小时）
```
rollout success_rate = 0.0（全程 0）
mean_reward          ≈ -305（恒定失败）
```
- 失败原因：rollout 用 temperature=1.0 采样 + max_traj_len=60 截断，把 SFT 的 11.76%（argmax 确定性）打成了 0%（采样+截断），稀疏 reward 下无正信号可学。
- 注：B2 是 MiniCPM-o，走独立 DiscreteActionHead 单次前向，**不涉及 A 的视觉 bypass bug**；B2 的 GRPO 0% 是「探索温度/稀疏 reward」问题，不是视觉问题。

---

## 3. 关键结论

1. **A1 塌缩根因是「训练/推理路径不一致」**：SFT teacher-force 传了 `pixel_values`，eval/rollout 的 `generate()` 没传 → 视觉被静默绕过。教训：VLM「训练正常、评测塌缩」时，先查「模型到底看到了什么输入」，而非先怀疑权重或 prompt。
2. **B2 是当前唯一有真实成功率的组**（11.76%），且成功集中在 CALVIN 简单任务。
3. **GRPO 在两种情况下都还没跑通**：A 之前是视觉 bug（rollout 看不到图），B2 是探索温度太高（11.76%→0%）。GRPO 需要「更温和的探索 + 更长的 rollout 步数」或「先提高 SFT 成功率」。
4. **动作头设计**：官方/社区（MiniCPM-RobotManip、StarVLA）都走 DiT 流匹配 + action chunking + streaming context，而非单次前向 MLP 或自回归（见 `docs/action_head_design.md`）。

---

## 4. 关键改动记录

| commit | 内容 |
|---|---|
| `ebd44fa` | **A 塌缩主因**：generate() 补传 pixel_values |
| `107ba82` | bug 复盘文档（transformers ≥4.50 slice） |
| `ab7233d` | B 的 bin 对齐 OpenVLA：256→255 中心 |
| `3616f1e` | prompt 格式：A 的 `In: Out:` + B 的 chat template |
| `bbcf3f9` | GRPO：group_size 4→8 + 同指令组 |
| `d070a84` | 结果账本 results.jsonl + 分任务报告 |
| `9721528` | 解冻 lm_head（后证明非根因） |

---

## 5. 下一步

1. **A1 完整评测**（34 指令，~2h），拿正式成功率（是否与 B2 同量级）。
2. **A1 的 GRPO**：视觉修好后，rollout 现在能真正看到图，GRPO 才有意义。
3. **B2 的 GRPO 重调**：降 rollout 温度（或提高 max_traj_len、增加 total_env_steps），解决 11.76%→0% 的探索问题。
4. **可选**：参考 MiniCPM-RobotManip 的 DiT 流匹配头 + action chunking，作为动作头升级方向。
