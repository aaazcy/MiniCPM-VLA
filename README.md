# MiniCPM-VLA

将多模态语音大模型 **MiniCPM-o 4.5** 改造为机械臂操控策略(VLA):新接 7 维动作头,分两阶段后训练(**SFT 行为克隆 + GRPO 强化学习**),在 **CALVIN** 基准上完成 6 组对照实验(OpenVLA 基线 vs Omni 模型、语音 vs 文本输入消融),验证"**直接听语音指令操控机械臂**"。

> **当前版本:冒烟测试版(Smoke-Test Release)** — 18 项冒烟门禁全部通过,训练流程可一键开训,实验结果待产出。

## 核心特性

- **全模态 VLA 化**:图像 + 文本 + 语音(Whisper 音频塔)三通道输入 → 7 维连续动作输出([−1,1] 末端位姿 + 夹爪)
- **两阶段后训练**:SFT(MSE 行为克隆)→ GRPO(组相对策略优化 + KL 约束)
- **参数高效**:仅微调 LoRA 与动作头,32GB 单卡可训;策略/参考模型双阶段换卡
- **严格对照**:6 组实验 + 语音/文本 2×2 交叉评测矩阵,数据泄漏防护(sha256 清单)

## 技术方案

### 阶段一:SFT(监督微调)

损失 = MSE(预测动作, 演示动作);AdamW(动作头 lr 1e-4 / LoRA lr 2e-5);2 epochs,patience 1 早停;bf16 + 梯度检查点。

### 阶段二:GRPO(强化学习后训练)

策略在 CALVIN 环境 rollout → 组内相对优势 → 裁剪策略损失 + KL 惩罚 vs 冻结的 SFT 参考模型。

| 超参(全组锁定) | 值 |
|---|---|
| 组结构 / 裁剪 ε / KL β | 4 组 × 4 轨迹 / 0.2 / 0.05 |
| 奖励 | R = R任务 + 0.5·R距离 + 1.0·R碰撞 + 0.1·R动作 |
| 总步数 / 评估 | 20k 环境步 / 每 200 回合确定性评估(温度 0) |
| 反向范围 | 仅 LoRA + 动作头 |

## 实验设计(6 组)

| 组 | 基座 | 可训参数 | 流程 | 输入 |
|---|---|---|---|---|
| A0 | OpenVLA-7B | LoRA + 原生动作头 | 仅 SFT | 图像+文本 |
| A1 | OpenVLA-7B | 同上 | SFT→GRPO | 图像+文本 |
| B0 | MiniCPM-o 4.5 | 仅动作头(骨干冻结) | 仅 SFT | 图像+文本 |
| B1 | MiniCPM-o 4.5 | 仅动作头 | SFT→GRPO | 图像+文本 |
| B2 ⭐ | MiniCPM-o 4.5 | LoRA(r=8,α=16)+动作头 | SFT→GRPO | 图像+文本 |
| B3 | MiniCPM-o 4.5 | 同 B2 | SFT→GRPO | 图像+**语音-only** |

**8 个研究问题**:A1 vs B2(主线)、A0 vs B0、A1 vs B1、A0 vs A1、B0 vs B1、B1 vs B2、B2 vs B3(文字 vs 语音)、2×2 交叉迁移(B2 测语音 / B3 测文本,零样本)。

**评测**:ID / OOD 成功率(10 种子 × 20 回合,±std)+ 鲁棒性指令 + 50 条失败轨迹定性分析。

## 快速开始

```bash
# 1. 冒烟测试门禁(三卡并行)
bash scripts/run_smokes.sh

# 2. 单组冒烟
python scripts/train_sft.py   --group b2 --smoke 5 --device cuda:1
python scripts/smoke_grpo.py  --group b2 --device cuda:1

# 3. 正式训练(三卡排程,GPU0: A0→A1 | GPU1: B2 | GPU2: B0→B1→B3)
VLA_DEVICE=cuda:0 bash scripts/run_group.sh a0 a1
VLA_DEVICE=cuda:1 bash scripts/run_group.sh b2
VLA_DEVICE=cuda:2 bash scripts/run_group.sh b0 b1 b3

# 4. 评测(主线 + 2×2 交叉矩阵)
python scripts/evaluate.py --group b2 --split id
python scripts/evaluate.py --group b2 --split csi --modality audio   # B2 零样本听语音
python scripts/evaluate.py --group b3 --split csi --modality text    # B3 零样本读文本
```

## 冒烟测试状态 ✅

| 类别 | 状态 |
|---|---|
| 19 项 CPU 冒烟(配置/泄漏/奖励/数据) | ✅ |
| A0/A1 SFT、A1 GRPO | ✅ |
| B0/B1/B2/B3 模型验证(前向/反向/logp/生成,B3 含音频) | ✅ |
| B0/B1/B2 SFT、B3 SFT(语音-only) | ✅ |
| B1/B2 GRPO、B3 GRPO(语音 rollout) | ✅ |

## 项目结构

```
configs/           # 训练超参(base.yaml 锁定)+ 6 组实验配置
src/
  models/          # OpenVLA / MiniCPM-o 包装 + 动作头
  training/        # SFT / GRPO 训练器、奖励函数
  envs/            # CALVIN 环境(headless)
  rollout/         # HF 推理引擎
  eval/            # 评测协议(ID/OOD/鲁棒/2×2 矩阵)
scripts/           # 训练、冒烟、评测、数据预处理脚本
docs/              # 实验方案、冒烟矩阵、数据管线、局限性
```

## 环境

- torch 2.5.1+cu124,transformers,hf-mirror 镜像
- 3×32G vGPU;HF 引擎(不依赖 vLLM)
- CALVIN env-A 窗口 + Bridge V2(1995 段);B3 用 CSI(合成 TTS 语音)

## 引用

[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) · [OpenVLA](https://github.com/openvla/openvla) · [CALVIN](https://github.com/mees/calvin) · [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
