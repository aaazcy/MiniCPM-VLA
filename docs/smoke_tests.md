# 冒烟测试阶段(Smoke Gate)

> 开训门禁:以下冒烟**全部通过**才允许启动正式训练。
> 一键执行:`bash scripts/run_smokes.sh`(三卡并行,约 30-60 分钟)

## 1. 冒烟矩阵

| # | 组 | 类型 | 验证内容 | 验收标准 |
|---|---|---|---|---|
| 1 | A0 | SFT | 真实数据 5 batch | MSE 有限且下降;smoke ckpt 可存 |
| 2 | A1 | SFT | 同上 | 同上 |
| 3 | A1 | GRPO | 环境 + 1 rollout + 1 更新 | traj.length>0;reward/logp/KL 有限 |
| 4 | B0 | 模型 | 前向/反向/logp/generate | 无 OOM;动作 ∈[-1,1] |
| 5 | B1 | 模型 | 同 B0(无 LoRA) | 同上 |
| 6 | B2 | 模型 | 前向/反向/logp/generate(LoRA) | 无 OOM;显存 <30G |
| 7 | B2 | SFT | 真实数据 5 batch | MSE 下降;ckpt 可存 |
| 8 | B2 | GRPO | 环境 + 1 rollout + 1 更新 | 同 #3 |
| 9 | B3 | 模型 | **音频路径**(wav→processor→前向/反向) | 同上 + 音频张量形状正确 |
| 10 | B3 | SFT | CSI 数据(图像+音频)5 batch | MSE 下降;音频 batch 无错 |
| 11 | B3 | GRPO | rollout 与 update **同一 wav** | logp 有限;KL 有限 |

## 2. 手动执行命令(或直接 run_smokes.sh)

```bash
# GPU0
python scripts/train_sft.py --group a1 --smoke 5 --device cuda:0
python scripts/smoke_grpo.py --group a1 --device cuda:0

# GPU1
python scripts/verify_models.py --group b3 --batch 1 --device cuda:1
python scripts/train_sft.py --group b3 --smoke 5 --device cuda:1
python scripts/smoke_grpo.py --group b3 --device cuda:1

# GPU2
python scripts/verify_models.py --group b0 --batch 1 --device cuda:2
python scripts/verify_models.py --group b1 --batch 1 --device cuda:2
python scripts/train_sft.py --group b2 --smoke 5 --device cuda:2
python scripts/smoke_grpo.py --group b2 --device cuda:2
```

## 3. 本环境已踩坑(冒烟阶段修掉的,记录在案)

| 坑 | 修复 |
|---|---|
| `CUDA_VISIBLE_DEVICES` 不可靠 | 一律 `--device cuda:N` |
| CALVIN env 参数名是 `show_gui` | 拼写错误曾导致连 X server |
| vGPU 无可用 EGL 插件 | `use_egl=false` + monkeypatch 强制 `ER_BULLET_HARDWARE_OPENGL` |
| pybullet 返回 int64/(1,1,3) 占位图 | `_process_obs` 防御:uint8 转换 + 经相机对象重渲染 |
| bf16 张量 `.numpy()` 崩溃 | 两个 wrapper 先 `.float()` |
| MiniCPM-o 视觉 14-tile OOM | 输入转 PIL + `max_slice_nums=1`(1024 tokens) |
| 9B 显存不足 | sdpa + grad ckpt + batch 1 |
| PeftModel `.model.embed_tokens` 失效 | Qwen3ForCausalLM 注册别名 |
| 动作头 fp32 × LLM bf16 | ActionHead 输入 cast |

## 4. 结果记录(截至 2026-10-27)

- [x] A0 SFT / A1 SFT / B0 模型 / B2 模型
- [ ] A1 GRPO、B1 模型、B2 SFT/GRPO、B3 模型/SFT/GRPO
