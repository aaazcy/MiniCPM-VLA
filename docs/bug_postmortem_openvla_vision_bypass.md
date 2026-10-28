# Bug Postmortem：OpenVLA 恒定动作塌缩（transformers ≥4.50 视觉静默绕过）

> 复盘文档，供后续排查参考。状态：**已修复（推理侧），待重评验证**。

## TL;DR

transformers ≥4.50 的 `generate()` 在**第一步**传的是「空 `DynamicCache`」而非 `None`。OpenVLA 的
`prepare_inputs_for_generation` 用 `if past_key_values is not None` 判断「是否有缓存」，空缓存也判定为
True，于是**第一步就把完整 prompt 切成 1 个 token**，模型进入「只处理最后一个 token」的缓存分支，
**图像（`pixel_values`）被静默丢弃** → 模型对任何输入都输出同样的动作（`action_smoothness = 0`）。

## 1. 症状

- 评测 `action_smoothness = 0.0`（每一步输出相同动作 → 匀速直线运动，不响应环境）
- `success_rate = 2.94%`（1/34，且该条是假阳性：恒定直线运动碰巧触发 oracle）
- 但 SFT `val_mse = 0.055`（训练正常，模型在 teacher-forcing 下预测动作很准）
- **全程无任何报错**（「静默」失败，最难查的一点）

## 2. 排查时间线（走过的弯路）

| 顺序 | 假设 | 改动 | 结果 |
|---|---|---|---|
| 1 | 动作头 `lm_head` 被冻结、学不了 CALVIN | 解冻 `lm_head`（commit `9721528`） | ❌ 塌缩依旧 |
| 2 | 生成路径 `attention_mask` 与追加的 `EMPTY_TOKEN` 没对齐 | 对齐 mask（commit `efbab1d`） | ❌ 塌缩依旧 |
| 3 | prompt 格式错了（缺 `In: ... Out:`） | 补官方格式（commit `3616f1e`） | ❌ 塌缩依旧 |
| 4 | **搜 issue / 查环境版本** | 发现 transformers ≥4.50 已知 bug | ✅ 命中根因 |

**核心教训**：VLM「训练正常、评测塌缩」时，应**先查「模型到底看到了什么输入」**（hook 视觉 backbone
是否被调用、喂黑图/白图看输出是否变化），而不是先怀疑权重（冻结头）或 prompt。

## 3. 根因

### 3.1 背景：自回归生成 + KV 缓存

A（OpenVLA）逐 token 生成 7 个动作 token。为了加速，用 KV 缓存：

- **第 1 步**（无缓存）：读**完整输入**（图像 + prompt），算 attention，把每个 token 的 KV 存进缓存；
- **第 2~7 步**（有缓存）：输入只有**最后 1 个新 token**，但注意力通过缓存「回看」前面所有 token
  （图像、指令、之前的动作）。

所以「输入只有 1 个 token」≠「只看 1 个 token」——信息在缓存里。

### 3.2 transformers ≥4.50 的破坏性改动

- 老版本：第 1 步时 `past_key_values = None`（「无缓存」的空值）。
- 新版本（≥4.50）：第 1 步时 `past_key_values = 一个空的 DynamicCache`（长度 0，**不是 None**）。

### 3.3 OpenVLA 被「骗」的过程

`modeling_prismatic.py` 的 `prepare_inputs_for_generation`（旧代码）：

```python
if past_key_values is not None:      # 本意：「有缓存」=「不是第1步」
    input_ids = input_ids[:, -1:]    # 只留最后一个 token
```

本意：第 1 步（`None`）不切 → 处理完整 prompt + 图像；后续步（有缓存）切 → 只处理新 token。

但 transformers 4.50 之后，第 1 步传的是**空缓存对象**，`is not None` 恒真 → **第 1 步就被切成 1 个
token** → forward 进入 `input_ids.shape[1] == 1` 的缓存分支 → **`pixel_values` 根本没被消费**。

以指令 `"open the drawer"` 为例，喂给 `generate()` 的输入：

```
token:  1     512  29901  1722  278   7482  556   13    3744  29901  29871
含义:   <s>   In    :     open  the   drawer .     \n    Out    :     <empty>
```

- **正常第 1 步**：读全部 11 个 token + 图像 → 模型理解「抽屉关着 + 指令开抽屉」→ 生成动作 token1。
- **bug 第 1 步**：`input_ids[:, -1:]` 把前 10 个 token 全切掉 → 只剩 `[<empty>]` 1 个 token，图像也被
  丢弃 → 模型啥都没看到 → 输出固定的垃圾 token → 后续 6 个 token 也基于它 → **全程恒定动作**。

### 3.4 为什么「静默」

- 空缓存对象 `is not None` 成立（合法对象），代码不报错；
- 切 prompt 是合法操作；
- 1-token 缓存分支本来就是合法路径（给后续步用的）；
- 所以模型「看起来在正常工作」，只是输出恒定。

## 4. 修复

`modeling_prismatic.py` 的 `prepare_inputs_for_generation`，把判断从「是否为 None」改成「缓存是否真的
有内容」：

```python
# 修复前
if past_key_values is not None:
    input_ids = input_ids[:, -1:]

# 修复后
cache_len = 0
if past_key_values is not None:
    if hasattr(past_key_values, "get_seq_length"):
        cache_len = past_key_values.get_seq_length()          # 空缓存返回 0
    elif isinstance(past_key_values, (list, tuple)) and past_key_values:
        cache_len = past_key_values[0][0].shape[2]            # 兼容旧 tuple 格式
if cache_len > 0:
    input_ids = input_ids[:, -1:]        # 真有缓存（后续步）才切
else:
    past_key_values = None               # 空缓存当 None → 走完整 prompt + 图像
```

修改位置（两处都要改，缺一不可）：

1. 本地模型快照：`/root/autodl-tmp/models/openvla-7b/modeling_prismatic.py`
2. transformers 远程代码缓存：`~/.cache/huggingface/modules/transformers_modules/openvla-7b/modeling_prismatic.py`
   （`trust_remote_code=True` 实际加载的是缓存副本；改完记得删对应 `__pycache__/*.pyc` 强制重编译）

## 5. 验证方法（关键回归测试）

官方 issue 给出的黑图/白图测试：**给一张纯黑图和一张纯白图，模型不能输出相同的动作**。

```python
# 加载修复后的模型，对黑图、白图各跑一次 predict_action
# 通过则：两次输出的动作不同（说明视觉生效）
# 失败则：两次输出相同（视觉仍被绕过）
```

## 6. 经验教训

1. **先搜 issue**。这个问题 2026-09 就有人在 HF discussion #8 / GitHub issue #346 报过，症状一字不差
   （"constant action for every image"、"silent vision bypass"）。我绕了冻结头→mask→prompt 三个弯路，
   早搜能省两小时。
2. **查环境版本**。transformers 从 <4.50 升到 4.53 的 breaking change 是根因。装新环境/升级依赖后，
   对「依赖 HF 内部缓存语义的模型」要重点回归。
3. **排查顺序**：VLM 训练/推理不一致，第一步应是「模型到底看到了什么输入」（hook `vision_backbone`
   是否被调用、黑图白图输出是否不同），而不是先动权重或 prompt。
4. **「静默失败」最危险**：这个 bug 不报错、loss 正常、看着像 working policy，只有靠「对输入做扰动
   看输出是否变化」这类探针才能暴露。

## 7. 参考

- HF discussion #8（openvla/openvla-7b）：*Silent vision bypass on transformers ≥4.50: constant action
  for every image*
- GitHub issue：openvla/openvla#346（fix 说明 + 黑图/白图回归测试）
- OpenVLA `modeling_prismatic.py` 的 `prepare_inputs_for_generation` / `predict_action`
