# 数据准备文档(Data Preparation)

本文档记录本实验数据集的完整准备流程,保证可复现。

## 0. 数据源总览

| 数据源 | 原始规模 | 获取方式 | 最终规模 |
|---|---|---|---|
| CALVIN(task_ABC_D) | 官方 zip **517 GB**(1,894,126 个 step 文件) | **远程选择性抽取**(HTTP Range,不整包下载) | 环境 A:127 训练窗口 + 34 验证窗口(≈2.5 GB 原始步) |
| BridgeData V2 | 60,096 轨迹(官方原始 ~62GB+) | HF 镜像流式拉取(LeRobot 格式) | 2,000 段随机子采样 |
| CSI 语音(B3) | 复用 CALVIN 轨迹 | 本地 TTS 合成(edge-tts) | 107 条 16 kHz wav |
| 鲁棒指令集 | CALVIN 官方自带改写标注 | 从 zip 抽取 `lang_paraphrase-MiniLM-L3-v2` | 34 条改写指令 |

## 1. 关键事实(实测验证,写代码前必读)

1. **CALVIN npz 是"单步文件"**:每个 `episode_XXXXXXX.npz` 只有 1 步观测(keys:
   `rgb_static`(200,200,3)、`rel_actions`(7,)、`actions`(7,)、`robot_obs`(15,)、
   `scene_obs`(24,)等)。episode 分组靠 `ep_start_end_ids.npy`(N,2 闭区间)。
2. **episode 很长**:训练集 147 个 episode,平均 12,210 步;验证集 4 个 episode
   (99,022 步)。真正的语义单元是 **64 步语言窗口**。
3. **语言标注在独立文件**:`lang_annotations/auto_lang_ann.npy` =
   `{"language": {"ann"/"task"/"emb"}, "info": {"indx": (K,2) 64步窗口}}`。
4. **环境映射**:官方修复包 `task_ABC_D_scene_info.zip`(525 B)给出
   `scene_info.npy`:`B=[0,598909]`、`C=[598910,1191338]`、`A=[1191339,1795044]`
   (step 索引,环境 A 共 46 episodes / 603,660 步)。
5. **无 testing 分割**:task_ABC_D.zip 只有 training + validation → OOD 测试遵循
   官方 ABC→D 协议:**验证集 episode 在 `calvin_scene_D` 场景渲染**。
6. **动作已归一化**:官方 README 明确 `rel_actions` 本身就是 [-1,1](位置×50、
   姿态×20、抓手±1),环境直接消费,无需再缩放。
7. **外层 zip 是 deflate 压缩**(method=8),Range 抽取后必须 `zlib.decompress(-15)`。

## 2. CALVIN 选择性抽取管线(4 步)

```bash
# ① 解析 zip 中央目录 → 189万文件索引(229MB,断点续传)
python scripts/fetch_zip_index.py \
    --url http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip \
    --out data/calvin/task_ABC_D.index.json
#   (网络不稳时用 scripts/fetch_cd_chunks.sh 分块下载 .part 再解析)

# ② 流式过滤出元数据(ijson 单遍,适配小内存)
python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
    --include 'task_ABC_D/training/ep_start_end_ids.npy' \
    --include 'task_ABC_D/training/ep_lens.npy' \
    --include 'task_ABC_D/training/lang_annotations/auto_lang_ann.npy' \
    --include 'task_ABC_D/training/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy' \
    --include 'task_ABC_D/validation/ep_start_end_ids.npy' \
    --include 'task_ABC_D/validation/ep_lens.npy' \
    --include 'task_ABC_D/validation/lang_annotations/auto_lang_ann.npy' \
    --include 'task_ABC_D/validation/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy' \
    --out data/calvin/meta.index.json
python scripts/selective_extract.py --index data/calvin/meta.index.json --all \
    --out-dir data/calvin/meta
cp data/calvin/meta/training/scene_info.npy data/calvin/meta/task_ABC_D/training/  # 修复包

# ③ 窗口级抽样计划(环境 A 内,64 步窗口 stride=48)
python scripts/plan_calvin_subset.py \
    --training-meta data/calvin/meta/task_ABC_D/training \
    --validation-meta data/calvin/meta/task_ABC_D/validation \
    --env A --train-window-stride 48 --val-window-stride 32
#   → plan.json: 127 训练窗口 / 34 验证窗口(step id 列表)

# ④ 抽取所选 step 文件(断点续传、跳过已存在、deflate 解压+大小校验)
python scripts/query_zip_index.py --index data/calvin/task_ABC_D.index.json \
    --episode-list data/calvin/plan.json --out data/calvin/steps.index.json
python scripts/selective_extract.py --index data/calvin/steps.index.json --all \
    --out-dir data/calvin/raw
```

## 3. 预处理 → 统一轨迹格式

所有来源统一打包为"一个语言窗口/一条轨迹一个 npz":

```python
<root>/<split>/<traj_id>.npz
    rgb:    (T, 3, 224, 224) uint8   # 静态相机,resize 224×224
    actions:(T, 7) float32 [-1,1]     # 7D 连续动作 [dx,dy,dz,rx,ry,rz,gripper]
    proprio:(T, 8)                    # robot_obs[:8](留档,当前组均为 image-only)
    lang:   str                       # 该窗口的语言指令
    init_robot_obs: (15,)             # 起始状态(评估用,官方重置协议)
    init_scene_obs: (24,)
```

### 3.1 CALVIN
```bash
python scripts/preprocess_data.py --dataset calvin \
    --raw data/calvin/raw/task_ABC_D \
    --plan data/calvin/plan.json \
    --lang-train data/calvin/meta/task_ABC_D/training/lang_annotations/auto_lang_ann.npy \
    --lang-val   data/calvin/meta/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy
# 8 进程并行;产出 sft_train(85%)/sft_val(15%)/id_test + 防泄漏切分清单
```
- 启发式过滤(抓手停滞/抖动)**默认关闭**:CALVIN 遥操作数据天然含"移动中抓手静止"
  等模式,原启发式会误砍 66% 数据;全组数据一致,过滤非必需。

### 3.2 Bridge V2(经 hf-mirror)
```bash
python scripts/preprocess_bridge_remote.py --max-trajectories 2000 --workers 8
```
- 源:`IPEC-COMMUNITY/bridge_orig_lerobot`(LeRobot 格式镜像;huggingface.co 直连
  不可达,统一走 `https://hf-mirror.com/.../resolve/main/...`)
- 每 episode:下载 parquet(动作/状态)+ `image_0` mp4 → av 解码逐帧 resize →
  打包 npz;处理完即删临时文件(内存友好)
- **Bridge 动作归一化**(与 CALVIN 不同,已在 limitations 披露):
  `pos/0.5`、`rot/π`(clip)、抓手 {0,1}→{-1,1}
- 断点续传:已完成的 episode 自动跳过;seed 固定 → 子集与首次运行一致

### 3.3 CSI 语音(B3)
```bash
python scripts/preprocess_csi_tts.py \
    --calvin-root data/calvin --csi-root data/csi --split sft_train
```
- 对每个 sft_train 轨迹的指令做 TTS(edge-tts → 24kHz → librosa 重采样 16kHz 单声道)
- 按指令文本 sha256 命名(`wav/<hash16>.wav`),rollout/数据集/合成三处同一约定
- 帧对齐:整条指令音频按 CALVIN 8 步窗口切段(`wav_segments/`)
- **合成语音,非真人语音** — 已写入 limitations

### 3.4 鲁棒指令集
```bash
python scripts/make_robust_set.py \
    --calvin-root data/calvin \
    --paraphrase data/calvin/meta/task_ABC_D/validation/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy \
    --original   data/calvin/meta/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy
# → splits/robust_instructions.json(34 条改写指令,同 id_test 的窗口/初始状态)
```

## 4. 防泄漏切分(硬保证)

`src/data/splits.py` 生成带 sha256 的清单并断言:
- `sft_train ∩ sft_val = ∅`,`id_test ∩ (sft_train ∪ sft_val) = ∅`
- 加载器每次实例化时重新校验;违反即抛异常

最终清单:
```
calvin_sft_train(107)  calvin_sft_val(20)  calvin_id_test(34)
bridge_train(2000)     robust_instructions(34)
```

## 5. 最终数据集规模

| 部分 | 大小 | 说明 |
|---|---|---|
| CALVIN 轨迹(161 段) | ~0.7 GB | 224px 压缩 npz |
| Bridge 轨迹(2000 段) | ~2-3 GB | 完成后 |
| CSI 语音 | ~50 MB | 16kHz wav + 分段 |
| 元数据/索引/计划 | ~0.25 GB | 保留以便复现 |

## 6. 已知坑(踩过,勿再犯)

1. 本容器旧版 `struct.unpack` 要求缓冲区长度**精确等于**格式大小(用
   `buf[:struct.calcsize(fmt)]` 切片)。
2. `python` 不在非登录 shell 的 PATH 里 → 用 `export PATH=/root/miniconda3/bin:$PATH`
   或全路径 `/root/miniconda3/bin/python`。
3. huggingface.co 直连超时 → 一律走 `hf-mirror.com`(含 API 端点)。
4. Freiburg 服务器连接不稳 → 所有远程下载都有重试 + 断点续传。
5. `datasets` 库与 PyAV 版本兼容:av==13.1.0(太新会因 `metadata_errors` 参数报错)。
6. edge-tts 输出 24 kHz,必须重采样 16 kHz 才能进 Whisper 编码器。
