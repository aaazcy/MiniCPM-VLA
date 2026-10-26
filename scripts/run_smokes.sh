#!/usr/bin/env bash
# scripts/run_smokes.sh — 冒烟测试阶段(开训门禁),三卡并行。
#
# NOTE: 本容器 bash 的数组/("$@")有 bug,全部使用显式位置参数,禁止数组。
# 每卡一组串行任务(& 后台并行);日志写 runs/smoke_gpu<N>.log。
set -u

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/runs"

echo "==> 启动冒烟测试(三卡并行,详见 docs/smoke_tests.md)"

# GPU0: A 组
(
  set -e
  python "$ROOT/scripts/train_sft.py" --group a1 --smoke 5 --device cuda:0
  python "$ROOT/scripts/smoke_grpo.py" --group a1 --device cuda:0
  echo "GPU0 SMOKES DONE"
) > "$ROOT/runs/smoke_gpu0.log" 2>&1 &
PID0=$!

# GPU1: B3(音频)
(
  set -e
  python "$ROOT/scripts/verify_models.py" --group b3 --batch 1 --device cuda:1
  python "$ROOT/scripts/train_sft.py" --group b3 --smoke 5 --device cuda:1
  python "$ROOT/scripts/smoke_grpo.py" --group b3 --device cuda:1
  echo "GPU1 SMOKES DONE"
) > "$ROOT/runs/smoke_gpu1.log" 2>&1 &
PID1=$!

# GPU2: B0/B1/B2
(
  set -e
  python "$ROOT/scripts/verify_models.py" --group b0 --batch 1 --device cuda:2
  python "$ROOT/scripts/verify_models.py" --group b1 --batch 1 --device cuda:2
  python "$ROOT/scripts/train_sft.py" --group b2 --smoke 5 --device cuda:2
  python "$ROOT/scripts/smoke_grpo.py" --group b2 --device cuda:2
  echo "GPU2 SMOKES DONE"
) > "$ROOT/runs/smoke_gpu2.log" 2>&1 &
PID2=$!

echo "GPU0 pid=$PID0 | GPU1 pid=$PID1 | GPU2 pid=$PID2"
echo "tail -f runs/smoke_gpu0.log  # 监控"
wait $PID0; S0=$?
wait $PID1; S1=$?
wait $PID2; S2=$?
echo "smoke exit codes: gpu0=$S0 gpu1=$S1 gpu2=$S2"
[ "$S0" = "0" ] && [ "$S1" = "0" ] && [ "$S2" = "0" ] && echo "ALL SMOKES PASSED" || echo "SOME SMOKES FAILED — check logs"
