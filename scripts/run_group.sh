#!/usr/bin/env bash
# scripts/run_group.sh — run groups sequentially or on separate GPUs.
#
# Sequential (default):  CUDA_VISIBLE_DEVICES=0 bash scripts/run_group.sh
# Specific groups:       bash scripts/run_group.sh a1 b2
# Multi-GPU (manual):    CUDA_VISIBLE_DEVICES=0 bash scripts/run_group.sh a1 &
#                        CUDA_VISIBLE_DEVICES=1 bash scripts/run_group.sh b2 &
#
# Disk-staged workflow (50G quota): Phase A groups (OpenVLA) and Phase B groups
# (MiniCPM-o) are run in separate sessions; archive one family's weights before
# starting the other (see README §1).
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Wait until the current GPU's memory is released before starting a group.
# (After killing/ending a training process the vGPU can take a while to free
# memory; launching immediately would OOM — this bit A0 once.)
wait_gpu_free() {
  local gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" | head -1)
  while [ "${mem:-0}" -gt 2000 ]; do
    echo "  [wait] GPU$gpu_id still holds ${mem}MiB, waiting for release ..."
    sleep 10
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" | head -1)
  done
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# NOTE: this container's bash has a broken "$@" / array expansion (everything becomes
# "0"), so we iterate positional parameters explicitly instead of using arrays.
if [ -z "$1" ]; then
  set -- a0 a1 b0 b1 b2 b3
fi

DEV="${VLA_DEVICE:-cuda}"

for g in ${1:-} ${2:-} ${3:-} ${4:-} ${5:-} ${6:-}; do
  [ -z "$g" ] && continue
  echo "================================================================"
  echo "  GROUP ${g^^} (device=$DEV)"
  echo "================================================================"
  python "$ROOT/scripts/check_env.py" || { echo "check_env failed for $g"; exit 1; }
  wait_gpu_free

  case "$g" in
    a0|b0)  # SFT only
      python "$ROOT/scripts/train_sft.py" --group "$g" --device "$DEV" ;;
    a1|b1|b2|b3)
      python "$ROOT/scripts/train_sft.py" --group "$g" --device "$DEV"
      python "$ROOT/scripts/train_grpo.py" --group "$g" --device "$DEV" ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac

  # evaluation (ID / OOD / robustness) after training completes
  for split in id ood robustness; do
    python "$ROOT/scripts/evaluate.py" --group "$g" --split "$split" --device "$DEV" || true
  done
done

echo "All groups finished. Plotting:"
python "$ROOT/scripts/plot_results.py" --outdir "$ROOT/viz"
