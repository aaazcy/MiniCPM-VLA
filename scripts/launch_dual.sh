#!/usr/bin/env bash
# scripts/launch_dual.sh — run A groups on GPU0 and B groups on GPU1 in parallel.
#
# Usage:
#   bash scripts/launch_dual.sh [A_GROUPS] [B_GROUPS]
#   bash scripts/launch_dual.sh "a0 a1" "b0 b1 b2 b3"     # default
#
# Each side runs scripts/run_group.sh with its own CUDA_VISIBLE_DEVICES and log file.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A_GROUPS="${1:-a0 a1}"
B_GROUPS="${2:-b0 b1 b2 b3}"

export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/runs"

echo "==> launching A groups ($A_GROUPS) on GPU0"
CUDA_VISIBLE_DEVICES=0 bash "$ROOT/scripts/run_group.sh" $A_GROUPS \
    > "$ROOT/runs/log_a.log" 2>&1 &
PID_A=$!

echo "==> launching B groups ($B_GROUPS) on GPU1"
CUDA_VISIBLE_DEVICES=1 bash "$ROOT/scripts/run_group.sh" $B_GROUPS \
    > "$ROOT/runs/log_b.log" 2>&1 &
PID_B=$!

echo "A -> $ROOT/runs/log_a.log (pid $PID_A)"
echo "B -> $ROOT/runs/log_b.log (pid $PID_B)"
echo "tail -f $ROOT/runs/log_a.log   # monitor"
wait $PID_A
STATUS_A=$?
wait $PID_B
STATUS_B=$?
echo "A exit=$STATUS_A | B exit=$STATUS_B"
