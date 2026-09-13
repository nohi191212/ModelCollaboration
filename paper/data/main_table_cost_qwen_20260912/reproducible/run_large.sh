#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration
WORK=/tmp/codex_task_main_cost_20260912
OUT="$ROOT/outputs/main_table_cost_20260912"
mkdir -p "$OUT/components" "$OUT/logs"
GPU=$1
KIND=$2
TASK=$3
TAG=$4
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4
python "$WORK/measure_large.py" --root "$ROOT" --model-kind "$KIND" --task "$TASK" --output "$OUT/components/${TAG}_latency.json" --samples 66 --batch-size 1 --gpu-memory-utilization .85
python "$WORK/measure_flops.py" --root "$ROOT" --model-kind "$KIND" --task "$TASK" --output "$OUT/components/${TAG}_flops.json" --samples 8
echo complete > "$OUT/logs/${TAG}.done"
