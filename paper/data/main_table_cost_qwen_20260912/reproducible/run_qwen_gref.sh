#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration
WORK=/tmp/codex_task_main_cost_20260912
OUT="$ROOT/outputs/main_table_cost_qwen_20260912"
mkdir -p "$OUT/components" "$OUT/logs"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 VLLM_WORKER_MULTIPROC_METHOD=spawn
python "$WORK/measure_large.py" --root "$ROOT" --model-kind Qwen3.8 --task grefcoco --output "$OUT/components/qwen_gref_latency.json" --samples 66 --batch-size 1 --gpu-memory-utilization .85
python "$WORK/measure_flops.py" --root "$ROOT" --model-kind Qwen3.8 --task grefcoco --output "$OUT/components/qwen_gref_flops.json" --samples 8
echo complete > "$OUT/logs/qwen_gref.done"
