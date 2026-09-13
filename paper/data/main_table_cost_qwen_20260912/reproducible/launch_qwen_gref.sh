#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration
WORK=/tmp/codex_task_main_cost_20260912
OUT="$ROOT/outputs/main_table_cost_qwen_20260912"
mkdir -p "$OUT/components" "$OUT/logs"
nohup bash "$WORK/run_qwen_gref.sh" > "$OUT/logs/qwen_gref.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/qwen_gref.pid"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nohup python "$WORK/measure_router.py" --root "$ROOT" --expert instancevg --endpoint Qwen3.8 --output "$OUT/components/instancevg_router.json" > "$OUT/logs/router.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/router.pid"
cat "$OUT/logs/"*.pid
