#!/usr/bin/env bash
set -euo pipefail
WORK=/tmp/codex_task_main_cost_20260912
OUT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/main_table_cost_20260912
bash -n "$WORK/run_small.sh"
python -m py_compile "$WORK/measure_router.py"
nohup bash "$WORK/run_small.sh" 0 instancevg MiniCPM minicpm > "$OUT/logs/instancevg_small.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/instancevg_small.pid"
nohup bash "$WORK/run_small.sh" 1 yolo26x Qwen3.8 qwen > "$OUT/logs/yolo26x_small.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/yolo26x_small.pid"
cat "$OUT/logs/instancevg_small.pid" "$OUT/logs/yolo26x_small.pid"
