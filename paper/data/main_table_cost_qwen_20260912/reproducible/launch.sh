#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/main_table_cost_20260912
WORK=/tmp/codex_task_main_cost_20260912
mkdir -p "$OUT/logs" "$OUT/components"
bash -n "$WORK/run_large.sh"
python -m py_compile "$WORK/measure_large.py" "$WORK/measure_flops.py" "$WORK/measure_specialist.py"
nohup bash "$WORK/run_large.sh" 0 MiniCPM grefcoco minicpm > "$OUT/logs/minicpm.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/minicpm.pid"
nohup bash "$WORK/run_large.sh" 1 Qwen3.8 construction qwen > "$OUT/logs/qwen.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/qwen.pid"
cat "$OUT/logs/minicpm.pid" "$OUT/logs/qwen.pid"
