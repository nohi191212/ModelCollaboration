#!/usr/bin/env bash
set -euo pipefail
OUT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration/outputs/main_table_cost_20260912
WORK=/tmp/codex_task_main_cost_20260912
ps -p 6908 -o args= | grep -F "$WORK/measure_large.py" | grep -F 'MiniCPM'
test "$(ps -p 7122 -o ppid= | tr -d ' ')" = 6908
kill -TERM 7122 6908
export VLLM_WORKER_MULTIPROC_METHOD=spawn
nohup bash "$WORK/run_large.sh" 0 MiniCPM grefcoco minicpm > "$OUT/logs/minicpm_spawn.log" 2>&1 < /dev/null &
echo $! > "$OUT/logs/minicpm.pid"
cat "$OUT/logs/minicpm.pid"
