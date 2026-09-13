#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration
WORK=/tmp/codex_task_main_cost_20260912
OUT="$ROOT/outputs/main_table_cost_20260912"
GPU=$1
EXPERT=$2
ENDPOINT=$3
TAG=$4
while [[ ! -f "$OUT/logs/${TAG}.done" ]]; do
  PID=$(cat "$OUT/logs/${TAG}.pid")
  STATE=$(ps -p "$PID" -o stat= || true)
  if [[ -z "$STATE" || "$STATE" == Z* ]]; then
    echo "Large-model stage stopped without completion: $TAG" >&2
    exit 1
  fi
  sleep 5
done
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4
PYTHON=python
if [[ "$EXPERT" == instancevg ]]; then PYTHON="$ROOT/envs/instancevg/bin/python"; fi
"$PYTHON" "$WORK/measure_specialist.py" --root "$ROOT" --output "$OUT/components/${EXPERT}_specialist.json" --component specialist --expert "$EXPERT" --samples 64 --batch-size 1 --device cuda:0
python "$WORK/measure_router.py" --root "$ROOT" --expert "$EXPERT" --endpoint "$ENDPOINT" --output "$OUT/components/${EXPERT}_router.json"
echo complete > "$OUT/logs/${EXPERT}_small.done"
