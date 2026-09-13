#!/usr/bin/env bash
set -euo pipefail
WORK=/tmp/codex_task_main_cost_20260912
ROOT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration
python -m pip install --no-deps --target "$WORK/yolo_deps" ultralytics==8.3.222
export PYTHONPATH="$WORK/yolo_deps${PYTHONPATH:+:$PYTHONPATH}"
bash "$WORK/run_small.sh" 1 yolo26x Qwen3.8 qwen
