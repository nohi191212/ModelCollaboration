#!/usr/bin/env bash
set -eu
ROOT=/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration
command -v python
python -c 'import torch; print(torch.__version__); import vllm; print(vllm.__version__)'
ls "$ROOT/envs"
ls "$ROOT/outputs/router_full_20260910"
ls "$ROOT/outputs/mini_siglip_strict_20260910_300ep/8M"
sed -n '1,170p' "$ROOT/code/routing/raw_student.py"
sed -n '1,150p' "$ROOT/code/routing/student_artifacts.py"
