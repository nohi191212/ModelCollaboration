from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[2]
dest=Path(__file__).resolve().parent
for source,target in [('measure_large_vllm_runtime_20260911.py','measure_large.py'),('measure_large_flops_reference_20260911.py','measure_flops.py'),('measure_expert_router_runtime_20260911.py','measure_specialist.py')]:
    shutil.copy2(root/'tmp'/source,dest/target)
