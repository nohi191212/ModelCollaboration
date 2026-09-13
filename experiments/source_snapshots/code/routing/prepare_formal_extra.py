"""Add coverage runs omitted by the bounded first ablation queue."""
import json
from pathlib import Path

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
WORK = ROOT / 'outputs/router_full_20260910'


def signature(cfg):
    return json.dumps({k: v for k, v in cfg.items() if k not in {'id', 'stage'}}, sort_keys=True)


def main():
    queue = json.loads((WORK / 'formal_queue.json').read_text())['runs']
    used = {signature(item['config']) for item in queue}
    baselines = [item for item in queue if item['tag'] == 'baseline_100pct']
    extra = []
    for item in baselines:
        anchor = dict(item['config'])
        variants = [(f'representation_{d}', {'representation_dim': d}) for d in [32, 64]]
        variants.extend((f'student_{s}', {'student': s}) for s in ['1M', '2M', '4M', '8M'] if s != anchor['student'])
        for tag, changes in variants:
            cfg = dict(anchor); cfg.update(changes); cfg['stage'] = 'formal_ablation_extra'
            if signature(cfg) in used: continue
            run_id = f'extra_{len(extra) + 1:04d}'; cfg['id'] = run_id
            extra.append({'id': run_id, 'tag': tag, 'config': cfg}); used.add(signature(cfg))
    payload = {'status': 'ready', 'runs': extra, 'count': len(extra), 'purpose': 'Complete representation 32/64 and student-size coverage; test remains evaluation-only.'}
    (WORK / 'formal_extra_queue.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps({'count': len(extra)}))


if __name__ == '__main__': main()
