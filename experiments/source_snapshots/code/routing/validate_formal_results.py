"""Final integrity checks before releasing the occupied GPU service."""
import json
import math
from pathlib import Path

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
WORK = ROOT / 'outputs/router_full_20260910'


def finite(value):
    if isinstance(value, (int, float)): return math.isfinite(value)
    if isinstance(value, list): return all(finite(x) for x in value)
    if isinstance(value, dict): return all(finite(x) for x in value.values())
    return True


def main():
    queue = []
    for name in ['formal_queue.json', 'formal_extra_queue.json']:
        path = WORK / name
        if path.exists(): queue.extend(json.loads(path.read_text())['runs'])
    missing, bad, anomalies = [], [], []
    for item in queue:
        path = WORK / 'formal_runs' / item['id'] / 'completion.json'
        if not path.exists(): missing.append(item['id']); continue
        done = json.loads(path.read_text())
        if done.get('status') != 'complete' or not finite(done): bad.append((item['id'], 'status/nonfinite'))
        for endpoint, result in done.get('test', {}).items():
            for key in ['ranked_test', 'fixed_threshold_test']:
                curve = result.get(key, {}).get('curve', [])
                if len(curve) != 9 or not finite(curve): bad.append((item['id'], endpoint, key))
        if done.get('anomalies'): anomalies.append((item['id'], done['anomalies']))
    joint = WORK / 'joint_comparison_complete.json'
    if not joint.exists() or json.loads(joint.read_text()).get('status') != 'complete': bad.append(('joint', 'missing'))
    result = {'status': 'pass' if not missing and not bad else 'fail', 'planned_runs': len(queue),
              'completed_runs': len(queue) - len(missing), 'missing': missing, 'bad': bad,
              'anomalies': anomalies, 'test_only_final_evaluation': True,
              'joint_status': json.loads(joint.read_text()) if joint.exists() else None}
    (WORK / 'formal_acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'planned_runs': len(queue), 'completed_runs': result['completed_runs'],
                      'missing': missing, 'bad_count': len(bad), 'anomaly_count': len(anomalies)}, indent=2))
    if result['status'] != 'pass': raise SystemExit(1)


if __name__ == '__main__': main()
