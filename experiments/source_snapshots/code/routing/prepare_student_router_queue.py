"""Matched frozen-router controls for actual student sources and depths."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from prepare_followup_queue import canonical, signature

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    work = ROOT / 'outputs/router_exploration_25pct_20260910'
    controls = json.loads((work / 'distillation_controls/student_control_queue.json').read_text())
    details = {row['id']: row for row in json.loads((work / 'distillation_controls/student_control_design.json').read_text())['configs']}
    initial = json.loads((work / 'initial_queue.json').read_text())
    candidates = defaultdict(list)
    prior = initial
    if args.precheck:
        for cfg in initial:
            if cfg['seed'] == 42 and cfg['target'] == 'error':
                for endpoint in ['Qwen3.8', 'MiniCPM']:
                    candidates[(cfg['expert'], endpoint)].append((0., {k: v for k, v in cfg.items() if k not in ['id', 'stage', 'seed', 'endpoint']}))
    else:
        for name in ['followup', 'interaction']:
            prior += json.loads((work / (name + '_queue.json')).read_text())
            analysis = json.loads((work / (name + '_queue_analysis.json')).read_text())
            assert analysis['all_queue_fits_complete']
            for warning in analysis['warnings']:
                assert json.loads((work / 'runs' / warning['id'] / 'anomaly_review.json').read_text())['decision'] == 'no_bug'
            for row in analysis['groups']:
                cfg = row['config']
                if cfg['fraction'] != .25 or not cfg['freeze_encoder'] or cfg.get('hidden_shuffle', False):
                    continue
                if not set(cfg['branches']) & {'image', 'text'}:
                    continue
                assert row['seeds_complete']
                scores = [r['score'] for r in row['runs'] if r['seed'] == 42]
                assert len(scores) == 1
                candidates[(cfg['expert'], row['endpoint'])].append((statistics.mean(scores), cfg))
    assert len(candidates) == 16
    used = {signature(cfg): cfg['id'] for cfg in prior}
    queue = []
    comparisons = []
    pairs = []
    anchors = []
    for (expert, endpoint), rows in sorted(candidates.items()):
        score, anchor = sorted(rows, key=lambda row: (-row[0], json.dumps(row[1], sort_keys=True)))[0]
        anchors.append({'expert': expert, 'endpoint': endpoint, 'config': anchor, 'seed42_score': None if args.precheck else score})
        for control in controls:
            size = str(control['budget_millions']) + 'M' if 'budget_millions' in control else '4M'
            for seed in [42]:
                reference = canonical(dict(anchor, student=size, seed=seed, freeze_encoder=True), anchor['hidden_layer'], endpoint)
                reference.pop('student_control', None)
                alternative = dict(reference, student_control=control['id'])
                assert {k: v for k, v in alternative.items() if k != 'student_control'} == reference
                ids = []
                for role, cfg in [('reference', reference), ('control', alternative)]:
                    key = signature(cfg)
                    if key not in used:
                        cfg = dict(cfg, id=f'student_router_{len(queue) + 1:06d}', stage='student_source_and_depth')
                        queue.append(cfg)
                        used[key] = cfg['id']
                    ids.append(used[key])
                    comparisons.append({'expert': expert, 'endpoint': endpoint, 'group': control['group'], 'role': role,
                                        'control': control['id'], 'seed': seed, 'run_id': used[key]})
                pairs.append({'expert': expert, 'endpoint': endpoint, 'student_control': control['id'],
                              'control_parameters': details[control['id']]['parameters'], 'seed': seed,
                              'reference_id': ids[0], 'control_id': ids[1], 'group': control['group']})
    assert len(pairs) == 16 * 36
    assert {pair['student_control'] for pair in pairs} == set(details)
    if args.precheck:
        report = {'status': 'configuration_precheck_passed', 'matched_pairs': len(pairs), 'students': len(controls),
                  'real_candidates_selected': False, 'gpu_launched': False}
        (work / 'student_router_config_precheck.json').write_text(json.dumps(report, indent=2))
    else:
        for filename in ['student_router_queue.json', 'student_router_design.json']:
            assert not (work / filename).exists(), filename
        (work / 'student_router_queue.json').write_text(json.dumps(queue, indent=2))
        report = {'anchors': anchors, 'comparisons': comparisons, 'pairs': pairs, 'new_fits': len(queue),
                  'scope': 'Same frozen router settings and seeds; controls change encoder provenance/depth. Fixed-width controls compare to original 4M depth2; not all are within the deployment parameter budget.',
                  'selection': 'Best encoder-using frozen candidate per pair after all followup and interaction results; compare seed42 only across both stages.',
                  'all_exploration_complete': False, 'test_rows_used': 0}
        (work / 'student_router_design.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'matched_pairs': len(pairs), 'new_fits': None if args.precheck else len(queue), 'precheck_only': args.precheck, 'gpu_launched': False}))
