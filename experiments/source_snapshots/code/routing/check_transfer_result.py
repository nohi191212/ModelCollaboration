"""Replay saved transfer scores, source-only thresholds and target metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0, str(ROOT / 'code'))
from routing.evaluate_router import evaluate
from routing.transfer_thresholds import source_thresholds, target_threshold_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    dest = Path(args.run)
    work = ROOT / 'outputs/router_exploration_25pct_20260910'
    cfg = json.loads((dest / 'config.json').read_text())
    done = json.loads((dest / 'completion.json').read_text())
    assert done['status'] == 'complete' and not done['target_used_for_selection']
    assert done['target_training_rows_used'] == 0
    if not done['smoke']:
        assert done['strict_student_verified']
    population = json.loads((work / 'task_holdout/transfer_sources' / (cfg['heldout_task'] + '.json')).read_text())
    manifest = json.loads((dest / 'source_samples.json').read_text())
    assert set(manifest) == set(population['source_experts'])
    scores = {}
    for expert, description in population['source_experts'].items():
        assert description['task'] != cfg['heldout_task']
        for split in ['train', 'val']:
            records = [json.loads(x) for x in (work / 'datasets' / expert / split / 'records.jsonl').read_text().splitlines()]
            if done['smoke']:
                records = records[:8]
            assert manifest[expert][split + '_ids'] == [r['sample_id'] for r in records]
        scores[expert] = np.load(dest / (expert + '_source_scores.npy'))
        assert scores[expert].shape == (len(records),)
        assert np.isfinite(scores[expert]).all() and ((scores[expert] >= 0) & (scores[expert] <= 1)).all()
    calibration = json.loads((dest / 'source_thresholds.json').read_text())
    assert calibration == source_thresholds(scores)
    history = [json.loads(x) for x in (dest / 'history.jsonl').read_text().splitlines()]
    for row in history:
        assert np.isfinite(row['training_loss']) and row['training_loss'] >= 0
        losses = row['per_expert_validation_loss']
        assert set(losses) == set(scores)
        assert all(np.isfinite(v) and v >= 0 for v in losses.values())
        assert abs(sum(losses.values()) / len(losses) - row['source_validation_loss']) < 1e-12
    assert min(row['source_validation_loss'] for row in history) == done['source_validation_loss']
    metrics = json.loads((dest / 'target_metrics.json').read_text())
    fixed = json.loads((dest / 'target_fixed_threshold_metrics.json').read_text())
    assert set(metrics) == set(fixed) == set(population['target_experts'])
    warnings = []
    checks = []
    for expert in population['target_experts']:
        records = [json.loads(x) for x in (work / 'datasets' / expert / 'val/records.jsonl').read_text().splitlines()]
        if done['smoke']:
            records = records[:8]
        target_scores = np.load(dest / (expert + '_scores.npy'))
        assert target_scores.shape == (len(records),)
        assert np.isfinite(target_scores).all() and ((target_scores >= 0) & (target_scores <= 1)).all()
        if float(target_scores.std()) < 1e-6:
            warnings.append({'expert': expert, 'reason': 'near_constant_scores'})
        assert set(metrics[expert]) == set(fixed[expert]) == {'Qwen3.8', 'MiniCPM'}
        for endpoint in ['Qwen3.8', 'MiniCPM']:
            assert evaluate(records, endpoint, target_scores) == metrics[expert][endpoint]
            assert target_threshold_results(records, endpoint, target_scores, calibration) == fixed[expert][endpoint]
            checks.append({'expert': expert, 'endpoint': endpoint, 'saved_curve_replay': 'passed', 'unchanged_source_threshold_replay': 'passed'})
    report = {'status': 'replayed_requires_warning_review' if warnings else 'replay_passed',
              'smoke': done['smoke'], 'checks': checks, 'warnings': warnings,
              'scope': 'Checks saved scores and metrics, not an independent checkpoint inference or final scientific conclusion.'}
    (dest / 'result_replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if warnings:
        review = dest / 'anomaly_review.json'
        assert review.exists() and json.loads(review.read_text())['decision'] == 'no_bug', warnings
