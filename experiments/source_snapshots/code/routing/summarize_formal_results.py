"""Summarize formal test-set results after the scheduler finishes."""
import html
import json
from pathlib import Path

ROOT = Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
WORK = ROOT / 'outputs/router_full_20260910'


def main():
    queue = []
    for queue_file in ['formal_queue.json', 'formal_extra_queue.json']:
        path = WORK / queue_file
        if path.exists(): queue.extend(json.loads(path.read_text())['runs'])
    rows, missing = [], []
    for item in queue:
        run = WORK / 'formal_runs' / item['id']
        done_path = run / 'completion.json'
        if not done_path.exists():
            missing.append(item['id']); continue
        done = json.loads(done_path.read_text())
        cfg = item['config']
        for endpoint, result in done.get('test', {}).items():
            fixed = result['fixed_threshold_test']['curve']
            ranked = result['ranked_test']['curve']
            at25 = next(x for x in fixed if x['budget'] == .25)
            rank25 = next(x for x in ranked if x['budget'] == .25)
            rows.append({'id': item['id'], 'tag': item['tag'], 'task': cfg['task'], 'expert': cfg['expert'],
                         'endpoint': endpoint, 'fraction': cfg['fraction'], 'student': cfg['student'],
                         'branches': '+'.join(cfg['branches']), 'fusion': cfg['fusion'], 'target': cfg['target'],
                         'strategy': cfg['strategy'], 'representation_dim': cfg['representation_dim'],
                         'hidden_layer': cfg.get('hidden_layer', ''), 'fixed25_metric': at25['metric'],
                         'fixed25_actual_fraction': at25['actual_fraction'], 'fixed25_rescue': at25['rescue'],
                         'fixed25_harm': at25['harm'], 'rank25_metric': rank25['metric'],
                         'validation_selection': result['validation_selection'], 'best_epoch': result['best_epoch'],
                         'seconds': done['seconds']})
    rows.sort(key=lambda x: (x['task'], x['expert'], x['endpoint'], x['fraction'], x['id']))
    (WORK / 'formal_results.json').write_text(json.dumps({'status': 'complete' if not missing else 'partial',
        'planned': len(queue), 'completed': len(set(x['id'] for x in rows)), 'missing': missing, 'rows': rows}, indent=2))
    headers = ['id', 'tag', 'task', 'expert', 'endpoint', 'fraction', 'student', 'branches', 'target', 'fixed25_metric',
               'fixed25_actual_fraction', 'fixed25_rescue', 'fixed25_harm', 'rank25_metric', 'best_epoch', 'seconds']
    md = ['# Formal router experiments (strict test evaluation)', '',
          f'Completed runs: {len(set(x["id"] for x in rows))}/{len(queue)}',
          'Validation selects the epoch and deployment threshold; the table below reports test-set values at the requested 25% budget.', '']
    md.append('| ' + ' | '.join(headers) + ' |'); md.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        md.append('| ' + ' | '.join(str(row.get(h, '')).replace('|', '\\|') for h in headers) + ' |')
    if missing: md += ['', 'Missing: ' + ', '.join(missing)]
    (WORK / 'formal_results.md').write_text('\n'.join(md))
    table_rows = []
    for row in rows:
        table_rows.append('<tr>' + ''.join(f'<td>{html.escape(str(row.get(h, "")))}</td>' for h in headers) + '</tr>')
    page = '<!doctype html><meta charset="utf-8"><title>Formal router results</title><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ddd;padding:4px}th{position:sticky;top:0;background:#eef}tr:nth-child(even){background:#fafafa}</style>'
    page += '<h1>Formal router experiments</h1><p>Strict test evaluation; validation-only epoch and threshold selection.</p><table><thead><tr>'
    page += ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table>'
    (WORK / 'formal_results.html').write_text(page)
    print(json.dumps({'planned': len(queue), 'completed': len(set(x['id'] for x in rows)), 'missing': missing}, indent=2))


if __name__ == '__main__':
    main()
