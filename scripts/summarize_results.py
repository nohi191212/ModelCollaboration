"""Summarize saved main-table results; no model or dataset access."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
groups=json.loads((root/'results/results.json').read_text())['groups']
counts={'pairs':len(groups),'better_than_confidence':0,'fewer_calls':0,'above_both_models':0}
for g in groups:
    l=g['methods']['learned']['test_point'];c=g['methods']['confidence']['test_point']
    counts['better_than_confidence']+=l['metric']>c['metric']
    counts['fewer_calls']+=l['actual_fraction']<c['actual_fraction']
    counts['above_both_models']+=l['metric']>max(g['small']['test'],g['large']['test'])
print(json.dumps(counts,indent=2))
