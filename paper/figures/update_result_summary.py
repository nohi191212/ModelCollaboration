"""Refresh export metadata; leave all per-run measurements unchanged."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
path=root/'data/results.json'
d=json.loads(path.read_text(encoding='utf-8'))
groups=d['groups']
# A real observed tie differs by 1.11e-16 after five-seed averaging.
tol=1e-12
d['summary']={
 'learned_beats_confidence':sum(g['methods']['learned']['test_point']['metric']>g['methods']['confidence']['test_point']['metric']+tol for g in groups),
 'learned_beats_both':sum(g['methods']['learned']['test_point']['metric']>max(g['small']['test'],g['large']['test'])+tol for g in groups),
 'learned_beats_large':sum(g['methods']['learned']['test_point']['metric']>g['large']['test']+tol for g in groups)}
d['source']='Five-seed router study: dse_results/objective_multiseed_20260912/results_five_seeds.json and runs/repeat_*/test_evaluation.json; original specialist/VLM outcomes are reused.'
d['summary_comparison']='Five-seed mean metrics; differences within 1e-12 in raw fractions are treated as ties. CUB/ResNet/Q has exactly two rescues and two harms across five runs.'
path.write_text(json.dumps(d,indent=2),encoding='utf-8')
p=root/'data/five_seed_counts.json';c=json.loads(p.read_text());c['above_endpoints']=d['summary']['learned_beats_both'];p.write_text(json.dumps(c,indent=2))
for file,old,new in [('short_abstract.tex','Nine collaborations','Eight collaborations'),('experiments_mainresults.tex','nine outperform','eight outperform')]:
 p=root/'sections'/file;t=p.read_text();assert old in t or new in t;t=t.replace(old,new);p.write_text(t)
print(d['summary'])
