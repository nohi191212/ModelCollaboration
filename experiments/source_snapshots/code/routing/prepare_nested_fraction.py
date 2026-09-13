"""Freeze a grouped 10% subset nested inside the existing 25%; no validation redraw."""
import json,math
from collections import defaultdict
from pathlib import Path

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    parent=json.loads((work/'preparation_complete.json').read_text())
    for task in parent['counts']:
        dest=work/'manifests'/task
        assert not (dest/'train10.jsonl').exists() and not (dest/'train10_indices.json').exists()
    summaries={}
    for task,count in parent['counts'].items():
        dest=work/'manifests'/task
        rows=[json.loads(x) for x in (dest/'train.jsonl').read_text().splitlines()]
        order=json.loads((dest/'group_order.json').read_text())
        groups=defaultdict(list)
        for r in rows:groups[r['source_group']].append(r)
        selected=order['selected_groups'];assert set(selected)==set(groups)
        target_groups=round(count['eligible_groups']*.1)
        if task=='cub':
            per_class=defaultdict(list)
            for group in selected:
                categories={r['ground_truth']['category_id'] for r in groups[group]}
                assert len(categories)==1
                per_class[next(iter(categories))].append(group)
            quotas={c:target_groups*len(gs)/len(selected) for c,gs in per_class.items()}
            allocated={c:math.floor(q) for c,q in quotas.items()}
            for c in sorted(quotas,key=lambda c:(-(quotas[c]-allocated[c]),c))[:target_groups-sum(allocated.values())]:allocated[c]+=1
            chosen=[g for c in sorted(per_class) for g in per_class[c][:allocated[c]]]
        else:
            chosen=order['ordered_groups'][:target_groups]
        chosen=set(chosen)
        assert len(chosen)==target_groups and chosen<=set(selected)
        indices=[i for i,r in enumerate(rows) if r['source_group'] in chosen]
        sub=[rows[i] for i in indices]
        validation_ids={json.loads(x)['sample_id'] for x in (dest/'val.jsonl').read_text().splitlines()}
        assert not validation_ids&{r['sample_id'] for r in sub}
        summary={'fraction':.1,'parent_fraction':.25,'indices':indices,'sample_ids':[r['sample_id'] for r in sub],
                 'selected_groups':len(chosen),'eligible_groups':count['eligible_groups'],'train_rows':len(sub),
                 'eligible_train_rows':count['eligible_train_rows'],'actual_row_fraction':len(sub)/count['eligible_train_rows'],
                 'validation_unchanged':True,'selection':'Existing group order; CUB class-stratified proportional quotas with largest-remainder rounding. No new random draw.'}
        (dest/'train10.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sub))
        (dest/'train10_indices.json').write_text(json.dumps(summary,indent=2))
        summaries[task]={k:v for k,v in summary.items() if k not in ['indices','sample_ids']}
    (work/'nested_fraction_complete.json').write_text(json.dumps({'status':'complete','tasks':summaries,'test_rows_used':0},indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
