"""Fixed within-split group permutations; unequal-size unique groups remain explicit."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

if __name__=='__main__':
    work=ROOT/'outputs/router_exploration_25pct_20260910'
    parents={}
    def representative(key):
        parents.setdefault(key,key)
        while parents[key]!=key:
            parents[key]=parents[parents[key]];key=parents[key]
        return key
    cub=json.loads((ROOT/'data/router_splits_strict_20260910/cub_image_correspondence.json').read_text())
    for row in cub:
        for identity in row['official_ids']:parents[representative('cub:'+identity)]=representative('cub_sample:'+row['sample_id'])
    # Original source URLs capture repeated NLVR2 source images, including renamed files.
    nlvr={}
    for split in ['train','dev']:
        for line in (ROOT/'data/nlvr2/annotations/data'/(split+'.json')).read_text().splitlines():
            row=json.loads(line);key='nlvr:'+row['identifier'].split('-')[1]
            for side in ['left_url','right_url']:parents[representative('url:'+row[side])]=representative(key)
            nlvr[row['identifier']]=key
    summaries=[]
    for task in ['cub','construction','grefcoco','nlvr2']:
        dest=work/'manifests'/task
        for split in ['train','train10','val']:
            output=dest/('hidden_shuffle_'+split+'.json')
            if output.exists():raise FileExistsError(output)
            records=[json.loads(x) for x in (dest/(split+'.jsonl')).read_text().splitlines()]
            groups=defaultdict(list)
            for i,row in enumerate(records):
                if 'source_group' in row:key=row['source_group']
                elif task=='cub':key=representative('cub_sample:'+row['sample_id'])
                elif task=='nlvr2':key=representative(nlvr[row['sample_id']])
                elif task=='grefcoco':key=str(row['image_id'])
                else:key=row['sample_id']
                groups[key].append(i)
            by_size=defaultdict(list)
            for key,rows in groups.items():by_size[len(rows)].append(key)
            rng=np.random.default_rng(1042 if split=='val' else 42)
            permutation=np.arange(len(records));changed_groups=0
            for size,keys in sorted(by_size.items()):
                if len(keys)<2:continue
                ordered=[keys[i] for i in rng.permutation(len(keys))]
                for source,target in zip(ordered,ordered[1:]+ordered[:1]):
                    assert source!=target
                    permutation[groups[source]]=groups[target];changed_groups+=1
            assert sorted(permutation.tolist())==list(range(len(records)))
            unchanged=int((permutation==np.arange(len(records))).sum())
            result={'task':task,'split':split,'indices':permutation.tolist(),'sample_ids':[r['sample_id'] for r in records],
                    'rows':len(records),'groups':len(groups),'changed_groups':changed_groups,'unchanged_rows':unchanged,
                    'changed_row_fraction':1-unchanged/len(records),'rule':'Permute entire equal-sized source groups within each split. Singleton size bins remain unchanged; never drop rows.',
                    'seed':1042 if split=='val' else 42,'test_rows_used':0}
            output.write_text(json.dumps(result,indent=2));summaries.append({k:v for k,v in result.items() if k not in ['indices','sample_ids']})
    (work/'hidden_shuffle_preparation.json').write_text(json.dumps({'status':'complete','splits':summaries},indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
