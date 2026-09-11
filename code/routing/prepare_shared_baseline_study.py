"""Predeclare mandatory-input shared search; inspect train/validation only."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
out=a.output
out.mkdir(parents=True,exist_ok=True)
if (out/'protocol.json').exists():raise FileExistsError(out/'protocol.json')
work=a.root/'outputs/router_full_20260910'
cache=a.root/'outputs/router_training_cache_complete_20260908'
specs={'cub':('cub','residual_blocks'),'glsim':('cub','global_local_encoder'),
       'groundingdino':('grefcoco','multimodal_encoder'),'instancevg':('grefcoco','joint_encoder'),
       'nlvr2':('nlvr2','multimodal_encoder'),'vilt':('nlvr2','ordered_image_text'),
       'yolo26x':('construction','feature_blocks'),'rtdetr_x_fp32':('construction','query_decoder')}
base={'student':'4M','representation_dim':512,'lr':.001,'target':'four_state','strategy':'natural','hidden_percent':90}
stages=[{'parameter':'hidden_percent','values':[30,60,90]},
        {'parameter':'student','values':['1M','2M','4M','8M']},
        {'parameter':'representation_dim','values':[32,64,128,256,512]},
        {'parameter':'lr','values':[.0001,.0003,.001,.003,.01]},
        {'parameter':'target','values':['four_state','gain','error']},
        {'parameter':'strategy','values':['natural','weighted','balanced']}]
protocol={'name':'shared mandatory image and hidden baseline','base':base,'stages':stages,
          'fixed':{'fraction':1.0,'seed':42,'freeze_encoder':True,'fusion':'concat','harm_cost':1.0,
                   'batch_size':256,'weight_decay':.01,'max_epochs':50,'patience':8,'cpu_threads':1},
          'selection':'mean validation ranked-budget area 0-25%, equal model-pair means within each task then equal task means; never average numerical values of independently selected hyperparameters',
          'search':'sequential coordinate search in declared stage order; at most 18 shared settings / 288 individual model-pair fits; not exhaustive Cartesian-grid optimization',
          'tie_break':'lower average active encoder plus head parameter count, then stable candidate id',
          'time_limit_seconds':7200,'maximum_shared_settings':18,
          'final_threshold':'after global setting is locked, choose each pair threshold from validation nominal 1..99%; exact metric ties prefer lower validation actual calls then lower nominal budget',
          'inputs':'image and hidden mandatory; real task text mandatory for gRefCOCO/NLVR2; confidence and prediction output included for every task; no invented constant text branch on CUB/construction',
          'hidden_policy':'one shared relative depth; native classification/visual block or joint multimodal representation, fixed group per expert; ceil(percent * actual group depth / 100)',
          'limitations':['CUB expert previously exposed to internal validation images; this study does not retrain that expert',
                        'previous test results have already been inspected; new hyperparameter choice is validation-only, not a newly blinded test set',
                        'frozen cached image/text encodings are equivalent inputs, not end-to-end image encoder fine-tuning',
                        'shared encoder architecture and head widths do not imply identical total parameter counts with different task input dimensions'],
          'students':{},'experts':{},'prepared_without_test_data':True}
for student in stages[1]['values']:
    source=a.root/'outputs/mini_siglip_strict_20260910_300ep'/student
    done=json.loads((source/'completion.json').read_text())
    if done['status']!='complete' or done['epochs']!=300:raise ValueError(('incomplete strict student',student))
    cfg=json.loads((source/'config.json').read_text())
    protocol['students'][student]={'directory':str(source),'checkpoint':str(source/'epoch_300.pt'),'parameters':cfg['parameters']}
for expert,(task,group) in specs.items():
    inventory=json.loads((cache/expert/'layer_inventory.json').read_text())[group]
    depth=len(inventory)
    layers={}
    for percent in stages[0]['values']:
        rank=math.ceil(depth*percent/100)
        layers[str(percent)]={'key':f'{group}__layer_{rank:02d}','rank':rank,'depth':depth,'module':inventory[rank-1]['module']}
    branches=['image']+(['text'] if task in ['nlvr2','grefcoco'] else [])+['hidden','confidence','output']
    counts={}
    for split in ['train','val']:
        directory=work/'data'/expert/split
        rows=[json.loads(line) for line in (directory/'records.jsonl').open()]
        ids=[r['sample_id'] for r in rows]
        counts[split]=len(rows)
        positions=json.loads((directory/'index.json').read_text())
        first=Path(positions[0]['shard'])
        with np.load(first/'states.npz') as states:
            for layer in layers.values():
                if layer['key'] not in states:raise ValueError((expert,split,layer['key'],'not cached'))
                layer['dimension']=int(states[layer['key']].shape[1])
        for student in protocol['students']:
            feature=work/'features'/student/task/split
            if json.loads((feature/'sample_ids.json').read_text())!=ids:raise ValueError(('feature ordering',student,expert,split))
            for branch in ['image','text','valid']:
                array=np.load(feature/(branch+'.npy'),mmap_mode='r')
                if len(array)!=len(rows):raise ValueError(('feature size',student,expert,split,branch))
    protocol['experts'][expert]={'task':task,'branches':branches,'layers':layers,'rows':counts}
    print('INPUTS_READY',expert,branches,counts,layers,flush=True)
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('READY',out/'protocol.json',flush=True)
