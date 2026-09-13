from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--expert',required=True)
p.add_argument('--endpoint',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.root/'code'))
from routing import train_router_formal as formal
from routing.router import prepare_images,load_distilled_router
from siglip_distillation.mini_siglip import MiniSiglip
from tokenizers import Tokenizer
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.mha.set_fastpath_enabled(False)
work=a.root/'outputs/router_shared_image_hidden_20260910'
run=f'shared_12_{a.expert}_{a.endpoint}'
dest=work/'runs'/run
cfg=json.loads((dest/'config.json').read_text())
assert cfg['seed']==42 and cfg['student']=='8M' and cfg['target']=='four_state'
records,data=formal.load_data(cfg,'test',torch.device('cuda'))
for key in ['hidden','confidence']:
    norm=np.load(dest/(key+'_normalization.npz'))
    data[key]=(data[key]-torch.from_numpy(norm['mean']).cuda())/torch.from_numpy(norm['std']).cuda()
model=formal.FeatureRouter(cfg,data['hidden'].shape[1],data['confidence'].shape[1],data['output'].shape[1]).cuda()
ckpt=torch.load(dest/'best.pt',map_location='cuda',weights_only=True)
model.load_state_dict(ckpt['state_dict'],strict=True)
model.eval().requires_grad_(False)
with torch.inference_mode():
    raw=torch.cat([model(data,torch.arange(i,min(i+8192,len(records)),device='cuda')).cpu() for i in range(0,len(records),8192)]).numpy()
scores=raw[:,1]-raw[:,2]
saved=np.load(work/'test_evaluation'/run/'paired_test_arrays.npz')['scores']
reference=json.loads((work/'test_results.json').read_text())
ref=next(g for g in reference['groups'] if g['id']==run)
point=ref['best_test_point']
threshold=point['threshold']
mask=scores>=threshold if point['operator']=='ge' else scores>threshold
saved_mask=saved>=threshold if point['operator']=='ge' else saved>threshold
assert np.array_equal(mask,saved_mask), 'Recomputed selected decisions differ from main-table archive'
indices=np.random.default_rng(20260912).choice(len(records),64,replace=False)
encoder_dir=a.root/'outputs/mini_siglip_strict_20260910_300ep/8M'
checkpoint=torch.load(encoder_dir/'epoch_300.pt',map_location='cpu',weights_only=True)
encoder=MiniSiglip(**checkpoint['model_config']).cuda()
encoder.load_state_dict(checkpoint['model'],strict=True)
encoder.eval().requires_grad_(False)
tokenizer=Tokenizer.from_file(str(encoder_dir/'student_tokenizer.json'))

def prepare(i):
    with Image.open(records[i]['image_path']) as im:
        pixels,valid=prepare_images([[im]])
    pixels=pixels.flatten(0,1).cuda()
    tokens=torch.tensor([tokenizer.encode(records[i]['expression']).ids],device='cuda') if 'text' in cfg['branches'] else None
    return pixels,valid.cuda(),tokens

def forward(i,prepared):
    pixels,valid,tokens=prepared
    with torch.autocast('cuda',dtype=torch.bfloat16):
        image=encoder.encode_image(pixels).to(torch.float16).float().reshape(1,2,768)
        text=encoder.encode_text(tokens).to(torch.float16).float() if tokens is not None else None
    row={k:v[i:i+1] for k,v in data.items()}
    row['image']=image;row['valid']=valid
    if text is not None:row['text']=text
    return model(row,torch.zeros(1,device='cuda',dtype=torch.long))

times=[];head_times=[];flops=[];ops=[]
with torch.inference_mode():
    for i in indices[:4]:
        forward(int(i),prepare(int(i)))
    torch.cuda.synchronize()
    for i in indices:
        i=int(i)
        started=time.perf_counter()
        forward(i,prepare(i))
        torch.cuda.synchronize()
        times.append((time.perf_counter()-started)*1000)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        idx=torch.tensor([i],device='cuda')
        start.record();model(data,idx);end.record();torch.cuda.synchronize()
        head_times.append(start.elapsed_time(end))
    for i in indices[:8]:
        prepared=prepare(int(i))
        with FlopCounterMode(display=False) as counter:
            forward(int(i),prepared)
        flops.append(counter.get_total_flops()/1e9)
        ops.append({str(k):int(v) for k,v in counter.get_flop_counts().get('Global',{}).items()})
report={'status':'complete','run':run,'config':cfg,'checkpoint':str(dest/'best.pt'),
 'score':'logit_rescue-logit_harm','test_rows_recomputed':len(records),'selected_decisions_match_main':True,
 'max_score_abs_difference':float(np.max(np.abs(saved-scores))),'actual_call_fraction':float(mask.mean()),
 'sample_count':len(indices),'sample_ids':[records[int(i)]['sample_id'] for i in indices],
 'batch_size':1,'latency_ms':times,'mean_latency_ms_per_input':float(np.mean(times)),
 'head_mean_latency_ms':float(np.mean(head_times)), 'gflops_per_input':flops,'mean_gflops_per_input':float(np.mean(flops)),
 'operators':ops,'scope':'image loading, bilinear resize, tokenize when used, two image slots (including gray placeholder), active distilled encoders, feature assembly and original routing head; cached specialist features; supported FLOPs only'}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k in ['run','actual_call_fraction','selected_decisions_match_main','mean_latency_ms_per_input','mean_gflops_per_input']},indent=2))
