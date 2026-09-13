"""Frozen strict-student features; use new tokenizer and only frozen exploration manifests."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from siglip_distillation.mini_siglip import MiniSiglip
from routing.student_artifacts import student_artifacts

if __name__=='__main__':
    ap=argparse.ArgumentParser();choice=ap.add_mutually_exclusive_group(required=True)
    choice.add_argument('--size',choices=['1M','2M','4M','8M']);choice.add_argument('--control')
    ap.add_argument('--batch-size',type=int,default=256)
    a=ap.parse_args();torch.set_num_threads(4)
    artifact=student_artifacts(a.size,a.control);name=artifact['feature_key']
    work=ROOT/'outputs/router_exploration_25pct_20260910';dest=work/'features'/name
    if (dest/'completion.json').exists(): raise FileExistsError('Features already complete')
    modeldir=artifact['directory']
    checkpoint=torch.load(artifact['checkpoint'],weights_only=True,map_location='cpu')
    model=MiniSiglip(**checkpoint['model_config']);model.load_state_dict(checkpoint['model']);model.eval().requires_grad_(False).cuda()
    tokenizer=Tokenizer.from_file(str(modeldir/'student_tokenizer.json'))
    cache=ROOT/'outputs/router_training_cache_complete_20260908/input_cache'
    paths=json.loads((cache/'image_paths.json').read_text());path_to_id={p:i for i,p in enumerate(paths) if p is not None}
    pixels=np.load(cache/'images.npy',mmap_mode='r')
    report={'student':str(artifact['checkpoint']),'tokenizer':str(modeldir/'student_tokenizer.json'),'tasks':{},'test_samples_encoded':0}
    for task in ['cub','construction','nlvr2','grefcoco']:
        started=time.monotonic();parts={s:[json.loads(x) for x in (work/'manifests'/task/(s+'.jsonl')).read_text().splitlines()] for s in ['train','val']}
        selected_paths=set();texts=set();rows_by_split={}
        for split,rows in parts.items():
            inputs=[]
            for row in rows:
                if task=='nlvr2': pair=[row['image1'],row['image2']];text=row['sentence']
                else:
                    pair=[row['image_path'],None]
                    text=row['expression'] if task=='grefcoco' else ('Classify the bird species in this image.' if task=='cub' else 'Identify personal protective equipment, fall protection, unprotected edge and excavator proximity safety violations in this image.')
                assert text.strip()
                selected_paths.update(p for p in pair if p is not None);texts.add(text);inputs.append((pair,text))
            rows_by_split[split]=inputs
        missing=selected_paths-set(path_to_id)
        if missing: raise KeyError(('Images missing from verified input cache',task,sorted(missing)[:3]))
        ordered=[None]+sorted(selected_paths);image_index={p:i for i,p in enumerate(ordered)}
        image_features=np.empty((len(ordered),768),dtype=np.float16)
        sorted_texts=sorted(texts);text_index={t:i for i,t in enumerate(sorted_texts)};text_features=np.empty((len(sorted_texts),768),dtype=np.float16)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            gray=torch.full((1,3,224,224),128.,device='cuda')/127.5-1
            image_features[0]=model.encode_image(gray).float().cpu().numpy()[0]
            for i in range(1,len(ordered),a.batch_size):
                batch=ordered[i:i+a.batch_size];x=torch.from_numpy(np.array(pixels[[path_to_id[p] for p in batch]])).cuda().float()/127.5-1
                image_features[i:i+len(batch)]=model.encode_image(x).float().cpu().numpy()
                if i% (a.batch_size*40)==1: print(name,task,'images',i,len(ordered),flush=True)
            for i in range(0,len(sorted_texts),a.batch_size):
                batch=sorted_texts[i:i+a.batch_size]
                ids=torch.tensor([x.ids for x in tokenizer.encode_batch(batch)],device='cuda',dtype=torch.long)
                text_features[i:i+len(batch)]=model.encode_text(ids).float().cpu().numpy()
        assert np.isfinite(image_features).all() and np.isfinite(text_features).all()
        tokenizer.no_truncation()
        truncated=sum(len(x.ids)>64 for x in tokenizer.encode_batch(sorted_texts))
        tokenizer.enable_truncation(max_length=64)
        for split,inputs in rows_by_split.items():
            d=dest/task/split;d.mkdir(parents=True,exist_ok=True)
            np.save(d/'image.npy',image_features[[[image_index[p] for p in pair] for pair,text in inputs]])
            np.save(d/'valid.npy',np.array([[p is not None for p in pair] for pair,text in inputs],dtype=bool))
            np.save(d/'text.npy',text_features[[text_index[text] for pair,text in inputs]])
            (d/'sample_ids.json').write_text(json.dumps([r['sample_id'] for r in parts[split]]))
        report['tasks'][task]={'images_encoded':len(ordered),'unique_texts':len(sorted_texts),'truncated_unique_texts':truncated,'seconds':time.monotonic()-started}
        print(name,task,'COMPLETE',report['tasks'][task],flush=True)
    report['status']='complete';(dest/'completion.json').write_text(json.dumps(report,indent=2))
