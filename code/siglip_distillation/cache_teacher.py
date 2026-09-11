from pathlib import Path
import argparse,json
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F
from transformers import AutoTokenizer, SiglipModel

def worker(rank,root):
    torch.set_num_threads(8); torch.cuda.set_device(rank)
    r=Path(root); cache=r/'data/siglip_distillation'; modeldir=r/'models/siglip2-base-patch16-224'
    model=SiglipModel.from_pretrained(modeldir,local_files_only=True,dtype=torch.bfloat16).eval().to(rank)
    for p in model.parameters(): p.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(modeldir,local_files_only=True)
    images=np.load(cache/'images_uint8.npy',mmap_mode='r')
    texts=[x['text'] for x in json.loads((cache/'texts.json').read_text())]
    with torch.inference_mode():
        for kind,source in [('image',images),('text',texts)]:
            start=len(source)*rank//4; end=len(source)*(rank+1)//4
            target=cache/f'teacher_{kind}_{rank}.npy'
            arr=np.lib.format.open_memmap(str(target)+'.partial',mode='w+',dtype=np.float16,shape=(end-start,768))
            for i in range(start,end,256):
                batch=source[i:min(i+256,end)]
                if kind=='image':
                    x=torch.from_numpy(np.array(batch)).to(rank,dtype=torch.bfloat16)/127.5-1
                    y=model.vision_model(pixel_values=x).pooler_output
                else:
                    inputs=tokenizer(batch,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(rank)
                    y=model.text_model(**inputs).pooler_output
                y=F.normalize(y.float(),dim=-1)
                if not torch.isfinite(y).all(): raise RuntimeError(f'Nonfinite teacher features {kind} {i}')
                arr[i-start:i-start+len(batch)]=y.cpu().numpy()
                if (i-start)//256%20==0: print('TEACHER',rank,kind,i-start,end-start,flush=True)
            arr.flush(); del arr; Path(str(target)+'.partial').rename(target)
    print('TEACHER DONE',rank,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); args=ap.parse_args()
    cache=Path(args.root)/'data/siglip_distillation'
    if not (cache/'teacher_ready.json').exists():
        mp.spawn(worker,args=(args.root,),nprocs=4,join=True)
        (cache/'teacher_ready.json').write_text(json.dumps({'teacher':'google/siglip2-base-patch16-224','shards':4,'dtype':'float16','normalized':True}))
