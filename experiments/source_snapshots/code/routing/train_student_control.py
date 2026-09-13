"""Explicit distillation/depth controls, separate from the running strict rerun."""
import argparse,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')
sys.path.insert(0,str(ROOT/'code'))
from siglip_distillation.mini_siglip import MiniSiglip
from routing.distillation_objectives import relation_loss

def budget_width(millions,depth):
    # Exact count for the current dual encoder (4096 words, text64, output768).
    return min(range(32,1025,4),key=lambda w:abs(24*depth*w*w+(26*depth+2602)*w+132608-millions*1_000_000))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--smoke-steps',type=int,default=0)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    device=torch.device(args.device);cache=ROOT/'data/siglip_distillation_strict_20260910'
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    audit=json.loads((cache/'split_audit.json').read_text())
    assert audit['status']=='split_checks_passed' and audit['heldout_image_identity_overlap']==audit['heldout_normalized_text_overlap']==0
    assert (cache/'materialized_cache_verification.json').is_file()
    width=cfg['width'] if 'width' in cfg else budget_width(cfg['budget_millions'],cfg['depth'])
    model=MiniSiglip(width=width,depth=cfg['depth']).to(device)
    counts={'total':sum(p.numel() for p in model.parameters()),'vision':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('patch','image_pos','vision','image_head'))),'text':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('word','text')))}
    assert counts['total']==24*cfg['depth']*width*width+(26*cfg['depth']+2602)*width+132608
    work=ROOT/'outputs/router_exploration_25pct_20260910/distillation_controls'
    dest=work/('smoke_students' if args.smoke_steps else 'students')/cfg['id'];dest.mkdir(parents=True,exist_ok=False)
    (dest/'config.json').write_text(json.dumps({'model':model.config,'parameters':counts,'control_config':cfg,'smoke':bool(args.smoke_steps)},indent=2))
    heldout=cfg.get('heldout_task')
    inputs=cache
    if heldout is not None:
        assert heldout in ['cub','construction','grefcoco','nlvr2']
        inputs=work.parent/'task_holdout'/heldout
        selection=json.loads((inputs/'preparation.json').read_text())
        assert selection['heldout_task']==heldout and selection['source_cache']==str(cache)
        assert selection['heldout_source_images']==selection['heldout_normalized_text_overlap']==0
    shutil.copyfile(inputs/'student_tokenizer.json',dest/'student_tokenizer.json')
    pixels=np.load(cache/'images_uint8.npy',mmap_mode='r')
    tokens=torch.from_numpy(np.load(inputs/'text_ids.npy')).to(device=device,dtype=torch.long)
    # Match the strict reference: normalized teacher cache stored in float16,
    # then converted to float32 for loss evaluation.
    image_teacher=F.normalize(torch.from_numpy(np.concatenate([np.load(cache/f'teacher_image_{i}.npy') for i in range(4)])).to(device=device,dtype=torch.float32),dim=-1).half().float()
    text_teacher=F.normalize(torch.from_numpy(np.concatenate([np.load(cache/f'teacher_text_{i}.npy') for i in range(4)])).to(device=device,dtype=torch.float32),dim=-1).half().float()
    assert len(pixels)==len(image_teacher)==audit['images'] and len(text_teacher)==audit['texts']
    pixel_source=np.arange(len(pixels))
    text_source=np.arange(len(text_teacher))
    if heldout is not None:
        pixel_source=np.load(inputs/'image_source_indices.npy')
        text_source=np.load(inputs/'text_source_indices.npy')
        image_teacher=image_teacher[torch.tensor(pixel_source,device=device)]
        text_teacher=text_teacher[torch.tensor(text_source,device=device)]
        assert len(pixel_source)==selection['images'] and len(text_source)==selection['texts']
    ni,nt=len(pixel_source),len(tokens)
    assert ni==len(image_teacher) and nt==len(text_teacher)
    np.save(dest/'image_source_indices.npy',pixel_source);np.save(dest/'text_source_indices.npy',text_source)
    (dest/'data_selection.json').write_text(json.dumps({'heldout_task':heldout,'input_directory':str(inputs),'images':ni,'texts':nt,'source_cache':str(cache)}))
    if device.type=='cuda' and cfg['cache_pixels_on_gpu']:
        pixels=torch.from_numpy(np.array(pixels[pixel_source])).to(device)
    pairs=np.load(inputs/'relation_pairs.npy' if heldout is not None else work/'relation_data_official/pairs.npy')
    assert json.loads((work/'relation_data_official/completion.json').read_text())['status']=='complete'
    if any(stage['kind']=='relation' for stage in cfg['stages']):assert len(pairs)>0,'No eligible relation pairs; do not restore heldout-task data'
    started=time.monotonic();total_steps=0;stage_results=[]
    for stage_index,stage in enumerate(cfg['stages']):
        kind=stage['kind'];assert kind in ['image','joint','relation']
        relation=kind=='relation';n=len(pairs) if relation else (ni if kind=='image' else max(ni,nt))
        batch=cfg['batch_size'];steps=math.ceil(n/batch);epochs=1 if args.smoke_steps else stage['epochs']
        optimized=[]
        for name,p in model.named_parameters():
            active=kind!='image' or name.startswith(('patch','image_pos','vision','image_head'))
            p.requires_grad_(active)
            if active:optimized.append(p)
        optimizer=torch.optim.AdamW(optimized,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps*epochs)
        model.train();stage_start=time.monotonic()
        for epoch in range(epochs):
            generator=torch.Generator().manual_seed(cfg['seed']+stage_index*10000+epoch)
            image_order=torch.randperm(ni,generator=generator).numpy()
            text_order=torch.randperm(nt,generator=generator).numpy()
            pair_order=torch.randperm(len(pairs),generator=generator).numpy() if relation else None
            for step in range(steps):
                positions=np.arange(step*batch,min((step+1)*batch,n))
                if relation:ii,ti=pairs[pair_order[positions]].T
                else:ii=image_order[positions%ni];ti=text_order[positions%nt]
                image_ids=torch.tensor(ii,device=device);text_ids=torch.tensor(ti,device=device)
                x=pixels.index_select(0,image_ids) if isinstance(pixels,torch.Tensor) else torch.from_numpy(np.array(pixels[pixel_source[ii]])).to(device)
                x=x.to(torch.bfloat16 if device.type=='cuda' else torch.float32)/127.5-1
                ids=tokens[text_ids];yi=image_teacher[image_ids];yt=text_teacher[text_ids]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    si=model.encode_image(x);li=(1-(si*yi).sum(-1)).mean()
                    if kind=='image':lt=li*0;lr=li*0;loss=li
                    else:
                        st=model.encode_text(ids);lt=(1-(st*yt).sum(-1)).mean()
                        lr=relation_loss(si,st,yi,yt,cfg['temperature']) if relation else li*0
                        loss=lt+cfg['relation_weight']*lr if relation else li+lt
                if not torch.isfinite(loss):raise RuntimeError(('nonfinite distillation loss',cfg['id'],stage_index,epoch,step))
                loss.backward();torch.nn.utils.clip_grad_norm_(optimized,float('inf'),error_if_nonfinite=True);optimizer.step();scheduler.step();total_steps+=1
                row={'stage':kind,'stage_index':stage_index,'epoch':epoch+1,'step':step+1,'samples':len(ii),'image_alignment':li.item(),'text_alignment':lt.item(),'relation':lr.item(),'loss':loss.item(),'seconds':time.monotonic()-started}
                with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if step%20==0:print(row,flush=True)
                if args.smoke_steps and step+1>=args.smoke_steps:break
            torch.save({'model':model.state_dict(),'model_config':model.config,'parameters':counts,'control_config':cfg,'stage':kind,'epoch':epoch+1},dest/'last.pt')
        torch.save({'model':model.state_dict(),'model_config':model.config,'parameters':counts,'control_config':cfg,'stage':kind,'epoch':epochs},dest/f'stage_{stage_index}_{kind}.pt')
        stage_results.append({'kind':kind,'epochs':epochs,'seconds':time.monotonic()-stage_start})
    torch.save({'model':model.state_dict(),'model_config':model.config,'parameters':counts,'control_config':cfg},dest/'final.pt')
    restored=MiniSiglip(**model.config);restored.load_state_dict(torch.load(dest/'final.pt',map_location='cpu',weights_only=True)['model'])
    result={'status':'complete','smoke':bool(args.smoke_steps),'stages':stage_results,'steps':total_steps,'seconds':time.monotonic()-started,'parameters':counts,
            'initialization':'fresh random; no old student checkpoint loaded','test_rows_used':0,'validation_rows_used':0,'relation_data':str(inputs/'relation_pairs.npy') if heldout is not None else 'relation_data_official','heldout_task':heldout}
    (dest/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
