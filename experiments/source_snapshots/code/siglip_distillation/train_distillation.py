"""Four independent experiments, each using four GPUs, sharing one CUDA image cache."""
from pathlib import Path
import argparse,csv,json,math,os,time,shutil
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datetime import timedelta
from torch.nn import functional as F
from mini_siglip import MiniSiglip,make_model

def worker(worker_id,specs,shared,args,barrier,stores):
    world=args.gpus_per_experiment
    experiment,gpu=divmod(worker_id,world)
    size=specs[experiment];torch.cuda.set_device(gpu);torch.set_num_threads(2)
    dist.init_process_group('nccl',init_method='file://'+stores[experiment],rank=gpu,world_size=world,device_id=torch.device(f'cuda:{gpu}'),timeout=timedelta(minutes=5))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    dest=Path(args.output)/f'{size}M';dest.mkdir(parents=True,exist_ok=True)
    model=make_model(size).to(gpu)
    counts={'total':sum(p.numel() for p in model.parameters()),
            'vision':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('patch','image_pos','vision','image_head'))),
            'text':sum(p.numel() for n,p in model.named_parameters() if n.startswith(('word','text')))}
    config=dict(model=model.config,parameters=counts,**vars(args),gpus=list(range(world)),world_size=world,size_millions=size)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    ni=len(shared['images']);nt=len(shared['text_ids']);steps=math.ceil(max(ni,nt)/args.batch_size)
    epochs=1 if args.smoke_steps else args.epochs
    if args.smoke_steps: steps=args.smoke_steps
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=steps*epochs,eta_min=0)
    begin=0
    checkpoint=dest/'last.pt'
    if args.resume:
        ck=torch.load(checkpoint,map_location=f'cuda:{gpu}',weights_only=False)
        if ck['model_config']!=model.config: raise ValueError('Checkpoint model differs')
        for key in ('batch_size','lr','epochs','weight_decay','seed'):
            if ck['training_args'][key]!=getattr(args,key): raise ValueError(f'Resume config changed: {key}')
        model.load_state_dict(ck['model']);optim.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);begin=ck['epoch']+1
    elif checkpoint.exists(): raise FileExistsError(f'Use --resume or a new output folder: {checkpoint}')
    complete=begin>=epochs
    if gpu==0:
        (dest/'config.json').write_text(json.dumps(config,indent=2))
        shutil.copyfile(Path(args.root)/'code/siglip_distillation/exp_settings.md',dest/'exp_settings.md')
        for name in ['student_tokenizer.json','ready.json']:
            shutil.copyfile(Path(args.root)/'data/siglip_distillation'/name,dest/name)
    model.train()
    ddp=DistributedDataParallel(model,device_ids=[gpu],broadcast_buffers=False)
    def take(key,idx):
        # Source storage belongs to the parent on GPU 0; only this minibatch is copied.
        return shared[key].index_select(0,idx.to(0)).to(gpu)
    barrier.wait()
    if complete:
        shared.clear();dist.destroy_process_group();print('ALREADY COMPLETE',size,begin,flush=True);return
    write_headers=not (dest/'loss_steps.csv').exists() or not args.resume
    with (dest/'loss_steps.csv' if gpu==0 else Path(os.devnull)).open('a' if args.resume else 'w',buffering=1,newline='') as log, (dest/'loss_epochs.csv' if gpu==0 else Path(os.devnull)).open('a' if args.resume else 'w',buffering=1,newline='') as epochlog:
        writer=csv.writer(log);ewriter=csv.writer(epochlog)
        if write_headers:
            writer.writerow(['epoch','step','samples','image_loss','text_loss','total_loss','lr','seconds'])
            ewriter.writerow(['epoch','image_loss','text_loss','total_loss','seconds','peak_gpu_gib'])
        print('START',size,gpu,counts,model.config,flush=True)
        for epoch in range(begin,epochs):
            gen=torch.Generator().manual_seed(args.seed+epoch)
            pi=torch.randperm(ni,generator=gen);pt=torch.randperm(nt,generator=gen)
            total=torch.zeros(3,device=gpu);seen=0;t0=time.monotonic()
            for step in range(steps):
                ts=time.monotonic()
                b=args.batch_size if args.smoke_steps else min(args.batch_size,max(ni,nt)-step*args.batch_size)
                # Partition one global batch; weight unequal final-batch partitions correctly.
                start=step*args.batch_size+b*gpu//world;end=step*args.batch_size+b*(gpu+1)//world
                ii=pi[torch.arange(start,end)%ni]
                ti=pt[torch.arange(start,end)%nt]
                x=take('images',ii).to(torch.bfloat16)/127.5-1
                ids=take('text_ids',ti).long()
                yi=take('image_targets',ii).float();yt=take('text_targets',ti).float()
                optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    si,st=ddp(x,ids)
                    li=(1-(si*yi).sum(-1)).sum();lt=(1-(st*yt).sum(-1)).sum()
                    loss=(li+lt)*(world/b)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {size} {epoch} {step}')
                loss.backward()
                gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                lr=optim.param_groups[0]['lr'];optim.step();scheduler.step()
                values=torch.stack([li.detach(),lt.detach(),li.detach()+lt.detach()])
                dist.all_reduce(values);values/=b
                total+=values*b;seen+=b
                writer.writerow([epoch+1,step+1,b,*values.tolist(),lr,time.monotonic()-ts])
                if gpu==0 and (step%20==0 or args.smoke_steps): print('LOSS',size,epoch+1,step+1,values.tolist(),'grad',gradnorm.item(),flush=True)
                if args.smoke_steps: print('RANK UPDATE',size,gpu,step+1,'local_samples',len(ii),flush=True)
            means=(total/seen).tolist()
            ewriter.writerow([epoch+1,*means,time.monotonic()-t0,torch.cuda.max_memory_allocated(gpu)/2**30])
            state={'model_config':model.config,'parameters':counts,'model':model.state_dict(),
                'optimizer':optim.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,
                'training_args':vars(args),'loss':means}
            if gpu==0:
                torch.save(state,str(checkpoint)+'.partial');os.replace(str(checkpoint)+'.partial',checkpoint)
            if gpu==0 and ((epoch+1)%25==0 or epoch+1==epochs):
                torch.save({'model_config':model.config,'model':model.state_dict(),'epoch':epoch+1,'parameters':counts},dest/f'epoch_{epoch+1:03d}.pt')
            print('EPOCH',size,epoch+1,means,flush=True)
            dist.barrier()
        # Verify the saved dual encoder can be loaded and used independently.
        saved=torch.load(checkpoint,map_location=f'cuda:{gpu}',weights_only=False)
        restored=MiniSiglip(**saved['model_config']).to(gpu);restored.load_state_dict(saved['model']);restored.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=restored.encode_image(x[:2]);b=restored.encode_text(ids[:2])
        assert a.shape==(2,768) and b.shape==(2,768) and torch.isfinite(a).all() and torch.isfinite(b).all()
        print('VERIFIED SAVED MODEL',size,flush=True)
        if args.smoke_steps:
            # Verify DDP replicas agree after updates using actual parameter values.
            flat=torch.cat([p.detach().float().flatten() for p in model.parameters()])
            reference=flat.clone();dist.broadcast(reference,src=0)
            difference=(flat-reference).abs().max();dist.all_reduce(difference,op=dist.ReduceOp.MAX)
            assert difference.item()==0,difference.item()
            if gpu==0: print('GPU REPLICAS IDENTICAL',size,world,flush=True)
    torch.cuda.synchronize(gpu)
    torch.cuda.synchronize(0)
    shared.clear()
    dist.destroy_process_group()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--experiment',action='append',type=int,required=True,help='millions; each experiment uses GPUs 0,1,2,3')
    ap.add_argument('--gpus-per-experiment',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=2048);ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--weight-decay',type=float,default=.01)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--smoke-steps',type=int,default=0)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    specs=args.experiment
    assert sorted(specs)==[1,2,4,8],specs
    assert args.batch_size%args.gpus_per_experiment==0,args.batch_size
    assert torch.cuda.device_count()>=args.gpus_per_experiment
    cache=Path(args.root)/'data/siglip_distillation'
    assert (cache/'teacher_ready.json').is_file()
    torch.set_num_threads(8);torch.cuda.set_device(0)
    for gpu in range(1,args.gpus_per_experiment):
        if not torch.cuda.can_device_access_peer(gpu,0): raise RuntimeError(f'GPU {gpu} cannot share GPU 0 cache')
    shared={}
    for key,file in [('images','images_uint8.npy'),('text_ids','text_ids.npy')]:
        shared[key]=torch.from_numpy(np.load(cache/file)).to(0)
    for kind in ['image','text']:
        arr=np.concatenate([np.load(cache/f'teacher_{kind}_{i}.npy') for i in range(4)])
        shared[kind+'_targets']=F.normalize(torch.from_numpy(arr).to(0).float(),dim=-1).half()
    print('SHARED GPU CACHE', {k:(list(v.shape),str(v.device),v.numel()*v.element_size()/2**30) for k,v in shared.items()},flush=True)
    torch.cuda.synchronize(0)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    stores=[str(Path(args.output).resolve()/f'distributed_{size}_{time.time_ns()}') for size in specs]
    workers=len(specs)*args.gpus_per_experiment
    barrier=mp.get_context('spawn').Barrier(workers)
    mp.spawn(worker,args=(specs,shared,args,barrier,stores),nprocs=workers,join=True)
    shared.clear()
    torch.cuda.empty_cache()
    print('ALL FOUR EXPERIMENTS COMPLETE',flush=True)
