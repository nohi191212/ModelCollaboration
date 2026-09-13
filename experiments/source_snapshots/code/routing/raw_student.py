"""Differentiable strict student inputs: raw cached pixels, fresh strict tokenizer."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from siglip_distillation.mini_siglip import MiniSiglip
from routing.student_artifacts import student_artifacts

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

class RawStudent(nn.Module):
    def __init__(self,cfg,records,device):
        super().__init__();self.branches=cfg['branches'];self.device=device
        assert set(self.branches)&{'image','text'}, 'Fine-tuning requires an active student branch'
        artifact=student_artifacts(cfg['student'],cfg.get('student_control'));source=artifact['directory']
        checkpoint=torch.load(artifact['checkpoint'],map_location='cpu',weights_only=True)
        self.student=MiniSiglip(**checkpoint['model_config']);self.student.load_state_dict(checkpoint['model']);self.student.to(device)
        image_prefixes=('patch.','image_pos','vision.','image_head.')
        for name,p in self.student.named_parameters():
            p.requires_grad_('image' in self.branches if name.startswith(image_prefixes) else 'text' in self.branches)
        tokenizer=Tokenizer.from_file(str(source/'student_tokenizer.json'))
        cache=ROOT/'outputs/router_training_cache_complete_20260908/input_cache'
        paths=json.loads((cache/'image_paths.json').read_text());path_to_id={p:i for i,p in enumerate(paths) if p is not None}
        self.pixels=np.load(cache/'images.npy',mmap_mode='r');self.inputs={}
        for split,rows in records.items():
            pairs=[];texts=[]
            for row in rows:
                if cfg['task']=='nlvr2':pair=[row['image1'],row['image2']];text=row['sentence']
                else:
                    pair=[row['image_path'],None]
                    text=row['expression'] if cfg['task']=='grefcoco' else ('Classify the bird species in this image.' if cfg['task']=='cub' else 'Identify personal protective equipment, fall protection, unprotected edge and excavator proximity safety violations in this image.')
                pairs.append([path_to_id[p] if p is not None else -1 for p in pair]);texts.append(text)
            ids=np.array([x.ids for x in tokenizer.encode_batch(texts)],dtype=np.int64)
            assert ids.ndim==2 and ids.shape[1]==64
            self.inputs[split]={'pixels':np.array(pairs,dtype=np.int64),'tokens':ids}

    def forward(self,split,indices):
        ids=indices.detach().cpu().numpy();inputs=self.inputs[split];result={}
        if 'image' in self.branches:
            pairs=inputs['pixels'][ids];valid=pairs>=0
            pixels=np.empty((len(ids),2,3,224,224),dtype=np.uint8);pixels.fill(128)
            pixels[valid]=self.pixels[pairs[valid]]
            images=torch.from_numpy(pixels).to(self.device,dtype=torch.float32).flatten(0,1)/127.5-1
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                result['image']=self.student.encode_image(images).to(torch.float16).float().reshape(len(ids),2,768)
            result['valid']=torch.from_numpy(valid).to(self.device)
        if 'text' in self.branches:
            tokens=torch.from_numpy(inputs['tokens'][ids]).to(self.device)
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                result['text']=self.student.encode_text(tokens).to(torch.float16).float()
        return result
