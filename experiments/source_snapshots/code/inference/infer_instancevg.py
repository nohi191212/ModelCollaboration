import argparse
import json
import re
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from transformers import XLMRobertaTokenizer
from instancevg.models import build_model
from instancevg.datasets.pipelines.transforms import Normalize, Resize

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--split', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--limit', type=int)
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(20260908)
np.random.seed(20260908)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
source = args.root / 'code/third_party/InstanceVG'
cfg = Config.fromfile(str(source / 'configs/gres/InstanceVG-grefcoco.py'))
cfg.model.vis_enc.pretrain = None
cfg.model.process_visual = False
model = build_model(cfg.model)
checkpoint_path = args.root / 'models/grefcoco/instancevg/InstanceVG-grefcoco.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state = checkpoint['state_dict']
if all(k.startswith('module.') for k in state):
    state = {k[7:]: v for k, v in state.items()}
model.load_state_dict(state, strict=True)
print('STRICT_LOAD_OK', len(state), 'parameters', sum(p.numel() for p in model.parameters()), flush=True)
del state, checkpoint
model.eval().cuda()
tokenizer_path = args.root / 'models/nlvr2/Raghavan--beit3_base_patch16_224_nlvr2/sentencepiece.bpe.model'
tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
assert tokenizer.vocab_size == 64002, tokenizer.vocab_size
resize = Resize(img_scale=(320, 320), keep_ratio=False)
normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
input_path = args.root / 'outputs/experiments/20260827_grefcoco_native_grounding/prepared' / f'grefcoco_{args.split}_inputs.jsonl'
rows = [json.loads(line) for line in input_path.read_text().splitlines()]
if args.limit is not None:
    rows = rows[:args.limit]
args.output.parent.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
with args.output.open('x') as out, torch.inference_mode():
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        images, metas, tokens_batch, masks = [], [], [], []
        for row in batch:
            image = mmcv.imread(row['image_path'])
            assert image.shape[:2] == (row['height'], row['width']), row['sample_id']
            expression = re.sub(r"([.,'!?\"()*#:;])", '', row['expression'].lower()).replace('-', ' ').replace('/', ' ')
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(expression))
            if not token_ids:
                raise ValueError(f"empty token sequence: {row['sample_id']}")
            token_ids = [tokenizer.bos_token_id] + token_ids[:48] + [tokenizer.eos_token_id]
            masks.append([0] * len(token_ids) + [1] * (50 - len(token_ids)))
            tokens_batch.append(token_ids + [tokenizer.pad_token_id] * (50 - len(token_ids)))
            data = dict(img=image, filename=row['image_path'], img_shape=image.shape, ori_shape=image.shape,
                        expression=expression, empty=None, with_bbox=False, with_mask=False)
            data = normalize(resize(data))
            images.append(torch.from_numpy(np.ascontiguousarray(data['img'].transpose(2, 0, 1))))
            metas.append({k: data[k] for k in ['filename', 'expression', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'empty']})
        predictions = model(img=torch.stack(images).cuda(), ref_expr_inds=torch.tensor(tokens_batch, device='cuda'),
                            text_attention_mask=torch.tensor(masks, device='cuda'), img_metas=metas,
                            return_loss=False, rescale=True, with_bbox=True, with_mask=False)
        for row, pred in zip(batch, predictions['pred_bboxes']):
            boxes, scores = pred['boxes'].cpu(), pred['scores'].cpu()
            assert torch.isfinite(boxes).all() and torch.isfinite(scores).all(), row['sample_id']
            assert len(boxes) == len(scores), row['sample_id']
            objects = [{'bbox_xyxy_absolute': box, 'score': score} for box, score in zip(boxes.tolist(), scores.tolist())]
            out.write(json.dumps(dict(sample_id=row['sample_id'], split=row['split'], image_id=row['image_id'],
                                      model='InstanceVG-grefcoco', objects=objects), ensure_ascii=False) + '\n')
        out.flush()
        if offset == 0 or (offset + len(batch)) % 100 == 0 or offset + len(batch) == len(rows):
            print(json.dumps({'done': offset + len(batch), 'total': len(rows), 'elapsed_seconds': round(time.monotonic() - start, 2),
                              'peak_gpu_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}), flush=True)
summary = dict(status='complete', split=args.split, rows=len(rows), seconds=time.monotonic() - start,
               batch_size=args.batch_size, precision='float32', score_threshold=0.7, strict_weights=True,
               tokenizer=str(tokenizer_path), peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
