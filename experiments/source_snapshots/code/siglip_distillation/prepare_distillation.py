"""Only official training images and natural-language training inputs are used."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, argparse, io, collections
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT=Path('/mnt/HithinkOmniSSD/user_workspace/caisihang/research/ICASSP_ModelCollaboration')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); a=ap.parse_args()
    r=a.root; out=r/'data/siglip_distillation'; out.mkdir(exist_ok=True)
    if (out/'ready.json').exists():
        print((out/'ready.json').read_text()); return
    records={}; text_sources=collections.defaultdict(set)
    refs=json.loads((r/'data/grefcoco/official_annotations/grefs(unc).json').read_text())
    blocked={x['image_id'] for x in refs if x['split']!='train'}
    for x in refs:
        if x['split']!='train' or x['image_id'] in blocked: continue
        key='grefcoco/'+str(x['image_id'])
        records[key]={'source':'grefcoco','path':str(r/'data/grefcoco/images/train2014'/x['file_name'])}
        for s in x['sentences']: text_sources[s['raw']].add('grefcoco')
    nlvr_annotations=r/'data/nlvr2/annotations/data'
    heldout_urls=set()
    for split in ['dev','test1','test2']:
        for line in (nlvr_annotations/(split+'.json')).read_text().splitlines():
            x=json.loads(line);heldout_urls.update([x['left_url'],x['right_url']])
    for line in (nlvr_annotations/'train.json').read_text().splitlines():
        x=json.loads(line); stem=x['identifier'].rsplit('-',1)[0]
        for side in [0,1]:
            url=x['left_url' if side==0 else 'right_url']
            if url in heldout_urls:continue
            name=f'{stem}-img{side}.png';key='nlvr2/'+url
            records.setdefault(key,{'source':'nlvr2','path':str(r/'data/nlvr2/images/train'/str(x['directory'])/name)})
        text_sources[x['sentence']].add('nlvr2')
    cub=r/'data/cub200/official/CUB_200_2011'
    names=dict(line.split(maxsplit=1) for line in (cub/'images.txt').read_text().splitlines())
    for line in (cub/'train_test_split.txt').read_text().splitlines():
        idx,train=line.split()
        if train=='1': records['cub200/'+idx]={'source':'cub200','path':str(cub/'images'/names[idx])}
    construction=r/'data/constructionsite10k'
    meta=json.loads((construction/'official_repo/Annotations/dataset10k_metadata.json').read_text())
    print('Construction official split sizes', {k:len(v) for k,v in meta.items()},flush=True)
    ctrain=set(meta['training_split']); cseen=set()
    image_dir=construction/'extracted/train_images';image_dir.mkdir(exist_ok=True)
    for p in sorted((construction/'extracted').glob('train*.parquet')):
        for b in pq.ParquetFile(p).iter_batches(batch_size=64):
            for x in b.to_pylist():
                idx=x['image_id']
                if idx not in ctrain or idx in cseen: raise ValueError(f'Construction split mismatch or duplicate: {idx}')
                cseen.add(idx);p=image_dir/(idx+'.jpg')
                if not p.exists(): p.write_bytes(x['image']['bytes'])
                records['construction/'+idx]={'source':'construction','path':str(p)}
                caption=x['image_caption']
                if not isinstance(caption,str) or not caption.strip(): raise ValueError(f'Missing caption {idx}')
                text_sources[caption].add('construction')
    assert cseen==ctrain,(len(cseen),len(ctrain))
    counts=dict(collections.Counter(x['source'] for x in records.values()))
    assert set(counts)=={'grefcoco','nlvr2','cub200','construction'},counts
    rows=[dict(id=k,**v) for k,v in sorted(records.items())]
    (out/'images.json').write_text(json.dumps(rows,ensure_ascii=False))
    texts=sorted(text_sources)
    (out/'texts.json').write_text(json.dumps([{'text':t,'sources':sorted(text_sources[t])} for t in texts],ensure_ascii=False))
    tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator(texts,trainers.BpeTrainer(vocab_size=4096,
        special_tokens=['[PAD]','[UNK]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.enable_truncation(max_length=64); tokenizer.enable_padding(length=64,pad_id=0,pad_token='[PAD]')
    tokenizer.save(str(out/'student_tokenizer.json'))
    ids=np.asarray([v.ids for v in tokenizer.encode_batch(texts)],dtype=np.int32)
    np.save(out/'text_ids.npy',ids)
    pixels=np.lib.format.open_memmap(out/'images_uint8.npy',mode='w+',dtype=np.uint8,shape=(len(rows),3,224,224))
    def read_image(row):
        with Image.open(row['path']) as im:
            return np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).transpose(2,0,1)
    with ThreadPoolExecutor(max_workers=24) as pool:
        for i,x in enumerate(pool.map(read_image,rows)):
            pixels[i]=x
            if i%5000==0: print('IMAGES',i,len(rows),flush=True)
    pixels.flush()
    audit={'images':len(rows),'texts':len(texts),'images_by_dataset':counts,
        'text_sources':dict(collections.Counter(s for t in texts for s in text_sources[t])),
        'split':'official training only; gRefCOCO non-training image IDs and NLVR2 dev/test source URLs excluded; NLVR2 deduplicated by source URL',
        'preprocess':'PIL RGB bilinear 224x224; uint8 cached; (x/255-0.5)/0.5 at forward',
        'cub_text':'No class-label prompts; CUB contributes images only',
        'gpu_cache_gib':pixels.nbytes/2**30,'vocab_size':tokenizer.get_vocab_size()}
    (out/'ready.json').write_text(json.dumps(audit,indent=2)); print(audit,flush=True)

if __name__=='__main__': main()
