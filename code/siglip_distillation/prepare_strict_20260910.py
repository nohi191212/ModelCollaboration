import os
"""Build fixed router splits and filter old frozen-teacher rows; never reuse a student."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ap=argparse.ArgumentParser()
ap.add_argument('--materialize', action='store_true')
args=ap.parse_args()
r=Path(os.environ.get('ICASSP_DATA_ROOT', str(Path(__file__).resolve().parents[2])))
old=r/'data/siglip_distillation'
out=r/'data/siglip_distillation_strict_20260910'
splits=r/'data/router_splits_strict_20260910'
out.mkdir(exist_ok=True); splits.mkdir(exist_ok=True)
base=r/'outputs/router_training_cache_complete_20260908/inputs'
manifests={task:{p.stem:[json.loads(s) for s in p.read_text().splitlines()] for p in (base/task).glob('*.jsonl')} for task in ['cub','construction','grefcoco','nlvr2']}
old_images=json.loads((old/'images.json').read_text())
old_texts=json.loads((old/'texts.json').read_text())

# CUB's cached images are recompressed and renamed. Match only within the known class,
# using actual image pixels, accepting only close, clearly unique matches. No hashes.
cub=r/'data/cub200/official/CUB_200_2011'
names=dict(s.split(maxsplit=1) for s in (cub/'images.txt').read_text().splitlines())
flags=dict(s.split() for s in (cub/'train_test_split.txt').read_text().splitlines())
official=list(names.items())
mapping_file=splits/'cub_image_correspondence.json'
if not mapping_file.exists():
    def read_thumbnail(item):
        with Image.open(item) as im:
            return im.size, np.asarray(im.convert('RGB').resize((32,32)),dtype=np.float32)
    with ThreadPoolExecutor(max_workers=16) as pool:
        official_pixels=list(pool.map(read_thumbnail,[cub/'images'/n for k,n in official]))
        cached_pixels=list(pool.map(read_thumbnail,[Path(x['image_path']) for x in manifests['cub']['train']]))
    by_class=collections.defaultdict(list)
    for i,(k,n) in enumerate(official): by_class[int(n.split('.')[0])].append(i)
    correspondence=[]
    for row,(size,x) in zip(manifests['cub']['train'],cached_pixels):
        candidates=by_class[row['ground_truth']['category_id']]
        distances=np.array([np.mean((x-official_pixels[i][1])**2) for i in candidates])
        order=np.argsort(distances); first,second=int(order[0]),int(order[1])
        i=candidates[first]
        official_size=official_pixels[i][0]
        if distances[first]>25 or abs((size[0]/size[1])/(official_size[0]/official_size[1])-1)>.02:
            raise ValueError(('Ambiguous CUB identity',row['sample_id'],float(distances[first]),float(distances[second]),size,official_pixels[i][0]))
        matches=[official[candidates[j]][0] for j in range(len(candidates)) if distances[j]<=25]
        correspondence.append({'sample_id':row['sample_id'],'official_ids':matches,'nearest_official_name':official[i][1],'pixel_mse_32':float(distances[first]),'next_mse_32':float(distances[second])})
    assert {k for x in correspondence for k in x['official_ids'] if flags[k]=='1'}=={k for k in names if flags[k]=='1'}
    mapping_file.write_text(json.dumps(correspondence,indent=2))
else:
    correspondence=json.loads(mapping_file.read_text())
assert {x['sample_id'] for x in correspondence}=={x['sample_id'] for x in manifests['cub']['train']}
mapped={x['sample_id']:x['official_ids'] for x in correspondence}
parents={k:k for k in names}
def representative(k):
    while parents[k]!=k:
        parents[k]=parents[parents[k]];k=parents[k]
    return k
for x in correspondence:
    anchor=x['official_ids'][0]
    for k in x['official_ids'][1:]: parents[representative(k)]=representative(anchor)
test_groups={representative(k) for k in names if flags[k]=='0'}
classes=collections.defaultdict(list)
excluded_cub_test=[]
for x in manifests['cub']['train']:
    if representative(mapped[x['sample_id']][0]) in test_groups: excluded_cub_test.append(x['sample_id'])
    else: classes[x['ground_truth']['category_id']].append(x)
cub_val=set()
rng=random.Random(42)
for category in sorted(classes):
    groups=collections.defaultdict(list)
    for x in classes[category]: groups[representative(mapped[x['sample_id']][0])].append(x['sample_id'])
    ordered=sorted(groups);rng.shuffle(ordered)
    count=0;target=round(len(classes[category])*.2)
    for group in ordered:
        if count>=target: break
        cub_val.update(groups[group]);count+=len(groups[group])
cube_train=[x for x in manifests['cub']['train'] if x['sample_id'] not in cub_val and x['sample_id'] not in excluded_cub_test]
cube_val=[dict(x,split='val',source_split='train') for x in manifests['cub']['train'] if x['sample_id'] in cub_val]
manifests['cub']['train']=cube_train;manifests['cub']['val']=cube_val
allowed_cub={'cub200/'+k for x in cube_train for k in mapped[x['sample_id']] if flags[k]=='1'}
blocked_cub={'cub200/'+k for x in cube_val for k in mapped[x['sample_id']]}|{'cub200/'+k for k in names if representative(k) in test_groups}

allowed_construction={'construction/'+x['sample_id'] for x in manifests['construction']['train']}
blocked_construction={'construction/'+x['sample_id'] for split,rows in manifests['construction'].items() if split!='train' for x in rows}
assert not allowed_construction & blocked_construction

refs=json.loads((r/'data/grefcoco/official_annotations/grefs(unc).json').read_text())
blocked_gref={'grefcoco/'+str(x['image_id']) for x in refs if x['split']!='train'}
allowed_gref={'grefcoco/'+str(x['image_id']) for x in manifests['grefcoco']['train']}-blocked_gref

ann=r/'data/nlvr2/annotations/data'
nlvr={s:[json.loads(x) for x in (ann/(s+'.json')).read_text().splitlines()] for s in ['train','dev','test1','test2']}
blocked_urls={x[k] for s,rows in nlvr.items() if s!='train' for x in rows for k in ['left_url','right_url']}
allowed_nlvr={'nlvr2/'+x[k] for x in nlvr['train'] for k in ['left_url','right_url'] if x[k] not in blocked_urls}
blocked_nlvr={'nlvr2/'+u for u in blocked_urls}
nlvr_by_id={x['identifier']:x for x in nlvr['train']}

# Reconstruct training text provenance instead of filtering a deduplicated text list by task.
allowed_text=collections.defaultdict(set)
blocked_text=set()
for x in refs:
    key='grefcoco/'+str(x['image_id'])
    for s in x['sentences']:
        if x['split']=='train' and key in allowed_gref: allowed_text[s['raw']].add('grefcoco')
        else: blocked_text.add(s['raw'])
for split,rows in nlvr.items():
    for x in rows:
        if split=='train' and x['left_url'] not in blocked_urls and x['right_url'] not in blocked_urls:
            allowed_text[x['sentence']].add('nlvr2')
        else: blocked_text.add(x['sentence'])
for p in sorted((r/'data/constructionsite10k/extracted').glob('*.parquet')):
    for batch in pq.ParquetFile(p).iter_batches(batch_size=2048,columns=['image_id','image_caption']):
        for x in batch.to_pylist():
            key='construction/'+x['image_id']
            if key in allowed_construction: allowed_text[x['image_caption']].add('construction')
            elif key in blocked_construction: blocked_text.add(x['image_caption'])

# Exact normalized heldout sentences/captions are excluded even if also seen in training.
blocked_normalized={' '.join(t.split()).casefold() for t in blocked_text}
new_texts=[{'text':t,'sources':sorted(s)} for t,s in sorted(allowed_text.items()) if ' '.join(t.split()).casefold() not in blocked_normalized]
allowed=allowed_cub|allowed_construction|allowed_gref|allowed_nlvr
blocked=blocked_cub|blocked_construction|blocked_gref|blocked_nlvr
assert not allowed & blocked
image_indices=[i for i,x in enumerate(old_images) if x['id'] in allowed]
new_images=[old_images[i] for i in image_indices]
assert {x['id'] for x in new_images}==allowed, ('Missing allowed teacher image rows',len(allowed),len(new_images))
assert not {x['id'] for x in new_images} & blocked
assert not {str(Path(x['path']).resolve()) for x in new_images if x['source']=='grefcoco'} & {str(Path(x['image_path']).resolve()) for split,rows in manifests['grefcoco'].items() if split!='train' for x in rows}
assert not {' '.join(x['text'].split()).casefold() for x in new_texts} & blocked_normalized
text_map={x['text']:i for i,x in enumerate(old_texts)}
text_indices=[text_map[x['text']] for x in new_texts]

# Save the actual future router split: remove training samples with heldout source images.
manifests['grefcoco']['train']=[x for x in manifests['grefcoco']['train'] if 'grefcoco/'+str(x['image_id']) in allowed_gref]
manifests['nlvr2']['train']=[x for x in manifests['nlvr2']['train'] if all(nlvr_by_id[x['sample_id']][k] not in blocked_urls for k in ['left_url','right_url'])]
for task,parts in manifests.items():
    d=splits/task;d.mkdir(exist_ok=True)
    for split,rows in parts.items():
        (d/(split+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
audit={'status':'split_checks_passed','seed':42,'cub_split':'per-class 80/20 from official training; cached images matched to official pixels',
       'cub_multiple_candidate_rows':sum(len(x['official_ids'])>1 for x in correspondence),
       'cub_train_rows_excluded_for_test_near_duplicates':excluded_cub_test,
       'cub_duplicate_rule':'All class-constrained thumbnail matches with MSE <=25 are grouped, never forced to one identity; groups touching official test excluded',
       'cub_mapping_max_thumbnail_mse':max(x['pixel_mse_32'] for x in correspondence),
       'images':len(new_images),'texts':len(new_texts),
       'images_by_dataset':dict(collections.Counter(x['source'] for x in new_images)),
       'removed_images_by_dataset':dict(collections.Counter(x['source'] for x in old_images if x['id'] not in allowed)),
       'removed_texts':len(old_texts)-len(new_texts),
       'router_split_counts':{task:{s:len(rows) for s,rows in p.items()} for task,p in manifests.items()},
       'heldout_image_identity_overlap':0,'heldout_normalized_text_overlap':0,
       'image_identity_rules':{'cub':'official image ID after class-constrained pixel matching','construction':'official image ID','grefcoco':'COCO image ID','nlvr2':'original source URL'},
       'scope':'all project train/validation/test manifests and supplied official identities; no claim of arbitrary perceptual duplicate search or external teacher pretraining audit',
       'student_initialization':'fresh random; old student weights forbidden',
       'teacher_reuse':'frozen teacher outputs filtered by image/text identity; no student weights reused',
       'student_tokenizer':'retrained only on filtered training texts',
       'split_directory':str(splits)}
(out/'split_audit.json').write_text(json.dumps(audit,indent=2))
(out/'images.json').write_text(json.dumps(new_images,ensure_ascii=False))
(out/'texts.json').write_text(json.dumps(new_texts,ensure_ascii=False))
(out/'blocked_image_ids.json').write_text(json.dumps(sorted(blocked)))
(out/'source_indices.json').write_text(json.dumps({'images':image_indices,'texts':text_indices}))
print(json.dumps(audit,indent=2),flush=True)
if not args.materialize: raise SystemExit(0)
if (out/'teacher_ready.json').exists(): raise FileExistsError('New cache already materialized')

tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder=decoders.ByteLevel()
tokenizer.train_from_iterator([x['text'] for x in new_texts],trainers.BpeTrainer(vocab_size=4096,special_tokens=['[PAD]','[UNK]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tokenizer.enable_truncation(max_length=64);tokenizer.enable_padding(length=64,pad_id=0,pad_token='[PAD]')
tokenizer.save(str(out/'student_tokenizer.json'))
ids=np.asarray([x.ids for x in tokenizer.encode_batch([x['text'] for x in new_texts])],dtype=np.int32)
np.save(out/'text_ids.npy',ids)
src=np.load(old/'images_uint8.npy',mmap_mode='r')
dst=np.lib.format.open_memmap(out/'images_uint8.npy',mode='w+',dtype=np.uint8,shape=(len(new_images),3,224,224))
for start in range(0,len(new_images),1024):
    end=min(start+1024,len(new_images));dst[start:end]=src[image_indices[start:end]]
    if start%10240==0: print('COPY_IMAGES',start,len(new_images),flush=True)
dst.flush(); del dst,src
for kind,indices,old_rows in [('image',image_indices,old_images),('text',text_indices,old_texts)]:
    arrays=[np.load(old/f'teacher_{kind}_{i}.npy',mmap_mode='r') for i in range(4)]
    assert sum(len(a) for a in arrays)==len(old_rows)
    full=np.concatenate(arrays)[indices]
    assert np.isfinite(full).all()
    for i in range(4): np.save(out/f'teacher_{kind}_{i}.npy',full[len(full)*i//4:len(full)*(i+1)//4])
    print('TEACHER_ROWS_REUSED',kind,len(full),flush=True)
ready=dict(audit,preprocess='PIL RGB bilinear 224x224; uint8 cached; x/127.5-1',vocab_size=tokenizer.get_vocab_size())
(out/'ready.json').write_text(json.dumps(ready,indent=2))
(out/'teacher_ready.json').write_text(json.dumps({'teacher':'google/siglip2-base-patch16-224','shards':4,'dtype':'float16','normalized':True,'filtered_only':True}))
print('STRICT_CACHE_READY',time.time(),flush=True)
