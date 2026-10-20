"""Copy manually labeled data into a date-separated YOLO dataset."""
from pathlib import Path
from collections import Counter
import shutil,json,hashlib
ROOT=Path(__file__).resolve().parent

def main():
    if (ROOT/'dataset').exists():
        raise RuntimeError('dataset already exists; do not overwrite an existing split.')
    image_index={}
    for p in (ROOT/'raw').rglob('*.jpg'):
        image_index.setdefault(p.stem,[]).append(p)
    labels=sorted(p for p in (ROOT/'annotations').glob('*.txt') if p.name!='classes.txt')
    records=[]
    for label in labels:
        matches=image_index.get(label.stem,[])
        if len(matches)!=1: raise ValueError('Image matching failed: '+label.name)
        cls_counts=Counter()
        for line in label.read_text().splitlines():
            if not line.strip():continue
            v=list(map(float,line.split()))
            if len(v)!=5 or v[0] not in (0,1,2): raise ValueError('Invalid label '+label.name)
            c,x,y,w,h=v
            if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1): raise ValueError('Invalid coordinates '+label.name)
            cls_counts[int(c)]+=1
        split='val' if label.stem[-15:-7]=='20260728' else 'train'
        records.append(dict(source_image=str(matches[0].relative_to(ROOT)),source_label=str(label.relative_to(ROOT)),split=split,objects=dict(cls_counts)))
    summary={}
    for split in ['train','val']:
        subset=[r for r in records if r['split']==split]
        counts=Counter()
        for r in subset: counts.update(r['objects'])
        if not subset or any(counts[c]==0 for c in [0,1,2]): raise ValueError('Split must contain all three classes')
        summary[split]=dict(images=len(subset),objects=dict(counts))
    image_hashes={s:set() for s in ['train','val']}
    for row in records:
        image=ROOT/row['source_image']; label=ROOT/row['source_label']; split=row['split']
        image_hashes[split].add(hashlib.sha256(image.read_bytes()).hexdigest())
        for kind,source in [('images',image),('labels',label)]:
            dest=ROOT/'dataset'/kind/split/source.name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
    assert not image_hashes['train'].intersection(image_hashes['val'])
    config='path: G:/YOLO学习项目/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: obstacle\n  1: cola\n  2: football\n'
    (ROOT/'data.yaml').write_text(config,encoding='utf-8')
    manifest=dict(strategy='All labeled 2026-07-28 images in validation; earlier dates in training. Same laboratory background may still limit generalization.',summary=summary,records=records)
    (ROOT/'dataset_split.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Original manual labels and raw images preserved; no identical image appears across splits.')

if __name__=='__main__': main()
