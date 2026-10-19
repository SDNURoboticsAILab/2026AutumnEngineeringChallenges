# -*- coding: utf-8 -*-
"""精修机器自动标注：按类别设不同置信度阈值 + 跨类重叠消解。

背景：bootstrap 模型的 cola/football 很准（mAP50 0.90 / 0.86），但 obstacle 很弱
（mAP50 0.32，只有 41 个训练实例），于是它会在足球上又画一个 obstacle 框，
造成 53 处跨类重叠（同一目标两个类别）——这对训练有害。

策略：
  1) 只处理“机器刚生成的”标签文件（人工标注的文件一律不碰，以备份目录为准）
  2) 低阈值推理拿到候选框，再按类别阈值过滤：
         obstacle >= 0.45（弱类从严）
         cola     >= 0.40
         football >= 0.30
  3) 跨类重叠消解：按置信度从高到低接收，若与已接收的“不同类别”框 IoU > 0.5 则丢弃
  4) 过滤后没有框的图片 -> 删掉机器生成的标签文件（恢复为未标注，交给人来处理）

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe refine_auto_labels.py
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
LABELS = DATASET / 'labels'
WEIGHTS = PROJ / 'runs' / 'detect' / 'bootstrap' / 'weights' / 'best.pt'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}

THRESH = {0: 0.45, 1: 0.40, 2: 0.30}
CROSS_IOU = 0.50      # 不同类别框的 IoU 超过它就丢弃置信度低的那个
PROBE_CONF = 0.10     # 先低阈值把所有候选捞出来


def latest_backup():
    dirs = sorted(DATASET.glob('_labels_backup_*'))
    return dirs[-1] if dirs else None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(x, y, w, h):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def main():
    from ultralytics import YOLO

    bak = latest_backup()
    if bak is None:
        print('找不到备份目录，终止（需要备份来区分人工/机器标注）')
        return 2
    human = set()
    for split in ('train', 'val'):
        d = bak / split
        if d.exists():
            human |= {('%s/%s' % (split, f.name)) for f in d.glob('*.txt')}
    print('以备份为基准：人工标注 %d 个文件 -> %s' % (len(human), bak.name), flush=True)

    auto = []
    for split in ('train', 'val'):
        for f in (LABELS / split).glob('*.txt'):
            key = '%s/%s' % (split, f.name)
            if key not in human and f.name != 'classes.txt':
                auto.append((split, f.stem))
    print('机器生成、需要精修的标签文件: %d 个' % len(auto), flush=True)
    if not auto:
        return 0

    model = YOLO(str(WEIGHTS))
    kept = Counter()
    deleted = 0
    dropped_conf = 0
    dropped_conflict = 0

    CHUNK = 50
    for i in range(0, len(auto), CHUNK):
        chunk = auto[i:i + CHUNK]
        paths = [str(DATASET / 'images' / s / (n + '.jpg')) for s, n in chunk]
        results = model.predict(source=paths, imgsz=640, conf=PROBE_CONF,
                                iou=0.7, batch=8, device=0, verbose=False, save=False)
        for (split, stem), r in zip(chunk, results):
            cands = []
            for box in r.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < THRESH.get(cid, 0.5):
                    dropped_conf += 1
                    continue
                x, y, w, h = box.xywhn[0].tolist()
                cands.append((conf, cid, x, y, w, h))
            cands.sort(reverse=True)      # 置信度从高到低

            accepted = []
            for conf, cid, x, y, w, h in cands:
                box = xywh_to_xyxy(x, y, w, h)
                conflict = False
                for a in accepted:
                    if a[1] == cid:
                        continue
                    if iou(box, xywh_to_xyxy(a[2], a[3], a[4], a[5])) > CROSS_IOU:
                        conflict = True
                        break
                if conflict:
                    dropped_conflict += 1
                    continue
                accepted.append((conf, cid, x, y, w, h))

            out = LABELS / split / (stem + '.txt')
            if accepted:
                out.write_text('\n'.join('%d %.6f %.6f %.6f %.6f' % (c, x, y, w, h)
                                         for _, c, x, y, w, h in accepted) + '\n',
                               encoding='utf-8')
                kept[split] += 1
                for _, c, *_ in accepted:
                    kept['cls_%d' % c] += 1
            else:
                out.unlink(missing_ok=True)
                deleted += 1
        print('  进度 %d/%d' % (min(i + CHUNK, len(auto)), len(auto)), flush=True)

    print('\n=== 精修结果 ===', flush=True)
    print('保留标签的图片: train %d, val %d' % (kept['train'], kept['val']), flush=True)
    print('按类别保留的框: ' + ', '.join('%s=%d' % (NAMES[k - 0], kept['cls_%d' % k])
                                     for k in (0, 1, 2)), flush=True)
    print('置信度不足被丢弃的框: %d' % dropped_conf, flush=True)
    print('跨类重叠被丢弃的框: %d' % dropped_conflict, flush=True)
    print('过滤后无框、已恢复为未标注的图片: %d' % deleted, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
