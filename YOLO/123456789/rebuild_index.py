# -*- coding: utf-8 -*-
"""重建所有派生索引文件（数据集重新划分后必须跑一次）。

数据集一旦重新划分（图片换目录），下面这些“派生文件”就会全部失效：
    labeled_train.txt / labeled_val.txt     —— 已标注图片清单
    label_worklist.csv                      —— 每张图的来源类别与是否已有标签
    todo_no_label.txt / todo_obstacle.txt   —— 待标注清单
    label_plan.csv / label_batches/*.txt    —— 分批标注计划
本脚本按当前 dataset 的真实状态把它们全部重建。

用法（在 yolo_project 下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe rebuild_index.py
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe rebuild_index.py "原始图片目录"
"""
import csv
import sys
from collections import Counter, OrderedDict
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
DEFAULT_RAW = PROJ.parent / '2026AutumnEngineeringChallenges-main' / \
    '2026AutumnEngineeringChallenges-main' / 'YOLO'
SPLITS = ('train', 'val')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
CLASSES = ('obstacle', 'cola', 'football')
BATCH_SIZE = 40


def load_source_classes(raw_root):
    mapping = {}
    if raw_root and raw_root.exists():
        for cls in CLASSES:
            d = raw_root / cls
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in EXTS:
                        mapping[p.stem] = cls
    return mapping


def has_class0(lbl):
    if not lbl.exists():
        return False
    for line in lbl.read_text(encoding='utf-8', errors='ignore').splitlines():
        parts = line.split()
        if len(parts) == 5 and int(float(parts[0])) == 0:
            return True
    return False


def parse_session(name):
    stem = name.rsplit('.', 1)[0]
    parts = stem.split('_')
    date = hour = ''
    for i, p in enumerate(parts):
        if p.isdigit() and len(p) == 8:
            date = p
            if i + 1 < len(parts) and len(parts[i + 1]) >= 2:
                hour = parts[i + 1][:2] + 'h'
            break
    prefix = '_'.join(parts[:parts.index(date)]) if date in parts else stem
    return '%s %s %s' % (prefix, date, hour)


def main():
    raw_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RAW
    src_cls = load_source_classes(raw_root)
    print('原始图片目录: %s  已映射 %d 张' % (raw_root, len(src_cls)))

    rows = []          # 每张图的完整信息
    for split in SPLITS:
        img_dir = DATASET / 'images' / split
        lbl_dir = DATASET / 'labels' / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in EXTS:
                continue
            lbl = lbl_dir / (img.stem + '.txt')
            rows.append({
                'split': split,
                'image': img.name,
                'path': str(img),
                'source_class': src_cls.get(img.stem, 'unknown'),
                'has_label': lbl.exists(),
                'has_obstacle': has_class0(lbl),
            })

    # 1) 已标注图片清单
    for split in SPLITS:
        paths = [r['path'] for r in rows if r['split'] == split and r['has_label']]
        (PROJ / ('labeled_%s.txt' % split)).write_text('\n'.join(paths) + '\n', encoding='utf-8')
        print('labeled_%s.txt: %d 张' % (split, len(paths)))

    # 2) 工作清单
    with (PROJ / 'label_worklist.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['split', 'image', 'source_class', 'has_draft_label'])
        for r in rows:
            w.writerow([r['split'], r['image'], r['source_class'], int(r['has_label'])])
    print('label_worklist.csv: %d 行' % len(rows))

    # 3) 待办清单
    no_label = [r for r in rows if not r['has_label']]
    (PROJ / 'todo_no_label.txt').write_text(
        '\n'.join('%s/%s' % (r['split'], r['image']) for r in no_label) + '\n', encoding='utf-8')
    todo_ob = [r for r in rows if r['source_class'] == 'obstacle' and not r['has_obstacle']]
    (PROJ / 'todo_obstacle.txt').write_text(
        '\n'.join('%s/%s' % (r['split'], r['image']) for r in todo_ob) + '\n', encoding='utf-8')
    print('todo_no_label.txt: %d 张；todo_obstacle.txt: %d 张' % (len(no_label), len(todo_ob)))

    # 4) 分批标注计划（优先级 + 会话聚类）
    plan = []
    for r in todo_ob:
        if r['has_label']:
            prio = 'P1'
        elif r['split'] == 'train':
            prio = 'P2'
        else:
            prio = 'P3'
        plan.append(dict(r, priority=prio, session=parse_session(r['image'])))
    plan.sort(key=lambda x: (x['priority'], x['session'], x['image']))

    out_dir = PROJ / 'label_batches'
    out_dir.mkdir(exist_ok=True)
    batches = OrderedDict()
    for i in range(0, len(plan), BATCH_SIZE):
        batches['batch_%02d' % (i // BATCH_SIZE + 1)] = plan[i:i + BATCH_SIZE]
    for bname, items in batches.items():
        (out_dir / (bname + '.txt')).write_text(
            '\n'.join(r['path'] for r in items) + '\n', encoding='utf-8')

    with (PROJ / 'label_plan.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['batch', 'priority', 'split', 'image', 'session', 'has_draft'])
        for bname, items in batches.items():
            for r in items:
                w.writerow([bname, r['priority'], r['split'], r['image'], r['session'],
                            int(r['has_label'])])
    # batch_01 明细（含已有框信息）
    first = batches.get('batch_01', [])
    with (out_dir / 'batch_01_detail.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['split', 'image', 'session', 'boxes', 'n'])
        for r in first:
            lbl = DATASET / 'labels' / r['split'] / (Path(r['image']).stem + '.txt')
            cls = []
            if lbl.exists():
                for line in lbl.read_text(encoding='utf-8').splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        cls.append(parts[0])
            w.writerow([r['split'], r['image'], r['session'], '+'.join(cls), len(cls)])

    print()
    print('重新划分后的实际分布:')
    for split in SPLITS:
        sub = [r for r in rows if r['split'] == split]
        cnt = Counter(r['source_class'] for r in sub)
        print('  [%s] 图片 %d  (obstacle %d / cola %d / football %d)；已标注 %d；含obstacle框 %d'
              % (split, len(sub), cnt['obstacle'], cnt['cola'], cnt['football'],
                 sum(1 for r in sub if r['has_label']), sum(1 for r in sub if r['has_obstacle'])))
    print()
    print('标注批次: %d 批 -> %s' % (len(batches), ', '.join(
        '%s(%d)' % (k, len(v)) for k, v in batches.items())))
    print('优先级统计:', dict(Counter(r['priority'] for r in plan)))
    print('派生文件已全部重建完成。')


if __name__ == '__main__':
    main()
