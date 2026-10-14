# -*- coding: utf-8 -*-
"""把 todo_obstacle.txt 拆成可执行的标注批次。

优先级规则：
  P1  已有草稿标签（cola/football 已框好）—— 只需补 class 0，最快见效
  P2  train 中完全没有标签的图 —— 训练主力，必须补
  P3  val 中完全没有标签的图 —— 验证集也要有 obstacle，否则测不出这一类
排序规则：先优先级，再按“采集会话”（相机前缀 + 日期 + 小时）聚在一起，
同一会话画面高度相似，连着标标准最一致、速度最快。

产出：
  label_batches/batch_01.txt ... 每批一份图片绝对路径清单
  label_plan.csv                 完整的批次计划表
"""
from collections import Counter, OrderedDict
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
TODO = PROJ / 'todo_obstacle.txt'
OUT_DIR = PROJ / 'label_batches'
BATCH_SIZE = 40


def parse(name):
    """image_rgb_20260727_143017.jpg -> ('image_rgb', '20260727', '14h')"""
    stem = name.rsplit('.', 1)[0]
    parts = stem.split('_')
    date = ''
    hour = ''
    for i, p in enumerate(parts):
        if p.isdigit() and len(p) == 8:
            date = p
            if i + 1 < len(parts) and len(parts[i + 1]) >= 2:
                hour = parts[i + 1][:2] + 'h'
            break
    prefix = '_'.join(parts[:parts.index(date)]) if date in parts else stem
    return prefix, date, hour


def main():
    rows = []
    for line in TODO.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        split, name = line.split('/', 1)
        has_draft = (DATASET / 'labels' / split / (Path(name).stem + '.txt')).exists()
        prefix, date, hour = parse(name)
        session = '%s %s %s' % (prefix, date, hour)
        if has_draft:
            prio = 'P1'
        elif split == 'train':
            prio = 'P2'
        else:
            prio = 'P3'
        rows.append({'split': split, 'image': name, 'priority': prio,
                     'session': session, 'has_draft': has_draft, 'path': str(DATASET / 'images' / split / name)})

    rows.sort(key=lambda r: (r['priority'], r['session'], r['image']))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batches = OrderedDict()
    for i in range(0, len(rows), BATCH_SIZE):
        batches['batch_%02d' % (i // BATCH_SIZE + 1)] = rows[i:i + BATCH_SIZE]

    for bname, items in batches.items():
        (OUT_DIR / (bname + '.txt')).write_text(
            '\n'.join(r['path'] for r in items) + '\n', encoding='utf-8')

    with (PROJ / 'label_plan.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        fh.write('batch,priority,split,image,session,has_draft\n')
        for bname, items in batches.items():
            for r in items:
                fh.write('%s,%s,%s,%s,%s,%d\n' % (bname, r['priority'], r['split'],
                                                  r['image'], r['session'], int(r['has_draft'])))

    print('total %d images -> %d batches (batch size %d)' % (len(rows), len(batches), BATCH_SIZE))
    print('=' * 72)
    print('%-9s %4s  %-18s %-14s %s' % ('batch', 'n', 'split', 'priority', 'sessions'))
    for bname, items in batches.items():
        sp = Counter(r['split'] for r in items)
        pr = Counter(r['priority'] for r in items)
        se = len(set(r['session'] for r in items))
        print('%-9s %4d  %-18s %-14s %d sessions' % (
            bname, len(items),
            '/'.join('%s=%d' % kv for kv in sp.items()),
            '/'.join('%s=%d' % kv for kv in pr.items()),
            se))
    print('=' * 72)
    print('priority totals:', dict(Counter(r['priority'] for r in rows)))
    print('wrote label_batches/*.txt and label_plan.csv')


if __name__ == '__main__':
    main()
