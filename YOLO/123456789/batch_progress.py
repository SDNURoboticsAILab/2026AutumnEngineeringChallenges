# -*- coding: utf-8 -*-
"""标注进度跟踪：按 label_plan.csv 的分批统计 obstacle 完成情况。

用法（在 yolo_project 下运行）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe batch_progress.py
"""
from collections import OrderedDict
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
PLAN = PROJ / 'label_plan.csv'


def has_class0(split, stem):
    lbl = DATASET / 'labels' / split / (stem + '.txt')
    if not lbl.exists():
        return False
    for line in lbl.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) == 5 and int(float(parts[0])) == 0:
            return True
    return False


def main():
    if not PLAN.exists():
        raise SystemExit('缺少 label_plan.csv，先运行 make_label_plan.py')

    batches = OrderedDict()
    with PLAN.open(encoding='utf-8-sig', newline='') as fh:
        next(fh)
        for line in fh:
            batch, prio, split, image, session, draft = line.rstrip('\n').split(',')
            batches.setdefault(batch, {'total': 0, 'done': 0, 'prio': prio,
                                       'split': split, 'pending': []})
            item = batches[batch]
            item['total'] += 1
            if has_class0(split, Path(image).stem):
                item['done'] += 1
            elif len(item['pending']) < 2:
                item['pending'].append(image)

    print('%-9s %6s %6s %8s   %s' % ('batch', 'total', 'done', 'progress', 'next up'))
    print('-' * 78)
    tot = done = 0
    for name, item in batches.items():
        tot += item['total']
        done += item['done']
        pct = 100.0 * item['done'] / item['total'] if item['total'] else 0
        print('%-9s %6d %6d %7.1f%%   %s' % (name, item['total'], item['done'], pct,
                                             ', '.join(item['pending'])))
    print('-' * 78)
    print('TOTAL     %6d %6d %7.1f%%' % (tot, done, 100.0 * done / tot if tot else 0))
    if done == tot:
        print('obstacle 标注全部完成，可以跑 check_labels.py + audit_labels.py 后开始训练。')


if __name__ == '__main__':
    main()
