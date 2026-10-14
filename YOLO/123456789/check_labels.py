# -*- coding: utf-8 -*-
"""数据集自检：确认图片与标签一一对应、标签格式合法、三个类别都有实例。

用法（在 yolo_project 下运行）：
    python check_labels.py

输出示例：
    [train] images=759 labels=759 empty=0 invalid=0
           class 0 obstacle : 316
           class 1 cola     : 290
           class 2 football : 340
    PASS: 数据集结构正常，可以训练

任何一项异常都会给出具体文件名，照着改就行。
"""
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
SPLITS = ('train', 'val')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}


def check_split(split):
    img_dir = DATASET / 'images' / split
    lbl_dir = DATASET / 'labels' / split
    problems = []
    counts = Counter()

    if not img_dir.exists():
        problems.append('缺少图片目录: %s' % img_dir)
        return counts, problems

    imgs = {p.stem: p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS}
    lbls = {p.stem: p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'} if lbl_dir.exists() else {}

    missing = sorted(set(imgs) - set(lbls))
    extra = sorted(set(lbls) - set(imgs))
    if missing:
        problems.append('没有标签的图片(%d): %s' % (len(missing), ', '.join(missing[:5])))
    if extra:
        problems.append('多余的标签(%d): %s' % (len(extra), ', '.join(extra[:5])))

    empty = 0
    invalid = []
    for stem, path in sorted(lbls.items()):
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            empty += 1
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                invalid.append('%s:%d 字段数不是5' % (path.name, lineno))
                continue
            try:
                cls_id = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                invalid.append('%s:%d 不是数字' % (path.name, lineno))
                continue
            if cls_id not in NAMES:
                invalid.append('%s:%d 类别编号 %s 不在 0/1/2 内' % (path.name, lineno, cls_id))
            if any(v < 0 or v > 1 for v in vals):
                invalid.append('%s:%d 坐标超出 0~1' % (path.name, lineno))
            if vals[2] <= 0 or vals[3] <= 0:
                invalid.append('%s:%d 宽高为 0' % (path.name, lineno))
            counts[cls_id] += 1

    print('[%s] images=%d labels=%d empty=%d invalid=%d' %
          (split, len(imgs), len(lbls), empty, len(invalid)))
    for cid in sorted(NAMES):
        print('       class %d %-9s: %d' % (cid, NAMES[cid], counts[cid]))
    for p in problems:
        print('       !! ' + p)
    for p in invalid[:10]:
        print('       !! ' + p)
    return counts, problems + invalid


def main():
    print('数据集自检  ', DATASET)
    print('=' * 48)
    all_counts = Counter()
    all_problems = []
    for split in SPLITS:
        counts, problems = check_split(split)
        all_counts += counts
        all_problems += problems

    print('=' * 48)
    print('合计实例数: ', dict((NAMES[k], v) for k, v in sorted(all_counts.items())))
    no_inst = [NAMES[k] for k in NAMES if all_counts[k] == 0]
    if no_inst:
        print('警告: 以下类别一个框都没有 -> ' + ', '.join(no_inst))
    if all_problems:
        print('FAIL: 共 %d 处问题，见上面的 !! 行' % len(all_problems))
    elif no_inst:
        print('WARN: 结构没问题，但缺类别，先补齐再训练')
    else:
        print('PASS: 数据集结构正常，可以跑 python train.py')


if __name__ == '__main__':
    main()
