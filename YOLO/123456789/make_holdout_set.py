# -*- coding: utf-8 -*-
"""为 Level 4 准备“训练集之外”的测试图片（无数据泄漏的严谨做法）。

做法：从当前数据集中挑出若干张（每个类别 2 张、优先选目标较多的），
      **移动**（不是复制）到 test_images/，使它们彻底退出训练/验证集；
      然后重新训练一次模型，再用新模型对这些图片推理 —— 这样才是真正的
      “模型没见过的图片”。

原图与标签会备份到 dataset/_holdout_backup_<时间戳>/，随时可还原。

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe make_holdout_set.py
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe make_holdout_set.py --per-class 3
"""
import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
IMAGES = DATASET / 'images'
LABELS = DATASET / 'labels'
TEST = PROJ / 'test_images'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def boxes_of(split, stem):
    lbl = LABELS / split / (stem + '.txt')
    if not lbl.exists():
        return []
    out = []
    for line in lbl.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append(int(float(parts[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=2, help='每个类别挑几张（默认 2）')
    args = ap.parse_args()

    TEST.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = DATASET / ('_holdout_backup_' + stamp)
    (backup / 'images').mkdir(parents=True, exist_ok=True)
    (backup / 'labels').mkdir(parents=True, exist_ok=True)

    picked = []          # (split, stem, classes)
    used = set()         # 已选过的图片，避免同一张图被多个类别重复选中
    for cid, cname in NAMES.items():
        # 候选：标签里含该类、且目标数 >= 2 的优先（便于展示多目标场景）
        cands = []
        for split in ('val', 'train'):
            d = IMAGES / split
            if not d.exists():
                continue
            for img in sorted(d.iterdir()):
                if img.suffix.lower() not in EXTS:
                    continue
                if img.stem in used:
                    continue
                cls = boxes_of(split, img.stem)
                if cid in cls:
                    cands.append((len(cls), split, img.stem))
        cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        chosen = cands[:args.per_class]
        for _, split, stem in chosen:
            used.add(stem)
            picked.append((split, stem, cname))

    print('将从数据集中移出以下图片（改作 Level 4 测试图）：')
    manifest = ['# Level 4 测试图片清单（这些图片已从训练/验证集中移出）',
                '# 原始位置 -> test_images/', '']
    for split, stem, cname in picked:
        img_src = IMAGES / split / (stem + '.jpg')
        lbl_src = LABELS / split / (stem + '.txt')
        if not img_src.exists():
            continue
        shutil.copy2(img_src, backup / 'images' / (split + '__' + img_src.name))
        if lbl_src.exists():
            shutil.copy2(lbl_src, backup / 'labels' / (split + '__' + lbl_src.name))
        shutil.move(str(img_src), str(TEST / img_src.name))
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(TEST / (stem + '.txt')))
        cls = boxes_of('train', stem) if False else []          # 标签已移走，下面用备份里的信息
        print('  [%s] %-40s 类别=%s' % (split, stem, cname))
        manifest.append('%-5s %-40s %s' % (split, stem, cname))

    (TEST / '_manifest.txt').write_text('\n'.join(manifest) + '\n', encoding='utf-8')
    print('\n已移动到: %s' % TEST)
    print('备份（可还原）: %s' % backup)
    print('清单: %s' % (TEST / '_manifest.txt'))
    print('\n下一步：重新训练一次（让模型彻底没见过这些图），例如：')
    print('  C:\\Users\\dell\\yolo-env\\Scripts\\python.exe train.py --epochs 60 --name train_holdout')
    return 0


if __name__ == '__main__':
    sys.exit(main())
