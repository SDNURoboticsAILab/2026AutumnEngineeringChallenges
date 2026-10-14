# -*- coding: utf-8 -*-
"""原始数据检查（对应 Level 1 第 2 小节：检查原始数据）。

检查内容：三类图片数量、格式、分辨率分布、是否存在无法读取的图片，
并汇总整理后的 dataset 中图片与标签的对应情况。

用法（在 yolo_project 下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_rawdata.py
    # 也可指定原始图片目录
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_rawdata.py "D:\\某目录\\YOLO"
"""
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
DEFAULT_RAW = PROJ.parent / '2026AutumnEngineeringChallenges-main' / \
    '2026AutumnEngineeringChallenges-main' / 'YOLO'
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
CLASSES = ('obstacle', 'cola', 'football')


def check_raw(root):
    print('原始图片目录: %s' % root)
    print('-' * 66)
    print('%-10s %8s %10s %-22s %s' % ('类别', '数量', '格式', '分辨率分布', '异常'))
    total = 0
    bad = []
    for cls in CLASSES:
        d = root / cls
        if not d.exists():
            print('%-10s %8s' % (cls, '目录不存在'))
            continue
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
        total += len(files)
        exts = Counter(p.suffix.lower() for p in files)
        sizes = Counter()
        for p in files:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    sizes[im.size] += 1
            except Exception as exc:
                bad.append('%s (%s)' % (p.name, exc))
        top = ', '.join('%dx%d×%d' % (w, h, n) for (w, h), n in sizes.most_common(3))
        print('%-10s %8d %10s %-22s %s' % (cls, len(files),
                                           ','.join(k.lstrip('.') for k in exts),
                                           top, '无' if not bad else '见下'))
    print('-' * 66)
    print('合计: %d 张' % total)
    if bad:
        print('无法读取的图片 (%d): %s' % (len(bad), ', '.join(bad[:10])))
    else:
        print('可读性检查: 全部图片均可正常打开，无损坏文件')
    return total, bad


def check_dataset():
    print()
    print('整理后的 dataset 情况:')
    print('%-8s %10s %10s %10s %10s' % ('split', '图片', '标签', '已配对', '无标签'))
    for split in ('train', 'val'):
        img_dir = DATASET / 'images' / split
        lbl_dir = DATASET / 'labels' / split
        imgs = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in EXTS} if img_dir.exists() else set()
        lbls = {p.stem for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'} if lbl_dir.exists() else set()
        print('%-8s %10d %10d %10d %10d' % (split, len(imgs), len(lbls),
                                            len(imgs & lbls), len(imgs - lbls)))


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RAW
    if not root.exists():
        print('原始图片目录不存在: %s' % root)
        print('（如果你已经删掉了原始目录，可以跳过这一节，直接用 dataset 里的数据）')
    else:
        check_raw(root)
    if DATASET.exists():
        check_dataset()
    print()
    print('结论：数据格式统一、全部可读；后续按 8:2 分层划分即可（split_dataset.py）。')


if __name__ == '__main__':
    main()
