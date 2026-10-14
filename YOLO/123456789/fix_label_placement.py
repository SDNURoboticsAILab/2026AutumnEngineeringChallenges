# -*- coding: utf-8 -*-
"""修复“图片与标签不在同一个 split 目录”的错位问题。

背景：dataset/images/<split> 与 dataset/labels/<split> 必须一一对应，
本次检查发现两边出现错位（部分 val 的标签跑到了 labels/train，反之亦然，
还有图片本身换了目录），这会导致训练时把大量图片当成“没有标签的背景图”。

用法（在 yolo_project 下）：
    # 只体检，不改动
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe fix_label_placement.py

    # 真正修复（会备份到 dataset/_backup_<时间戳>/，并输出改动日志）
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe fix_label_placement.py --apply
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
SPLITS = ('train', 'val')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def index_images():
    idx = {}
    for split in SPLITS:
        d = DATASET / 'images' / split
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in EXTS:
                    idx.setdefault(p.stem, split)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='执行移动（默认只检查）')
    args = ap.parse_args()

    idx = index_images()
    print('图片索引: %d 张' % len(idx))
    print('=' * 66)

    moves = []       # (src, dst)
    orphans = []     # 找不到对应图片的标签
    for split in SPLITS:
        lbl_dir = DATASET / 'labels' / split
        if not lbl_dir.exists():
            continue
        for txt in sorted(p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'):
            want = idx.get(txt.stem)
            if want is None:
                orphans.append(txt)
            elif want != split:
                moves.append((txt, DATASET / 'labels' / want / txt.name))

    for split in SPLITS:
        lbl_dir = DATASET / 'labels' / split
        imgs = {p.stem for p in (DATASET / 'images' / split).iterdir()
                if p.suffix.lower() in EXTS} if (DATASET / 'images' / split).exists() else set()
        lbls = {p.stem for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'} if lbl_dir.exists() else set()
        print('[%s] 图片=%d 标签=%d 已配对=%d 无标签=%d 错位标签=%d'
              % (split, len(imgs), len(lbls), len(imgs & lbls), len(imgs - lbls),
                 len([1 for t, _ in moves if t.parent.name == split])))
    print()
    print('需要移动的标签: %d 个' % len(moves))
    for src, dst in moves[:8]:
        print('   %s/%s  ->  labels/%s/' % ('labels', src.parent.name + '/' + src.name, dst.parent.name))
    if len(moves) > 8:
        print('   ... 其余 %d 个' % (len(moves) - 8))
    if orphans:
        print('找不到对应图片的标签（需要人工判断）: %d 个' % len(orphans))
        for t in orphans[:5]:
            print('   %s' % t.name)

    if not moves and not orphans:
        print('结果: 图片与标签完全对应，无需修复')
        return 0

    if not args.apply:
        print()
        print('（当前是检查模式，未做任何改动。确认无误后加 --apply 执行修复）')
        return 1

    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = DATASET / ('_backup_' + stamp)
    backup.mkdir(parents=True, exist_ok=True)
    log = []
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup / (src.parent.name + '__' + src.name))
        if dst.exists():
            shutil.copy2(dst, backup / (dst.parent.name + '__' + dst.name + '.pre'))
        src.replace(dst)
        log.append('%s/%s -> labels/%s/%s' % ('labels', src.parent.name, dst.parent.name, dst.name))
    (backup / 'moved.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
    print()
    print('已移动 %d 个标签文件' % len(moves))
    print('备份目录: %s' % backup)
    print('改动日志: %s' % (backup / 'moved.log'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
