# -*- coding: utf-8 -*-
"""标注归位助手：把散落在别处的 YOLO txt 正确搬进 dataset/labels/{train,val}，并验证归位是否完整。

用法（在 yolo_project 下运行）：

    # 1) 只检查：告诉我这批 txt 分别该进 train 还是 val，有没有对不上图片的
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_transfer.py "D:\\某个目录"

    # 2) 直接搬（会先备份同名文件）
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_transfer.py "D:\\某个目录" --apply

    # 3) 不带参数：只体检当前 dataset 的归位状态（图片与标签是否一一对应、有没有残留 txt）
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_transfer.py

判定规则：txt 的文件名（去掉扩展名）必须与图片文件名（去掉扩展名）完全一致，
图片在 dataset/images/train 就放进 dataset/labels/train，在 val 就放进 val。
"""
import argparse
import shutil
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
SPLITS = ('train', 'val')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def index_images():
    """stem -> split 的索引"""
    idx = {}
    for split in SPLITS:
        img_dir = DATASET / 'images' / split
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() in EXTS:
                idx.setdefault(p.stem, split)
    return idx


def audit_dataset():
    print('=== dataset 归位体检 ===')
    idx = index_images()
    print('images 索引: %d 张图' % len(idx))
    label_with_class0 = 0
    for split in SPLITS:
        img_dir = DATASET / 'images' / split
        lbl_dir = DATASET / 'labels' / split
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in EXTS] if img_dir.exists() else []
        lbls = [p for p in lbl_dir.glob('*.txt') if p.name != 'classes.txt'] if lbl_dir.exists() else []
        stray = list(img_dir.glob('*.txt')) if img_dir.exists() else []
        has_c0 = 0
        for lbl in lbls:
            for line in lbl.read_text(encoding='utf-8', errors='ignore').splitlines():
                parts = line.split()
                if len(parts) == 5 and int(float(parts[0])) == 0:
                    has_c0 += 1
                    break
        label_with_class0 += has_c0
        print('[%s] 图片=%d  标签=%d  含obstacle框的标签=%d  图片目录里的残留txt=%d'
              % (split, len(imgs), len(lbls), has_c0, len(stray)))
        miss = [p.stem for p in imgs if not (lbl_dir / (p.stem + '.txt')).exists()]
        print('       还没有标签的图: %d 张' % len(miss))
    print('全部含 obstacle 框的标签文件: %d 个' % label_with_class0)
    return label_with_class0


def process_source(src, apply):
    src = Path(src)
    if not src.exists():
        raise SystemExit('来源目录不存在: %s' % src)

    idx = index_images()
    txts = sorted(src.rglob('*.txt'))
    print('来源目录: %s' % src)
    print('找到 txt: %d 个' % len(txts))
    if not txts:
        return

    groups = {'train': [], 'val': [], 'unmatched': []}
    for t in txts:
        split = idx.get(t.stem)
        groups[split if split else 'unmatched'].append(t)

    for split in SPLITS:
        print('  -> 应归入 %-5s : %d 个' % (split, len(groups[split])))
    print('  -> 找不到对应图片 : %d 个' % len(groups['unmatched']))
    for t in groups['unmatched'][:8]:
        print('       !! %s' % t.name)

    if not apply:
        print('\n（只检查，没有改动。确认无误后加 --apply 执行搬运）')
        return

    for split in SPLITS:
        dst = DATASET / 'labels' / split
        dst.mkdir(parents=True, exist_ok=True)
        for t in groups[split]:
            target = dst / t.name
            if target.exists():
                shutil.copy2(target, target.with_suffix('.txt.bak'))
            shutil.copy2(t, target)
        print('已复制 %d 个 txt 到 %s' % (len(groups[split]), dst))
    print('搬运完成。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', nargs='?', default='', help='标注 txt 所在目录')
    ap.add_argument('--apply', action='store_true', help='真正执行复制')
    args = ap.parse_args()

    if args.source:
        process_source(args.source, args.apply)
        print()
    audit_dataset()
    print()
    print('下一步验证（两条都要看）：')
    print('  C:\\Users\\dell\\yolo-env\\Scripts\\python.exe check_labels.py')
    print('  C:\\Users\\dell\\yolo-env\\Scripts\\python.exe batch_progress.py')


if __name__ == '__main__':
    main()
