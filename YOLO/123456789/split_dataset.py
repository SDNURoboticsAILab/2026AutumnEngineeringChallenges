# -*- coding: utf-8 -*-
"""按类别分层划分 train / val，并复制到 YOLO 目录结构。

用法（在 yolo_project 目录下，用 venv 里的 python 运行）：
    python split_dataset.py              # 首次划分
    python split_dataset.py --force      # 已有数据时强制重新划分
    python split_dataset.py "<原始图片根目录>"

原始图片根目录下应有 obstacle / cola / football 三个子文件夹。
默认指向仓库里的 YOLO 数据目录，可自行覆盖。

安全保护：dataset/images 下已有文件时默认拒绝执行（exit code 1）。
重复划分会改变 train/val 的成员名单，使已标注的标签与图片错位，
因此只有在确实要重新划分时才加 --force。
"""
import random
import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DEFAULT_SRC = PROJ.parent / '2026AutumnEngineeringChallenges-main' / \
    '2026AutumnEngineeringChallenges-main' / 'YOLO'

_POSITIONAL = [a for a in sys.argv[1:] if not a.startswith('--')]
FORCE = '--force' in sys.argv[1:]
SRC_ROOT = Path(_POSITIONAL[0]).resolve() if _POSITIONAL else DEFAULT_SRC
DST = PROJ / 'dataset'
CLASSES = ['obstacle', 'cola', 'football']
VAL_RATIO = 0.2
SEED = 42
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def main():
    if not SRC_ROOT.exists():
        raise SystemExit('原始图片目录不存在: %s' % SRC_ROOT)

    # 安全保护：已有划分结果时默认不动数据
    img_root = DST / 'images'
    existing = sum(1 for p in img_root.rglob('*') if p.is_file()) if img_root.exists() else 0
    if existing and not FORCE:
        print('!! dataset/images 下已有 %d 个文件，本次不执行。' % existing)
        print('   重复划分会改变 train/val 成员，使已标注的标签与图片错位。')
        print('   确实要重新划分：先清空 dataset\\images 与 dataset\\labels，或加 --force 参数；')
        print('   加 --force 后请把 labels 里的标签按图片实际所在的 split 重新归位。')
        return 1

    print('原始图片根目录:', SRC_ROOT)
    print('输出目录      :', DST)
    if existing:
        print('!! --force：已有数据将被本次划分结果覆盖 / 追加')

    random.seed(SEED)
    total = {'train': 0, 'val': 0}

    for name in CLASSES:
        src_dir = SRC_ROOT / name
        if not src_dir.exists():
            raise SystemExit('缺少类别目录: %s' % src_dir)

        imgs = sorted(p for p in src_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in EXTS)
        random.shuffle(imgs)

        n_val = int(len(imgs) * VAL_RATIO)
        if n_val == 0 and imgs:
            n_val = 1

        groups = (('val', imgs[:n_val]), ('train', imgs[n_val:]))
        for split, files in groups:
            out_dir = DST / 'images' / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)
            total[split] += len(files)
            print('%-9s %-5s %4d 张' % (name, split, len(files)))

    # 标签目录先建好，标注完把 txt 放进去
    for split in ('train', 'val'):
        (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print('-' * 32)
    print('合计: train=%d  val=%d  总计=%d' %
          (total['train'], total['val'], sum(total.values())))
    print('下一步: 标注 dataset\\images\\train 与 val，标签 txt 放到 dataset\\labels\\<split>\\')
    print('提示  : 用 verify_structure.py 核对结构，用 show_structure.py 导出目录树用于截图')
    return 0


if __name__ == '__main__':
    sys.exit(main())
