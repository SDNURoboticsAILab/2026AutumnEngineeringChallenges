# -*- coding: utf-8 -*-
"""预标注助手：用 COCO 预训练模型先把 cola / football 框出来，减少手工工作量。

用法（在 yolo_project 下，激活 venv 后运行）：
    python auto_label.py            # 对 dataset/images/train 和 val 都跑一遍
    python auto_label.py --conf 0.4 # 提高置信度阈值，少一些误框
    python auto_label.py --overwrite

说明：
  * COCO 里没有 obstacle 这个类，障碍物必须自己用 LabelImg 画。
  * 映射关系：bottle -> 1 (cola)，sports ball -> 2 (football)。
  * 生成的 txt 是【预标注草稿】，必须逐张在 LabelImg 里复核：删掉误框、补上漏框、
    补画 obstacle，否则模型学不到 obstacle 这一类，Level 4 会翻车。
  * 已经存在的 txt 默认不覆盖，方便你在草稿上继续改。
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
SPLITS = ('train', 'val')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# COCO 类别名 -> 本项目类别编号
CLASS_MAP = {
    'bottle': 1,        # -> cola
    'sports ball': 2,   # -> football
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--device', default='auto')  # auto / cpu / 0
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--splits', default='train,val')
    args = ap.parse_args()

    model = YOLO('yolo11n.pt')  # 通用预训练权重，首次运行自动下载

    import torch
    if args.device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    elif args.device.isdigit():
        device = int(args.device)
    else:
        device = args.device
    print('device:', device)
    stats = Counter()
    per_class = Counter()

    for split in args.splits.split(','):
        split = split.strip()
        img_dir = DATASET / 'images' / split
        lbl_dir = DATASET / 'labels' / split
        if not img_dir.exists():
            print('skip missing dir:', img_dir)
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(p for p in img_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in EXTS)
        print('--- %s : %d images ---' % (split, len(imgs)))

        pending = [p for p in imgs if args.overwrite or not (lbl_dir / (p.stem + '.txt')).exists()]
        print('    to pre-label:', len(pending))
        if not pending:
            continue

        results = model.predict(source=[str(p) for p in pending], imgsz=args.imgsz,
                                conf=args.conf, batch=args.batch, device=device,
                                verbose=False)

        for r in results:
            path = Path(r.path)
            lines = []
            for box in r.boxes:
                coco_name = model.names[int(box.cls[0])]
                if coco_name not in CLASS_MAP:
                    continue
                cls_id = CLASS_MAP[coco_name]
                x, y, w, h = box.xywhn[0].tolist()  # 归一化中心点宽高，正是 YOLO 需要的格式
                lines.append('%d %.6f %.6f %.6f %.6f' % (cls_id, x, y, w, h))
                per_class[cls_id] += 1

            if lines:
                (lbl_dir / (path.stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
                stats['written'] += 1
            else:
                stats['need_manual'] += 1  # 大概率是 obstacle 图，需要手画

    print('=' * 48)
    print('pre-labeled files :', stats['written'])
    print('still need manual :', stats['need_manual'])
    print('class 1 (cola)    :', per_class[1])
    print('class 2 (football):', per_class[2])
    print('提醒: 打开 LabelImg 逐张复核，并补画 class 0 (obstacle)。')
    print('复核完用 python check_labels.py 自检。')


if __name__ == '__main__':
    main()
