# -*- coding: utf-8 -*-
"""训练脚本：obstacle / cola / football 三分类目标检测。

首轮推荐参数（RTX 3050 Laptop 4GB，数据集 949 张 640x480）：
    model  yolo11n.pt      预训练权重，小数据集+小显存的首选
    epochs 100             patience=25 早停，通常 60~80 轮就收敛
    imgsz  640             原图就是 640x480，不用再放大
    batch  8               显存不够就改 4；不要盲目调大
    lr0    0.01（optimizer='auto' 时框架会自动选 AdamW 与合适学习率）
    cache  True            949 张小图全部进内存，训练速度翻倍

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe train.py
    ... train.py --epochs 20            # 冒烟测试
    ... train.py --subset               # 只用在已标注的图片上跑通流程
    ... train.py --force                # 忽略类别缺失的预检警告
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent

MODEL = 'yolo11n.pt'
EPOCHS = 100
IMGSZ = 640
BATCH = 8
LR0 = 0.01
OPTIMIZER = 'auto'      # auto: 小数据集自动选 AdamW，稳；想用 SGD 就写 'SGD'
PATIENCE = 25
WORKERS = 4             # Windows 上若卡在 dataloader，改成 0
CACHE = 'disk'          # 用磁盘缓存：RAM 缓存会把图像数组带进 DataLoader 子进程，Windows 上会 pickle 失败
DEVICE = 0              # 没有 GPU 写 'cpu'
SEED = 42
AMP = False             # 关掉 AMP：AMP 自检会去 GitHub 下载 yolo26n.pt，本机网络连不上会卡住
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}


def count_instances():
    """统计 dataset/labels 下每个类别的实例数，用于训练前预检。"""
    counter = Counter()
    for split in ('train', 'val'):
        lbl_dir = PROJ / 'dataset' / 'labels' / split
        if not lbl_dir.exists():
            continue
        for path in lbl_dir.glob('*.txt'):
            for line in path.read_text(encoding='utf-8').splitlines():
                parts = line.split()
                if len(parts) == 5:
                    counter[int(float(parts[0]))] += 1
    return counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--imgsz', type=int, default=IMGSZ)
    ap.add_argument('--lr0', type=float, default=LR0)
    ap.add_argument('--device', default=str(DEVICE))
    ap.add_argument('--subset', action='store_true', help='只用已标注图片（data_subset.yaml）')
    ap.add_argument('--name', default='train')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--force', action='store_true', help='跳过类别预检')
    args = ap.parse_args()

    data_yaml = PROJ / ('data_subset.yaml' if args.subset else 'data.yaml')
    if not data_yaml.exists():
        raise SystemExit('找不到数据集配置: %s' % data_yaml)

    # ---------- 训练前预检：缺类别就别白跑一轮 ----------
    counts = count_instances()
    print('实例统计: ' + ', '.join('%s=%d' % (NAMES.get(k, k), v)
                                   for k, v in sorted(counts.items())))
    missing = [NAMES[k] for k in NAMES if counts[k] == 0]
    if missing and not args.force:
        no_label = sum(1 for split in ('train', 'val')
                       for p in (PROJ / 'dataset' / 'images' / split).glob('*')
                       if not (PROJ / 'dataset' / 'labels' / split / (p.stem + '.txt')).exists())
        raise SystemExit(
            '\n[预检未通过] 缺类别的实例: %s\n'
            '  没有任何标签的图片: %d 张（Ultralytics 会把它们当作“背景图”参与训练，\n'
            '  比例过高时模型会被教成“这些图里什么都没有”，缺失的类别永远学不到）\n'
            '  正确做法：先补齐标注，再训练；只想验证流程可加 --subset，\n'
            '  确实要强行训练加 --force。\n' % (', '.join(missing), no_label))

    device = int(args.device) if args.device.isdigit() else args.device
    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        optimizer=OPTIMIZER,
        patience=PATIENCE,
        workers=WORKERS,
        cache=CACHE,
        device=device,
        seed=SEED,
        deterministic=True,
        amp=AMP,
        close_mosaic=10,
        plots=True,
        project=str(PROJ / 'runs' / 'detect'),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
    )

    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=device)
    print('=' * 56)
    print('mAP50    : %.4f' % float(metrics.box.map50))
    print('mAP50-95 : %.4f' % float(metrics.box.map))
    print('precision: %.4f' % float(metrics.box.mp))
    print('recall   : %.4f' % float(metrics.box.mr))
    for i, name in sorted(NAMES.items()):
        try:
            print('  %-9s mAP50=%.4f  mAP50-95=%.4f' % (name, metrics.box.ap50[i], metrics.box.ap[i]))
        except Exception:
            pass
    print('权重: %s' % (PROJ / 'runs' / 'detect' / args.name / 'weights' / 'best.pt'))
    print('曲线: %s' % (PROJ / 'runs' / 'detect' / args.name / 'results.png'))


if __name__ == '__main__':
    main()
