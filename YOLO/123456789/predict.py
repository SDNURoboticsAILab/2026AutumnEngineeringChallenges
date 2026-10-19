# -*- coding: utf-8 -*-
"""Level 4 推理脚本：用自己训练出来的 best.pt 检测新图片。

在 yolo_project 目录下运行：
    python predict.py                       # 默认检测 test_images/ 下所有图片
    python predict.py path\\to\\new.jpg      # 也可以指定单张图片或目录

结果图片保存在 results/predict/ ，终端会打印每个目标的名字与置信度。
"""
import sys
from pathlib import Path

from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJ / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
DEFAULT_SRC = PROJ / 'test_images'
CONF = 0.25
IMGSZ = 640


def parse_args():
    """支持：python predict.py [待检测目录或图片] [--weights 权重路径] [--conf 0.25] [--name predict]"""
    src = None
    weights = DEFAULT_WEIGHTS
    conf = CONF
    name = 'predict'
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--weights' and i + 1 < len(argv):
            weights = Path(argv[i + 1]).resolve()
            i += 2
        elif a == '--conf' and i + 1 < len(argv):
            conf = float(argv[i + 1])
            i += 2
        elif a == '--name' and i + 1 < len(argv):
            name = argv[i + 1]
            i += 2
        else:
            src = Path(a).resolve()
            i += 1
    return (src or DEFAULT_SRC), weights, conf, name


def main():
    src, weights, conf, run_name = parse_args()
    if not weights.exists():
        raise SystemExit('找不到自己训练的权重: %s\n请先运行 python train.py' % weights)
    if not src.exists():
        raise SystemExit('待检测路径不存在: %s' % src)

    print('加载权重: %s' % weights)
    model = YOLO(str(weights))
    results = model.predict(
        source=str(src),
        imgsz=IMGSZ,
        conf=conf,
        save=True,
        project=str(PROJ / 'results'),
        name=run_name,
        exist_ok=True,
    )

    print('-' * 60)
    for r in results:
        print(Path(r.path).name)
        if len(r.boxes) == 0:
            print('    （未检出目标，可把 --conf 降到 0.15 再试）')
        for box in r.boxes:
            name = model.names[int(box.cls[0])]
            c = float(box.conf[0])
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            print('    %-10s conf=%.3f  box=%s' % (name, c, xyxy))
    print('结果图片目录:', PROJ / 'results' / run_name)


if __name__ == '__main__':
    main()
