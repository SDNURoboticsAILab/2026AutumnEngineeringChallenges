# -*- coding: utf-8 -*-
"""用你已有的标注训练一个“自举模型”，再让它给其余未标注图片打上 football / cola / obstacle 框。

流程：
  1) 备份现有标签到 dataset/_labels_backup_<时间戳>/（可随时整体还原）
  2) 用已标注的图片（labeled_train.txt / labeled_val.txt）训练 yolo11n（GPU）
  3) 用这个模型对 todo_no_label.txt 里所有未标注图片做推理，把检测结果写成 YOLO txt
  4) 统计并打印结果

注意：这些是**机器预标注**，需要人工复核后才算正式标注；报告里应如实写成“模型预标注 + 人工复核”。

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe auto_annotate_rest.py
"""
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
LABELS = DATASET / 'labels'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}

EPOCHS = 40
IMGSZ = 640
BATCH = 8
DEVICE = 0
CONF = 0.25
AMP = False   # 关掉 AMP：AMP 自检会去 GitHub 下载 yolo26n.pt，本机网络连不上会卡住


def backup_labels():
    stamp = time.strftime('%Y%m%d_%H%M%S')
    dst = DATASET / ('_labels_backup_' + stamp)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for split in ('train', 'val'):
        src = LABELS / split
        if src.exists():
            target = dst / split
            target.mkdir(exist_ok=True)
            for f in src.glob('*.txt'):
                shutil.copy2(f, target / f.name)
                n += 1
    print('已备份 %d 个标签文件 -> %s' % (n, dst), flush=True)
    return dst


def train_bootstrap():
    from ultralytics import YOLO
    print('\n[训练] yolo11n，数据=data_subset.yaml（仅用已标注图片），epochs=%d imgsz=%d batch=%d device=%s'
          % (EPOCHS, IMGSZ, BATCH, DEVICE), flush=True)
    model = YOLO('yolo11n.pt')
    model.train(
        data=str(PROJ / 'data_subset.yaml'),
        epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        workers=4, cache=True, patience=15, seed=42, amp=AMP,
        project=str(PROJ / 'runs' / 'detect'), name='bootstrap', exist_ok=True,
    )
    best = PROJ / 'runs' / 'detect' / 'bootstrap' / 'weights' / 'best.pt'
    print('训练完成，权重: %s (存在=%s)' % (best, best.exists()), flush=True)
    return best


def load_todo():
    todo = PROJ / 'todo_no_label.txt'
    items = []
    for line in todo.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            split, name = line.split('/', 1)
            items.append((split, name))
    return items


def annotate(best):
    from ultralytics import YOLO
    items = load_todo()
    print('\n[推理] 待标注图片 %d 张，conf=%.2f' % (len(items), CONF), flush=True)
    model = YOLO(str(best))
    written = Counter()
    no_det = 0
    per_class = Counter()
    confs = []

    # 分批推理，避免一次性占满显存
    CHUNK = 50
    for i in range(0, len(items), CHUNK):
        chunk = items[i:i + CHUNK]
        paths = [str(DATASET / 'images' / s / n) for s, n in chunk]
        results = model.predict(source=paths, imgsz=IMGSZ, conf=CONF, batch=BATCH,
                                device=DEVICE, verbose=False, save=False)
        for (split, name), r in zip(chunk, results):
            lines = []
            for box in r.boxes:
                cid = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                lines.append('%d %.6f %.6f %.6f %.6f' % (cid, x, y, w, h))
                per_class[cid] += 1
                confs.append(float(box.conf[0]))
            if lines:
                out = LABELS / split / (Path(name).stem + '.txt')
                out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                written[split] += 1
            else:
                no_det += 1
        print('  进度 %d/%d，已写入 %d 张' % (min(i + CHUNK, len(items)), len(items),
                                              sum(written.values())), flush=True)
    return written, no_det, per_class


def main():
    print('=== 步骤 1：备份现有标签 ===', flush=True)
    bak = backup_labels()

    print('\n=== 步骤 2：训练自举模型 ===', flush=True)
    best = train_bootstrap()
    if not best.exists():
        print('训练没有产出权重，终止。')
        return 2

    print('\n=== 步骤 3：对其余未标注图片做预标注 ===', flush=True)
    written, no_det, per_class = annotate(best)

    print('\n=== 结果统计 ===', flush=True)
    print('写入标签的图片: train %d 张, val %d 张' % (written['train'], written['val']), flush=True)
    print('模型没检测到目标（仍无标签）: %d 张' % no_det, flush=True)
    print('生成的框: ' + ', '.join('%s=%d' % (NAMES.get(k, k), v) for k, v in sorted(per_class.items())), flush=True)
    print('标签备份（可整体还原）: %s' % bak, flush=True)
    print('\n提示：这是机器预标注，请在 labelImg 里逐张复核后再用于正式训练。', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
