# -*- coding: utf-8 -*-
"""Level 4 第 1 步：加载自己训练得到的模型权重，并给出可验证的证据。

对应考核要求：
    Level 4 → 实践步骤 → 1. 加载训练模型：加载自己训练得到的模型权重。

为什么要写成一个脚本，而不是在终端里敲两行：
    “我加载了模型”这句话本身是无法验证的。本脚本把 5 条互相独立的证据
    一起打印出来并逐条判定 PASS/FAIL：
      [1] 权重文件指纹（路径 / 大小 / 时间 / SHA256）
      [2] 加载结果（任务 / 类别数 / 类别名 / 参数量）
      [3] checkpoint 内部元数据 —— 权重文件自己记录了“是用哪个 data.yaml、
          训了多少轮、什么时候保存的、那一轮的指标是多少”
      [4] 与 results.csv 交叉验证 —— checkpoint 里的指标必须能在训练日志里找到
          完全对应的一轮，否则说明权重来路不明
      [5] 与官方 COCO 预训练权重 yolo11n.pt 对比 —— 80 类 vs 3 类

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe load_model.py
输出：results/evidence_level4/01_加载模型.txt
"""
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent
WEIGHTS = PROJ / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
RESULTS_CSV = PROJ / 'runs' / 'detect' / 'train' / 'results.csv'
PRETRAIN = PROJ / 'yolo11n.pt'
OUT = PROJ / 'results' / 'evidence_level4'
EXPECT_NAMES = ['obstacle', 'cola', 'football']

# checkpoint 里我们关心的训练参数
ARG_KEYS = ['data', 'model', 'epochs', 'imgsz', 'batch', 'optimizer', 'seed',
            'device', 'lr0', 'patience', 'project', 'name', 'pretrained']


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_ckpt(model, path):
    """优先用 YOLO 已经读进来的 ckpt，读不到就自己 torch.load。"""
    ckpt = getattr(model, 'ckpt', None)
    if isinstance(ckpt, dict) and 'train_args' in ckpt:
        return ckpt
    try:
        import torch
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except Exception as exc:                                   # pragma: no cover
        return {'_error': str(exc)}


FIELD_KEYS = ['metrics/precision(B)', 'metrics/recall(B)',
              'metrics/mAP50(B)', 'metrics/mAP50-95(B)']


def find_epoch_in_csv(match, tol=1e-4):
    """在 results.csv 里找出与 checkpoint 指标一致的那一轮。

    注意：不能只看 mAP50-95 —— 第 59 轮是 0.70944、第 82 轮是 0.70953，
    差值只有 9e-5，单看一个指标会把两轮混为一谈。
    因此这里要求 P / R / mAP50 / mAP50-95 四个指标同时落在容差内。
    """
    if not RESULTS_CSV.exists() or not match:
        return None, None
    try:
        target = {k: float(match[k]) for k in FIELD_KEYS}
    except (KeyError, TypeError, ValueError):
        return None, None
    rows = list(csv.DictReader(RESULTS_CSV.open(encoding='utf-8')))
    hit, best_diff = None, None
    for r in rows:
        r = {k.strip(): v for k, v in r.items()}
        try:
            diffs = [abs(float(r[k]) - target[k]) for k in FIELD_KEYS]
        except (KeyError, ValueError):
            continue
        worst = max(diffs)
        if best_diff is None or worst < best_diff:
            hit, best_diff = r, worst
    if hit is not None and best_diff is not None and best_diff < tol:
        return hit, best_diff
    return None, best_diff


def main():
    L = []
    checks = []          # (描述, 是否通过, 补充说明)

    def add(s=''):
        L.append(s)

    add('=== Level 4 第 1 步：加载训练模型 ===')
    add('')
    add('题目要求：加载自己训练得到的模型权重。')
    add('说明：本文件把“模型确实是自己训练出来的”这一点的证据逐条打印并判定。')
    add('')

    # ---------- [1] 权重文件信息 ----------
    add('[1] 权重文件信息')
    if not WEIGHTS.exists():
        add('  [FAIL] 找不到权重文件：%s' % WEIGHTS)
        add('         请先运行 train.py 完成训练。')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / '01_加载模型.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
        print('\n'.join(L))
        return 2

    st = WEIGHTS.stat()
    add('  路径      : %s' % WEIGHTS)
    add('  大小      : %d 字节（%.2f MB）' % (st.st_size, st.st_size / 1024 / 1024))
    add('  修改时间  : %s' % datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
    add('  SHA256    : %s' % sha256_of(WEIGHTS))
    add('  （SHA256 可用于确认提交的权重文件没有被替换或损坏）')
    add('')

    # ---------- [2] 加载结果 ----------
    add('[2] 加载结果（YOLO("%s")）' % WEIGHTS.name)
    model = YOLO(str(WEIGHTS))
    names = dict(sorted(model.names.items()))
    name_list = [names[i] for i in sorted(names)]
    n_par = sum(p.numel() for p in model.model.parameters())
    add('  任务 task       : %s' % model.task)
    add('  类别数 nc       : %d' % len(names))
    add('  类别名 names    : %s' % names)
    add('  参数量          : %d' % n_par)
    add('')

    checks.append(('加载成功且任务为 detect', model.task == 'detect', 'task=%s' % model.task))
    checks.append(('类别数为 3', len(names) == 3, 'nc=%d' % len(names)))
    checks.append(('类别名与项目一致', name_list == EXPECT_NAMES, '/'.join(name_list)))

    # ---------- [3] checkpoint 内部元数据 ----------
    add('[3] checkpoint 内部元数据（权重文件自己记录的训练履历）')
    ckpt = load_ckpt(model, WEIGHTS)
    if '_error' in ckpt:
        add('  [WARN] 读取 checkpoint 失败：%s' % ckpt['_error'])
    train_args = ckpt.get('train_args') or {}
    if train_args:
        add('  训练时使用的配置：')
        for k in ARG_KEYS:
            if k in train_args:
                add('    %-10s: %s' % (k, train_args[k]))
        add('  保存时间 date    : %s' % ckpt.get('date'))
        add('  ultralytics 版本 : %s' % ckpt.get('version'))
        metrics = ckpt.get('train_metrics') or {}
        if metrics:
            add('  保存那一刻的指标（train_metrics）：')
            for k in ('metrics/precision(B)', 'metrics/recall(B)',
                      'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness'):
                if k in metrics:
                    add('    %-24s: %.5f' % (k, float(metrics[k])))
        add('')
        add('  关于几个“看起来像空值”的字段（都是训练正常结束的标志，不是错误）：')
        add('    epoch = %s          → 训练已结束（保存时被置为 -1）' % ckpt.get('epoch'))
        add('    optimizer = %s      → 优化器状态已被剥离，权重只剩推理所需内容' % ckpt.get('optimizer'))
        add('    ema = %s            → EMA 权重已合并进 model 字段' % ckpt.get('ema'))
        add('    best_fitness = %s   → 训练结束后不再需要' % ckpt.get('best_fitness'))
        add('  这也是权重文件只有 5.24 MB 的原因（未剥离时约 11 MB）。')
        data_arg = str(train_args.get('data', ''))
        checks.append(('checkpoint 记录了本项目的数据集配置',
                       data_arg.endswith('data.yaml') and 'yolo_project' in data_arg, data_arg))
        checks.append(('checkpoint 记录了基础模型 yolo11n.pt',
                       train_args.get('model') == 'yolo11n.pt', str(train_args.get('model'))))
        checks.append(('checkpoint 记录了 100 轮 / imgsz 640 / batch 8',
                       train_args.get('epochs') == 100 and train_args.get('imgsz') == 640
                       and train_args.get('batch') == 8,
                       'epochs=%s imgsz=%s batch=%s' % (train_args.get('epochs'),
                                                        train_args.get('imgsz'),
                                                        train_args.get('batch'))))
    else:
        add('  [WARN] checkpoint 中没有 train_args')
    add('')

    # ---------- [4] 与训练日志交叉验证 ----------
    add('[4] 与 results.csv 交叉验证（权重指标必须能在训练日志里找到对应轮次）')
    metrics = ckpt.get('train_metrics') or {}
    row, diff = find_epoch_in_csv(metrics)
    if row is not None:
        add('  checkpoint 记录的四个指标：')
        add('    P=%.5f  R=%.5f  mAP50=%.5f  mAP50-95=%.5f'
            % (float(metrics['metrics/precision(B)']), float(metrics['metrics/recall(B)']),
               float(metrics['metrics/mAP50(B)']), float(metrics['metrics/mAP50-95(B)'])))
        add('  在 results.csv 中找到四个指标同时一致的一轮（最大误差 %.1e）：' % diff)
        add('    epoch=%s  P=%s  R=%s  mAP50=%s  mAP50-95=%s'
            % (row['epoch'], row['metrics/precision(B)'], row['metrics/recall(B)'],
               row['metrics/mAP50(B)'], row['metrics/mAP50-95(B)']))
        add('  → 说明 best.pt 就是第 %s 轮保存下来的权重，可追溯到具体训练记录。' % row['epoch'])
        checks.append(('checkpoint 指标与 results.csv 完全对应', True,
                       'epoch=%s，最大误差 %.1e' % (row['epoch'], diff)))
    else:
        add('  [WARN] 未能在 results.csv 中找到对应轮次')
        checks.append(('checkpoint 指标与 results.csv 完全对应', False, '未找到'))
    add('  注：Ultralytics 用 fitness 选择 best.pt，本版本 fitness 权重为 [0,0,0,1]，')
    add('      即 fitness 完全等于 mAP50-95 —— 所以 best.pt 是“mAP50-95 最高的那一轮”，')
    add('      不一定等于“mAP50 最高的那一轮”。')
    add('')

    # ---------- [5] 与官方 COCO 预训练权重对比 ----------
    add('[5] 与官方 COCO 预训练权重对比')
    if PRETRAIN.exists():
        pre = YOLO(str(PRETRAIN))
        pre_names = dict(sorted(pre.names.items()))
        sample = ', '.join(pre_names[i] for i in sorted(pre_names)[:8])
        add('  官方 %s : %d 类，例如 %s ...' % (PRETRAIN.name, len(pre_names), sample))
        add('  本项目 best.pt : %d 类，%s' % (len(names), name_list))
        add('  → 官方权重是 80 类通用模型；本模型只有 3 类且正好是本项目的目标类别。')
        checks.append(('与官方 COCO 权重类别数不同（80 vs 3）',
                       len(pre_names) == 80 and len(names) == 3,
                       'pretrain=%d, ours=%d' % (len(pre_names), len(names))))
    else:
        add('  [WARN] 未找到 %s，跳过对比' % PRETRAIN.name)
        checks.append(('与官方 COCO 权重对比', False, '缺少 %s' % PRETRAIN.name))
    add('')

    # ---------- [6] 结论 ----------
    add('[6] 结论')
    for desc, ok, extra in checks:
        add('  [%s] %s%s' % ('PASS' if ok else 'FAIL', desc,
                             ('（%s）' % extra) if extra else ''))
    all_ok = all(ok for _, ok, _ in checks)
    add('')
    add('  总结：%s' % ('模型加载正确，确认是使用本项目数据集（data.yaml）自行训练的权重，'
                        '可用于 Level 4 第 2 步的新图片推理。' if all_ok else
                        '存在未通过的检查项，请先排查后再进行推理。'))
    add('')
    add('后续用法（Level 4 第 2 步）：')
    add('  results = model.predict(source="test_images", imgsz=640, conf=0.25,')
    add('                          save=True, project="results", name="predict")')
    add('  或直接运行：python predict.py')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '01_加载模型.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    print('\n（已写入 %s）' % (OUT / '01_加载模型.txt'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
