# -*- coding: utf-8 -*-
"""生成 Level 3「验证标准与提交内容」的验收证据文件（可直接截图用）。

对应考核 Level 3 的 5 条要求：
  1. 模型正常开始训练的终端输出
  2. 训练完成后的结果目录
  3. Loss / Precision / Recall 训练结果图（results.png 等，本身就是文件）
  4. 训练得到的模型权重文件信息
  5. 使用训练得到的模型完成一次验证或测试

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe make_level3_evidence.py
输出目录：results/evidence_level3/
"""
import csv
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
RUN = PROJ / 'runs' / 'detect' / 'train'
OUT = PROJ / 'results' / 'evidence_level3'
PY = sys.executable
TRAIN_LOG = Path(r'C:\Users\dell\AppData\Local\Temp\dsh-subprocess-l9oyLf'
                 r'\dsh-subprocess-12164-2-997b3e3a8d84-stdout.log')


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding='utf-8')
    print('已生成 %s（%d 字符）' % (name, len(text)))


def real_data_path():
    """从 args.yaml 读取真实的数据集配置路径。

    终端日志在抓取过程中丢失了中文目录名（"研一""练习"），
    而 args.yaml 是 Ultralytics 直接以 UTF-8 写出的，路径完整可信。
    """
    try:
        for line in (RUN / 'args.yaml').read_text(encoding='utf-8').splitlines():
            if line.startswith('data:'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return ''


def path_note():
    """针对日志中中文目录名缺失的说明（不是修改日志内容，只是加注解）。"""
    p = real_data_path()
    return ('\n【路径说明】终端日志在抓取过程中丢失了中文目录名（"研一""练习"），\n'
            '  因此上文出现的 "D:\\git \\yolo_project\\..." 的真实路径为：\n'
            '    %s\n'
            '  该路径以 runs/detect/train/args.yaml 中的 data 字段为准。' % p)


def extract_train_log():
    if not TRAIN_LOG.exists():
        return None, None
    raw = TRAIN_LOG.read_bytes()
    text = raw.decode('utf-8', errors='ignore')
    lines = [l.rstrip() for l in text.replace('\r', '').split('\n')]
    # 去掉进度条那类刷屏行
    clean = [l for l in lines if '━' not in l and 'it/s' not in l and l.strip()]
    start = [l for l in clean if any(k in l for k in
             ('Ultralytics ', 'CUDA:0', 'engine\\trainer', 'Overriding model.yaml',
              'YOLO11n summary', 'Transferred', 'train: Fast image', 'train: Scanning',
              'train: New cache', 'Caching images'))]
    end = [l for l in clean if any(k in l for k in
            ('epochs completed', 'Optimizer stripped', 'Validating', 'Class', 'obstacle',
             'cola', 'football', 'all ', 'Speed:', 'Results saved', '计'))]
    return start, end


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1) 训练启动 ----
    start, end = extract_train_log()
    if start:
        write('01_训练启动输出.txt',
              '=== 模型开始训练的终端输出（Level 3 要求 1）===\n\n' + '\n'.join(start[:30])
              + '\n' + path_note())
    elif not (OUT / '01_训练启动输出.txt').exists():
        write('01_训练启动输出.txt',
              '（未找到训练日志，请重新运行 train.py 后截图终端）\n')
    else:
        print('未找到训练日志，保留已有的 01_训练启动输出.txt（不覆盖）')

    # ---- 2) 训练完成 + 结果目录 ----
    body = ['=== 训练结束与最终验证（Level 3 要求 1、5）===', '']
    body += (end[-30:] if end else ['（未找到日志尾部）'])
    body += ['', '=== 训练结果目录 runs/detect/train（Level 3 要求 2）===', '']
    for p in sorted(RUN.iterdir()):
        if p.is_dir():
            body.append('%-34s <dir>' % (p.name + '/'))
        else:
            body.append('%-34s %8.1f KB' % (p.name, p.stat().st_size / 1024))
    body += ['', '=== 各文件说明 ===']
    for name, desc in [
            ('results.png', '训练曲线总览：三 loss + P + R + mAP50 + mAP50-95'),
            ('results.csv', '每轮指标原始数据（100 行）'),
            ('confusion_matrix.png', '混淆矩阵'),
            ('BoxPR_curve.png / BoxP/R/F1_curve.png', '精度-召回曲线，用于选置信度阈值'),
            ('val_batch0~2_pred.jpg', '验证集预测效果'),
            ('val_batch0~2_labels.jpg', '验证集真值（与上一行对照看漏检/误检）'),
            ('labels.jpg', '数据集标签分布总览'),
            ('args.yaml', '本次训练的全部超参数'),
            ('weights/best.pt', '最佳权重'),
            ('weights/last.pt', '最后一轮权重')]:
        body.append('  %-38s %s' % (name, desc))
    write('02_训练结束_结果目录.txt', '\n'.join(body) + '\n' + path_note())

    # ---- 3) 指标汇总 ----
    rows = list(csv.reader((RUN / 'results.csv').open(encoding='utf-8')))
    header, data = rows[0], rows[1:]
    idx = {k.strip(): i for i, k in enumerate(header)}

    def col(r, key):
        return float(r[idx[key]])

    # Ultralytics 用 fitness 选择 best.pt；本版本 fitness 权重为 [0,0,0,1]，
    # 即 fitness 完全等于 mAP50-95 —— 所以 best.pt 对应的是 mAP50-95 最高的那一轮，
    # 不一定等于 mAP50 最高的那一轮。这里两个都算出来，避免标注错轮次。
    best95_i = max(range(len(data)), key=lambda i: col(data[i], 'metrics/mAP50-95(B)'))
    best50_i = max(range(len(data)), key=lambda i: col(data[i], 'metrics/mAP50(B)'))

    def one_line(tag, i):
        r = data[i]
        return ('%s (epoch %s)：P=%.3f  R=%.3f  mAP50=%.3f  mAP50-95=%.3f'
                % (tag, r[idx['epoch']], col(r, 'metrics/precision(B)'),
                   col(r, 'metrics/recall(B)'), col(r, 'metrics/mAP50(B)'),
                   col(r, 'metrics/mAP50-95(B)')))

    txt = ['=== 训练指标汇总（Level 3 要求 3）===', '']
    txt.append('训练轮数：%d（跑满）' % len(data))
    txt.append(one_line('末轮', len(data) - 1))
    txt.append(one_line('best.pt 对应轮次（按 fitness = mAP50-95 选中）', best95_i))
    if best50_i != best95_i:
        txt.append(one_line('mAP50 最高的轮次（仅作对比，不是 best.pt）', best50_i))
    txt += ['', '关键节点（Loss / Precision / Recall / mAP）：', '',
            '%-7s %-10s %-10s %-10s %-8s %-8s %-8s %-9s' %
            ('epoch', 'box_loss', 'cls_loss', 'dfl_loss', 'P', 'R', 'mAP50', 'mAP50-95')]
    picks = [0, 4, 9, 19, 29, 39, 49, best50_i, best95_i, 69, 79, 89, len(data) - 1]
    for i in sorted(set(picks)):
        r = data[i]
        txt.append('%-7s %-10.3f %-10.3f %-10.3f %-8.3f %-8.3f %-8.3f %-9.3f' % (
            r[idx['epoch']], col(r, 'train/box_loss'), col(r, 'train/cls_loss'),
            col(r, 'train/dfl_loss'), col(r, 'metrics/precision(B)'),
            col(r, 'metrics/recall(B)'), col(r, 'metrics/mAP50(B)'),
            col(r, 'metrics/mAP50-95(B)')))
    write('03_指标汇总.txt', '\n'.join(txt))

    # ---- 4) 权重信息 ----
    wdir = RUN / 'weights'
    lines = ['=== 模型权重文件信息（Level 3 要求 4）===', '', '文件列表：']
    for f in sorted(wdir.iterdir()):
        lines.append('  %-12s %7.2f MB   修改时间 %s'
                     % (f.name, f.stat().st_size / 1024 / 1024,
                        f.stat().st_mtime and __import__('datetime').datetime.fromtimestamp(
                            f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')))
    lines += ['', '模型加载信息：']
    try:
        out = subprocess.run(
            [PY, '-c',
             "from ultralytics import YOLO;"
             "m=YOLO(r'%s');"
             "print('任务:', m.task);"
             "print('类别:', m.names);"
             "print('参数量:', sum(p.numel() for p in m.model.parameters()))" % (wdir / 'best.pt')],
            capture_output=True, text=True, timeout=300,
            encoding='utf-8', errors='replace')
        lines += ['  ' + l for l in (out.stdout or out.stderr).strip().splitlines() if l.strip()]
    except Exception as exc:
        lines.append('  加载失败: %s' % exc)
    write('04_权重文件信息.txt', '\n'.join(lines))

    # ---- 5) 验证与测试 ----
    # 注意：本段会调用 predict.py 对 test_images/ 里的“当前全部图片”做推理。
    # 而 test_images/ 的内容会随实验进展变化（曾把 6 张图片移出数据集用于 Level 4 严格评测），
    # 直接重生成会把原始的测试记录覆盖掉。因此默认只在文件不存在时才生成；
    # 确实想用当前 test_images/ 重跑，把下面的开关改成 True。
    REFRESH_PREDICT = False

    if not REFRESH_PREDICT and (OUT / '05_验证与测试.txt').exists():
        print('已存在 05_验证与测试.txt 且 REFRESH_PREDICT=False，保留原文件不覆盖')
    else:
        lines = ['=== 使用自己训练的模型完成验证与测试（Level 3 要求 5）===', '',
                 '[A] 验证：在验证集上评估 runs/detect/train/weights/best.pt', '']
        lines += (end[-12:] if end else ['（见 02 文件）'])
        lines += [path_note()]
        lines += ['', '[B] 测试：对 test_images/ 里的图片推理', '']
        try:
            out = subprocess.run(
                [PY, str(PROJ / 'predict.py')], capture_output=True, text=True, timeout=600,
                encoding='utf-8', errors='replace',
                cwd=str(PROJ), env={**__import__('os').environ,
                                    'MPLCONFIGDIR': str(PROJ / '_mplcache')})
            lines += [l for l in (out.stdout or '').splitlines() if l.strip()]
            if out.returncode != 0 and out.stderr:
                lines.append('（stderr）' + out.stderr.strip()[-500:])
        except Exception as exc:
            lines.append('  推理失败: %s' % exc)
        write('05_验证与测试.txt', '\n'.join(lines))

    print('\n全部证据已生成到: %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
    
