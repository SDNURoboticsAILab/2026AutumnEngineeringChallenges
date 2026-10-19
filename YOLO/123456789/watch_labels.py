# -*- coding: utf-8 -*-
"""实时监视标注：只要有新的 obstacle 框落盘，就在窗口里立刻打印出来。

用法（在 yolo_project 下，另开一个窗口运行，标注时挂着就行）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe watch_labels.py
    # 想同时看 batch_01 进度，默认每 5 秒刷新一次进度行

它每 2 秒扫描一次 dataset/labels/{train,val}：
    * 新增了带 obstacle(第 0 类) 框的标签文件 -> 立刻打印 [新增]
    * 某个已有标签文件里 obstacle 框数量变了 -> 打印 [变化]
    * 什么都没变就安静等待（每 30 秒打一个心跳）
这样无论你用 labelImg 还是 label_tool.py，都能第一时间确认“到底存进去了没有”。
"""
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
LABELS = PROJ / 'dataset' / 'labels'
SPLITS = ('train', 'val')


def snapshot():
    """返回 {相对路径: (obstacle框数量, 文件修改时间)}"""
    snap = {}
    for split in SPLITS:
        d = LABELS / split
        if not d.exists():
            continue
        for txt in d.glob('*.txt'):
            if txt.name == 'classes.txt':
                continue
            n = 0
            try:
                for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
                    parts = line.split()
                    if len(parts) == 5 and parts[0] == '0':
                        n += 1
            except OSError:
                continue
            if n:
                snap['%s/%s' % (split, txt.name)] = n
    return snap


def batch01_progress():
    """按 label_plan.csv 统计各批已完成数量"""
    plan = PROJ / 'label_plan.csv'
    if not plan.exists():
        return ''
    done = total = 0
    b1_done = b1_total = 0
    import csv
    with plan.open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            split, image, batch = row['split'], row['image'], row['batch']
            total += 1
            if batch == 'batch_01':
                b1_total += 1
            txt = LABELS / split / (Path(image).stem + '.txt')
            has = False
            if txt.exists():
                for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
                    parts = line.split()
                    if len(parts) == 5 and parts[0] == '0':
                        has = True
                        break
            if has:
                done += 1
                if batch == 'batch_01':
                    b1_done += 1
    return 'batch_01 %d/%d ｜ 全部 %d/%d' % (b1_done, b1_total, done, total)


def main():
    print('监视中：dataset/labels/{train,val}（每 2 秒扫描一次，Ctrl+C 退出）')
    print('  当前已有 obstacle 框的标签文件: %d 个' % len(snapshot()))
    prev = snapshot()
    last_beat = time.time()
    while True:
        try:
            time.sleep(2)
            cur = snapshot()
            for key in sorted(set(cur) - set(prev)):
                print('[新增] %s  含 obstacle 框 %d 个  <- 保存成功！' % (key, cur[key]))
            for key in sorted(set(prev) & set(cur)):
                if cur[key] != prev[key]:
                    print('[变化] %s  obstacle 框 %d -> %d' % (key, prev[key], cur[key]))
            prev = cur
            if time.time() - last_beat > 30:
                print('[心跳] 已有 obstacle 框的标签文件 %d 个 ｜ %s'
                      % (len(cur), batch01_progress()))
                last_beat = time.time()
        except KeyboardInterrupt:
            print('\n已停止。')
            return 0


if __name__ == '__main__':
    sys.exit(main())
