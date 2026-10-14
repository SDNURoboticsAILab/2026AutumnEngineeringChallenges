# -*- coding: utf-8 -*-
"""读取最终模型的验证指标与混淆矩阵（把图上的信息变成精确数字，便于写报告）。

重要（方向问题）：
    Ultralytics 的混淆矩阵索引是 matrix[预测类别, 真实类别]。
    证据（ultralytics/utils/metrics.py，8.4.155）：
      · 第 434 行  self.matrix[dc, self.nc] += 1   # FP → 第 1 个下标是预测类别
      · 第 440 行  self.matrix[self.nc, gc] += 1   # FN → 第 2 个下标是真实类别
      · 第 593 行  ax.set_xlabel("True") / 第 594 行 ax.set_ylabel("Predicted")
      · 第 551 行  归一化用 matrix.sum(0)（按列），与“列 = 真实类别”一致
    因此正确读法是：
        行 = Predicted（预测类别），列 = True（真实类别）
        最后一列 = 真实背景 → 误检（FP）
        最后一行 = 预测为背景 → 漏检（FN）
    本脚本会用“列求和 = 该类真实实例数”做自检，直接把方向验证出来。

另有两点会影响数字大小，报告中需说明：
    · 混淆矩阵使用固定阈值 conf=0.25、IoU=0.45（metrics.py 第 408~409 行）；
    · 终端报告的 P / R 取 PR 曲线上 F1 最大点，工作点不同，数值不能直接互比。

用法：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe _read_metrics.py
"""
import sys
from pathlib import Path

from ultralytics import YOLO

PROJ = Path(__file__).resolve().parent
WEIGHTS = PROJ / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'


def main():
    m = YOLO(str(WEIGHTS))
    # project 用绝对路径：避免相对路径在已有 run 目录下被再次嵌套
    r = m.val(data=str(PROJ / 'data.yaml'), imgsz=640, device=0, workers=0,
              plots=True, verbose=False,
              project=str(PROJ / 'runs' / 'detect'), name='val_final', exist_ok=True)

    cls_names = [m.names[i] for i in sorted(m.names)]
    names = cls_names + ['background']

    cm = r.confusion_matrix.matrix
    n = len(names)

    print('\n===== 混淆矩阵（行 = 预测 Predicted，列 = 真实 True）=====')
    print('%-14s%s' % ('pred\\true', ''.join('%-12s' % x for x in names)))
    for i in range(n):
        print('%-14s%s' % (names[i], ''.join('%-12d' % int(cm[i][j]) for j in range(n))))

    col_sum = [sum(int(cm[i][j]) for i in range(n)) for j in range(n)]
    row_sum = [sum(int(cm[i][j]) for j in range(n)) for i in range(n)]

    print('\n----- 方向自检：列求和应等于每一类的真实实例数 -----')
    for j, name in enumerate(cls_names):
        print('  真实 %-9s 实例数 = %d' % (name, col_sum[j]))
    print('  真实 background（= 总误检数 FP） = %d' % col_sum[n - 1])
    print('  预测为 background（= 总漏检数 FN） = %d' % row_sum[n - 1])

    print('\n----- 分类别检出情况（由矩阵推导，阈值 conf=0.25 / IoU=0.45）-----')
    total_gt = sum(col_sum[:n - 1])
    total_fn = row_sum[n - 1]
    total_fp = col_sum[n - 1]
    for j, name in enumerate(cls_names):
        tp = int(cm[j][j])
        fn = int(cm[n - 1][j])
        fp = int(cm[j][n - 1])
        print('  %-9s 真实=%-4d 检出(TP)=%-4d 漏检(FN)=%-4d 误检(FP)=%-4d'
              % (name, col_sum[j], tp, fn, fp))
    print('  合计      真实=%-4d 检出=%-4d 漏检=%-4d 误检=%-4d'
          % (total_gt, total_gt - total_fn, total_fn, total_fp))

    print('\n===== 各类指标（取 PR 曲线 F1 最大点）=====')
    print('%-10s %8s %8s %8s %8s' % ('class', 'P', 'R', 'mAP50', 'mAP50-95'))
    for k, c in enumerate(r.box.ap_class_index):
        print('%-10s %8.3f %8.3f %8.3f %8.3f'
              % (m.names[c], float(r.box.p[k]), float(r.box.r[k]),
                 float(r.box.ap50[k]), float(r.box.ap[k])))
    print('%-10s %8.3f %8.3f %8.3f %8.3f'
          % ('all', float(r.box.mp), float(r.box.mr), float(r.box.map50), float(r.box.map)))
    print('\nfitness（= mAP50-95，本版本权重 [0,0,0,1]）: %.5f' % float(r.box.fitness()))
    print('速度: %.1f ms/张（推理）' % r.speed.get('inference', 0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
