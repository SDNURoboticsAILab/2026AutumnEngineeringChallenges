# -*- coding: utf-8 -*-
"""
模型验证脚本
============
功能：用训练好的 best.pt 在验证集上评估，输出 mAP50、mAP50-95、
      Precision、Recall 及各类别指标。

用法：
    python val.py
"""
from pathlib import Path

from ultralytics import YOLO

# ---------- 路径配置 ----------
BASE = Path(__file__).resolve().parent                          # 项目根目录
WEIGHTS = BASE / "runs" / "weights" / "best.pt"                 # 训练好的权重
DATA_YAML = BASE / "data.yaml"                                  # 数据集配置
RUNS_DIR = BASE / "runs"                                        # 输出目录


def main():
    assert WEIGHTS.exists(), f"权重不存在: {WEIGHTS}"
    assert DATA_YAML.exists(), f"数据集配置不存在: {DATA_YAML}"

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        project=str(RUNS_DIR),
        name="val_submit",
        exist_ok=True,
    )

    box = metrics.box
    print("=" * 50)
    print("验证结果")
    print("=" * 50)
    print(f"  整体 mAP50     : {box.map50:.4f}")
    print(f"  整体 mAP50-95  : {box.map:.4f}")
    print(f"  整体 Precision : {box.mp:.4f}")
    print(f"  整体 Recall    : {box.mr:.4f}")
    print("  各类别指标：")
    print(f"    {'类别':<10}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for i, name in model.names.items():
        print(f"    {name:<10}{box.p[i]:>8.3f}{box.r[i]:>8.3f}"
              f"{box.ap50[i]:>9.3f}{box.ap[i]:>10.3f}")


if __name__ == "__main__":
    main()
