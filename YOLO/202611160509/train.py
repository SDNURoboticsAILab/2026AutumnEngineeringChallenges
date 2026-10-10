# -*- coding: utf-8 -*-
"""
模型训练脚本
============
功能：使用 YOLO11n 预训练权重在 dataset 数据集上训练目标检测模型，
      训练完成后自动将最佳权重复制到 runs/weights/best.pt。

用法：
    python train.py                      # 默认 50 epochs / batch 16 / imgsz 640
    python train.py --epochs 100 --batch 8

说明：
    - 默认使用 GPU（device=0）；无 GPU 机器改为 --device cpu
    - Windows 下 workers 必须为 0（多进程 dataloader 会与 CUDA 冲突）
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------- 路径配置（脚本位于项目根目录） ----------
BASE = Path(__file__).resolve().parent                  # 项目根目录
DATA_YAML = BASE / "data.yaml"                          # 数据集配置
PRETRAINED = BASE / "runs" / "weights" / "yolo11n.pt"   # 预训练权重
WEIGHTS_DIR = BASE / "runs" / "weights"                 # 权重保存目录
RUNS_DIR = BASE / "runs"                                 # 训练输出目录


def main():
    parser = argparse.ArgumentParser(description="YOLO11n 目标检测训练")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认 50）")
    parser.add_argument("--batch", type=int, default=16, help="批大小（默认 16）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸（默认 640）")
    parser.add_argument("--device", default="0", help="训练设备：0=GPU / cpu")
    parser.add_argument("--name", default="train_submit", help="本次训练名称")
    args = parser.parse_args()

    assert DATA_YAML.exists(), f"数据集配置不存在: {DATA_YAML}"
    assert PRETRAINED.exists(), f"预训练权重不存在: {PRETRAINED}"

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,          # Windows 下必须为 0，否则触发 CUDA illegal memory access
        cache=False,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        seed=0,
    )

    # 将最佳权重复制到 runs/weights/best.pt
    best = RUNS_DIR / args.name / "weights" / "best.pt"
    assert best.exists(), f"未找到训练权重: {best}"
    shutil.copy2(best, WEIGHTS_DIR / "best.pt")
    print(f"训练完成，最佳权重已保存: {WEIGHTS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
