# -*- coding: utf-8 -*-
"""
模型推理/检测脚本
=================
功能：使用训练好的 best.pt 对单张图片或整个文件夹进行目标检测，
      输出带【目标框 + 类别名称 + 置信度】的效果图，可选保存 YOLO 标签。

用法：
    python predict.py --source 待检测图片文件夹
    python predict.py --source 图片.jpg --conf 0.3 --save-txt

参数：
    --source   待检测图片路径或文件夹（必填）
    --weights  权重路径（默认 runs/weights/best.pt）
    --conf     置信度阈值（默认 0.25）
    --imgsz    推理尺寸（默认 640）
    --save-txt 同时保存 YOLO 格式检测标签
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# ---------- 路径配置 ----------
BASE = Path(__file__).resolve().parent                          # 项目根目录
DEFAULT_WEIGHTS = BASE / "runs" / "weights" / "best.pt"
RUNS_DIR = BASE / "runs"                                        # 预测输出目录


def main():
    parser = argparse.ArgumentParser(description="YOLO11n 目标检测推理")
    parser.add_argument("--source", required=True, help="待检测的图片路径或文件夹（必填）")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸（默认 640）")
    parser.add_argument("--save-txt", action="store_true", help="同时保存 YOLO 标签结果")
    parser.add_argument("--name", default="predict", help="本次预测输出名称")
    args = parser.parse_args()

    source = Path(args.source)
    assert source.exists(), f"输入路径不存在: {source}"
    assert Path(args.weights).exists(), f"权重不存在: {args.weights}"

    model = YOLO(args.weights)
    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=0,
        save=True,               # 保存带框效果图
        save_txt=args.save_txt,  # 可选：保存 YOLO 标签
        save_conf=True,          # 标签文件附带置信度
        show_labels=True,        # 显示类别名称
        show_conf=True,          # 显示置信度
        line_width=3,            # 框线加粗，效果图清晰
        workers=0,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
    )
    print(f"检测完成，结果保存在: {RUNS_DIR / args.name}")


if __name__ == "__main__":
    main()
