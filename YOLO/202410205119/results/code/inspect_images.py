#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_images.py —— 读 dataset/images/ 下 6 张图的真实信息，输出标注前需要知道的东西
================================================================================
项目位置: 2026XXXXXX/inspect_images.py（辅助工具，非提交必需）

【为什么需要它】
Level 2 的标注必须用【图片真实分辨率】换算坐标。
现有 6 个标签是按 640x480 假设写的占位坐标；一旦真实图片分辨率不同，
就必须按新尺寸重新换算，否则框会偏。

【它会输出】
  1. 每张图的真实分辨率（宽 x 高）、色彩模式、文件大小；
  2. 该图当前标签内容的解析结果（类别的中文名 + 像素框坐标 + 是否越界）；
  3. 一份可直接填写的"标注清单"模板（每张图每个目标一行）；
  4. 若图片分辨率 != 标签换算所用尺寸，明确警告。

【运行】
    python inspect_images.py                    # 检查 train + val
    python inspect_images.py --ref-size 640x480 # 指定标签当前假定的尺寸
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# 本文件已收纳到 results/code/ 下，因此项目根目录需要上溯两级：
# results/code/xxx.py -> results/code -> results -> 项目根（你的学号/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 类别编号 -> 中文名（与 data.yaml 的 names 一致，仅用于打印可读性）
CLASS_ZH = {0: "obstacle 障碍物", 1: "cola 可乐", 2: "football 足球"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="检查 dataset/images 下图片的实际信息，并解析对应标签",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref-size", default="640x480",
                   help="现有标签换算时假定的图片尺寸，格式 WxH")
    return p.parse_args()


def parse_ref_size(s: str) -> tuple[int, int]:
    """把 '640x480' 解析成 (640, 480)，即 (宽, 高)。"""
    try:
        w, h = s.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except Exception:
        print(f"[!] --ref-size 格式不对: {s!r}，应形如 640x480。改用默认 640x480。")
        return 640, 480


def read_label(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读取 YOLO 标签，返回 [(class_id, cx, cy, w, h), ...]（均为归一化值）。"""
    out = []
    if not txt_path.is_file():
        return out
    for ln, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"      [!] {txt_path.name} 第 {ln} 行字段数为 {len(parts)}，应为 5，已跳过")
            continue
        try:
            out.append((int(parts[0]), *[float(x) for x in parts[1:]]))
        except ValueError:
            print(f"      [!] {txt_path.name} 第 {ln} 行含非数字，已跳过")
    return out


def main() -> None:
    args = parse_args()
    ref_w, ref_h = parse_ref_size(args.ref_size)

    try:
        from PIL import Image
    except ImportError:
        print("[X] 未安装 Pillow。请先执行: pip install pillow")
        return

    print("=" * 78)
    print(" dataset/images 实际图片信息检查")
    print("=" * 78)
    print(f" 标签当前假定的参考尺寸: {ref_w} x {ref_h}（宽 x 高）")
    print()

    total_imgs = 0
    size_mismatch: list[str] = []
    no_label: list[str] = []
    empty_label: list[str] = []
    oob: list[str] = []

    for split in ("train", "val"):
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split

        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS) if img_dir.is_dir() else []

        print("-" * 78)
        print(f"【{split}】 {img_dir}")
        print("-" * 78)

        if not images:
            print("  （该目录暂无图片）")
            print()
            continue

        for img in images:
            total_imgs += 1
            with Image.open(img) as im:
                w, h = im.size
                mode = im.mode
            print(f"  {img.name}")
            print(f"      分辨率 : {w} x {h}   色彩模式: {mode}   大小: {img.stat().st_size} B")

            # 分辨率与标签换算尺寸是否一致
            if (w, h) != (ref_w, ref_h):
                msg = f"{split}/{img.name} 实际 {w}x{h}，标签按 {ref_w}x{ref_h} 换算"
                size_mismatch.append(msg)
                print(f"      ⚠ 分辨率与标签假定尺寸不同 → 坐标必须重新换算！")

            # 对应标签
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.is_file():
                no_label.append(f"{split}/{img.name}")
                print(f"      标签   : ❌ 缺少 {lbl.name}")
                # 若没有真实图片则无像素框可算，仅提示
                print()
                continue

            boxes = read_label(lbl)
            if not boxes:
                empty_label.append(f"{split}/{img.name}")
                print(f"      标签   : ⚠ {lbl.name} 存在但内容为空（Make-Sense 导出 bug 的典型表现）")
                print()
                continue

            print(f"      标签   : {lbl.name} 共 {len(boxes)} 个目标")
            for i, (cid, cx, cy, bw, bh) in enumerate(boxes, 1):
                # 归一化中心点 -> 像素左上/右下（按图片真实分辨率换算）
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                name = CLASS_ZH.get(cid, f"未知类别 {cid}")
                flag = ""
                if x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1:
                    flag = "  ⚠ 框已超出图片范围"
                    oob.append(f"{split}/{img.name} 目标{i}")
                print(f"        目标{i}: {name:<16} 像素框 ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
                      f"  归一化 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}{flag}")
            print()

    # ---------------- 汇总 ----------------
    print("=" * 78)
    print(" 汇总")
    print("=" * 78)
    print(f"  图片总数            : {total_imgs}")
    print(f"  缺少标签的图片      : {len(no_label)} {no_label}")
    print(f"  标签为空的图片      : {len(empty_label)} {empty_label}")
    print(f"  分辨率不一致的图片  : {len(size_mismatch)}")
    for m in size_mismatch:
        print(f"      - {m}")
    print(f"  框超出图片范围的    : {len(oob)} {oob}")
    print()

    if total_imgs == 0:
        print("  ⚠ dataset/images 下还没有图片。")
        print("    请把 6 张原图按以下文件名放入后重跑本脚本：")
        print("      dataset/images/train/  img01.jpg  img02.jpg  img03.jpg  img04.jpg")
        print("      dataset/images/val/    img05.jpg  img06.jpg")
    elif size_mismatch:
        print("  → 存在分辨率不一致：现有标签的坐标不能用，必须按真实分辨率重新标注/换算。")
        print("    换算方法见 report.md 3.3 节，或用 add_annotation.py 直接写入新坐标。")
    else:
        print("  → 所有图片分辨率与标签假定尺寸一致，现有坐标可直接使用。")

    print()
    print("=" * 78)
    print(" 标注清单模板（把每个目标的像素坐标填进去，再用 add_annotation.py 写入）")
    print("=" * 78)
    print("""
  # 格式: <图片文件名> <类别编号> <x_min> <y_min> <x_max> <y_max>
  # 类别编号: 0=obstacle  1=cola  2=football
  # 像素坐标以图片【左上角】为原点，x 向右、y 向下

  img01.jpg  0  <x_min> <y_min> <x_max> <y_max>
  img01.jpg  1  <x_min> <y_min> <x_max> <y_max>
  img01.jpg  2  <x_min> <y_min> <x_max> <y_max>

  img02.jpg  0  <x_min> <y_min> <x_max> <y_max>
  img02.jpg  1  <x_min> <y_min> <x_max> <y_max>
  img02.jpg  2  <x_min> <y_min> <x_max> <y_max>

  img03.jpg  1  <x_min> <y_min> <x_max> <y_max>

  img04.jpg  2  <x_min> <y_min> <x_max> <y_max>

  img05.jpg  0  <x_min> <y_min> <x_max> <y_max>
  img05.jpg  1  <x_min> <y_min> <x_max> <y_max>

  img06.jpg  2  <x_min> <y_min> <x_max> <y_max>
  img06.jpg  1  <x_min> <y_min> <x_max> <y_max>
""")
    print("  提示：把上面这段存成 annotations.txt，然后执行")
    print("        python add_annotation.py --from-file annotations.txt")
    print("        脚本会按每张图的真实分辨率换算成 YOLO 归一化坐标并写入 labels/。")


if __name__ == "__main__":
    main()
