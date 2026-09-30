#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
show_labels.py —— 把 YOLO 标签画回原图，用于核对标注是否正确 / 生成标注截图
================================================================================
位置: results/code/show_labels.py（辅助脚本）

【作用】
读取 dataset/labels/ 下的 .txt 标签，按 YOLO 归一化坐标反算回像素坐标，
画到对应图片上，输出到 label_check/ 目录。
这样就能用肉眼直接检查"标注的框有没有贴住目标"。

同时它也是 Level 2 要求提交的「已完成目标框标注的图片截图」的来源。

【Windows 运行命令】（在项目根目录执行）
    python results\code\show_labels.py                 :: 全部 6 张
    python results\code\show_labels.py --stem img03    :: 只画某一张

【输出】
    label_check/train_img01.jpg ~ train_img05.jpg
    label_check/val_img02.jpg ~ val_img06.jpg
    图上的框旁会标注「类别中文名 置信度位置」以及类别编号。

【路径规范】用 __file__ 上溯两级定位项目根，无绝对路径。
"""

from __future__ import annotations

import argparse
from pathlib import Path

# 本文件在 results/code/ 下，项目根需上溯两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "dataset"
OUT_DIR = PROJECT_ROOT / "label_check"

# 类别编号 -> (中文名, 英文名)
CLASSES = {0: ("障碍物", "obstacle"), 1: ("可乐", "cola"), 2: ("足球", "football")}

# 每个类别一种颜色（RGB）
COLORS = {0: (255, 60, 60), 1: (0, 200, 0), 2: (255, 190, 0)}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把 YOLO 标签画回原图，用于核对标注",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stem", default="",
                   help="只处理指定图片名（不含扩展名），如 img03；留空则处理全部")
    p.add_argument("--line-width", type=int, default=3, help="框线宽")
    p.add_argument("--no-zh", action="store_true", help="标签只用英文类别名")
    return p.parse_args()


def find_font(size: int = 18):
    """找一个可用的中文字体；找不到返回 (None, False)。"""
    import os
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            try:
                from PIL import ImageFont
                return ImageFont.truetype(p, size), True
            except Exception:
                continue
    return None, False


def main() -> None:
    args = parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[X] 未安装 Pillow，请执行: pip install pillow")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font, zh_ok = find_font(18)

    total_img = 0
    total_box = 0
    missing = []                       # 记录缺标签的图片

    for split in ("train", "val"):
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        if not img_dir.is_dir():
            continue

        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS)
        for img_path in images:
            # 指定了 --stem 就只处理那一张
            if args.stem and img_path.stem != args.stem:
                continue

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            # 用 Pillow 读图：cv2.imread 在 Windows 上读不了含中文的路径
            with Image.open(img_path) as im:
                img = im.convert("RGB")
                W, H = img.size

            draw = ImageDraw.Draw(img)
            boxes = []

            if lbl_path.is_file():
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"  [!] {lbl_path.name} 有行字段数不是 5，已跳过: {line!r}")
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:])

                    # ---- 归一化 -> 像素（YOLO 公式的逆运算）----
                    # x_center=(xmin+xmax)/2/W  =>  xmin = (cx - bw/2) * W
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W
                    y2 = (cy + bh / 2) * H
                    boxes.append((cid, x1, y1, x2, y2))
            else:
                missing.append(f"{split}/{img_path.name}")

            # 画框 + 写类别名
            for cid, x1, y1, x2, y2 in boxes:
                color = COLORS.get(cid, (0, 255, 0))
                for k in range(args.line_width):
                    draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)

                zh, en = CLASSES.get(cid, ("?", "?"))
                label = en if args.no_zh else zh
                label = f"{label}({cid})"

                if zh_ok:
                    try:
                        tb = draw.textbbox((0, 0), label, font=font)
                        tw, th = tb[2] - tb[0], tb[3] - tb[1]
                    except Exception:
                        tw, th = 10 * len(label), 18
                    ty = y1 - th - 6
                    if ty < 0:
                        ty = y1 + 3
                    draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
                    draw.text((x1 + 4, ty + 2), label, fill=(0, 0, 0), font=font)
                else:
                    # 没中文字体就退回简单英文标注（不依赖字体文件）
                    draw.rectangle([x1, y1 - 14, x1 + 10 * len(label), y1], fill=color)
                    draw.text((x1 + 2, y1 - 13), label, fill=(0, 0, 0))

            out_path = OUT_DIR / f"{split}_{img_path.name}"
            img.save(out_path)
            total_img += 1
            total_box += len(boxes)
            print(f"  {split}/{img_path.name}  ({W}x{H}, {len(boxes)} 个框)"
                  f"  ->  {out_path.relative_to(PROJECT_ROOT)}")

    print()
    print("=" * 70)
    print(f" 完成：处理 {total_img} 张图片，共画出 {total_box} 个目标框")
    print(f" 输出目录：{OUT_DIR.relative_to(PROJECT_ROOT)}")
    if missing:
        print(f" [!] 以下图片缺少标签文件：{missing}")
    print(" 提示：本目录下的图可直接用于 Level 2「已完成标注的图片截图」")
    print("=" * 70)


if __name__ == "__main__":
    main()
