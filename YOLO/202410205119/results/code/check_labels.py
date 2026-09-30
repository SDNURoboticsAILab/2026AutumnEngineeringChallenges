#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_labels.py —— 标注质量校验：图/标签配对 + 坐标合法性 + 与真实图片对账
================================================================================
项目位置: 2026XXXXXX/check_labels.py（辅助工具，非提交必需）

【为什么需要它】
本项目在 Level 2 踩过 Make-Sense 导出 0 字节标签的坑，
而且坐标必须与【图片真实分辨率】匹配。这两类问题都不会让训练报错，
只会让指标变成 0 —— 所以必须在训练前用脚本查一遍。

【它检查什么】
  1. 图片与标签是否【同名一一对应】（少标签、多标签、大小写不一致）；
  2. 每个标签文件是否为空（0 字节 / 只有空行）；
  3. 每行是否恰好 5 个字段、class_id 是否在 0/1/2；
  4. 后四项是否都在 0~1 之间且 >0（写成像素值会被 YOLO 丢弃）；
  5. 框是否越界（cx±w/2、cy±h/2 必须落在 0~1 内）；
  6. 编码是否为 UTF-8、是否带 BOM；
  7. 类别覆盖情况（train / val 是否都包含三个类别）；
  8. 标签文件数量与 dataset/images 的实际图片数是否相符。

【运行】
    python check_labels.py
    python check_labels.py --strict    # 有任何 warning 也以非 0 退出（可用于 CI）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 本文件已收纳到 results/code/ 下，因此项目根目录需要上溯两级：
# results/code/xxx.py -> results/code -> results -> 项目根（你的学号/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAME = {0: "obstacle", 1: "cola", 2: "football"}

errors: list[str] = []
warnings: list[str] = []
passed: list[str] = []


def ok(msg: str) -> None:
    passed.append(msg)


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="校验 YOLO 数据集标签质量")
    p.add_argument("--strict", action="store_true", help="出现 warning 也返回非 0")
    p.add_argument("--expect-per-split", default="train=4,val=2",
                   help="期望的图片数量，形如 train=4,val=2；留空则不检查")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    expected: dict[str, int] = {}
    if args.expect_per_split:
        for item in args.expect_per_split.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    expected[k.strip()] = int(v)
                except ValueError:
                    warn(f"--expect-per-split 中 {item!r} 不是整数，已忽略")

    print("=" * 78)
    print(" 标注质量校验 check_labels.py")
    print("=" * 78)

    for split in ("train", "val"):
        img_dir = DATASET / "images" / split
        lbl_dir = DATASET / "labels" / split
        print(f"\n--- {split} ---")

        if not img_dir.is_dir():
            err(f"{split}: 图片目录不存在 {img_dir}")
            continue
        if not lbl_dir.is_dir():
            err(f"{split}: 标签目录不存在 {lbl_dir}")
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        labels = sorted(p for p in lbl_dir.glob("*.txt") if not p.name.endswith(".bak"))

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        print(f"  图片 {len(images)} 个 / 标签 {len(labels)} 个")

        # 1) 一一对应检查
        only_img = sorted(img_stems - lbl_stems)
        only_lbl = sorted(lbl_stems - img_stems)
        if only_img:
            if images:
                err(f"{split}: {len(only_img)} 张图片没有同名标签 -> {only_img}")
            else:
                warn(f"{split}: 尚无图片，但有 {len(only_lbl)} 个标签待配对 -> {only_lbl}")
        if only_lbl:
            err(f"{split}: {len(only_lbl)} 个标签没有同名图片 -> {only_lbl}")
        if not only_img and not only_lbl:
            ok(f"{split}: 图片与标签同名一一对应（{len(img_stems)} 组）")

        # 2) 期望数量
        if split in expected and len(img_stems) != expected[split]:
            if not images:
                warn(f"{split}: 期望 {expected[split]} 张图片，实际 0 张（原图尚未放入）")
            else:
                err(f"{split}: 期望 {expected[split]} 张图片，实际 {len(img_stems)} 张")

        # 3~6) 逐文件校验标签内容
        for lbl in labels:
            raw = lbl.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                warn(f"{split}/{lbl.name}: 含 UTF-8 BOM，建议去掉（部分解析器会出错）")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                err(f"{split}/{lbl.name}: 不是 UTF-8 编码（{e}）")
                continue

            lines = [l for l in text.splitlines() if l.strip()]
            if not lines:
                err(f"{split}/{lbl.name}: 标签内容为空（0 字节或只有空行）"
                    f" —— 这是 Make-Sense 导出 bug 的典型症状")
                continue

            for ln, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    err(f"{split}/{lbl.name}:{ln}: 字段数 {len(parts)}，应为 5")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    err(f"{split}/{lbl.name}:{ln}: 含非数字")
                    continue

                if cid not in CLASS_NAME:
                    err(f"{split}/{lbl.name}:{ln}: class_id={cid} 不在 0/1/2 内")
                if not all(0.0 < v <= 1.0 for v in (cx, cy, w, h)):
                    err(f"{split}/{lbl.name}:{ln}: 坐标 {cx},{cy},{w},{h} 不在 (0,1] 内"
                        f" —— 很可能写成了像素值")
                    continue
                if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6:
                    err(f"{split}/{lbl.name}:{ln}: 框在 x 方向越界")
                if cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                    err(f"{split}/{lbl.name}:{ln}: 框在 y 方向越界")

        if labels and not any(e.startswith(f"{split}/") for e in errors):
            ok(f"{split}: {len(labels)} 个标签文件全部通过格式与坐标校验")

        # 7) 类别覆盖
        seen: set[int] = set()
        for lbl in labels:
            try:
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        seen.add(int(line.split()[0]))
            except Exception:
                pass
        if labels:
            missing = sorted(set(CLASS_NAME) - seen)
            if missing:
                names_missing = [CLASS_NAME[m] for m in missing]
                if split == "train":
                    # 训练集缺类别是硬错误：模型永远学不会没见过的类别
                    err(f"{split}: 缺少类别 {missing}（{names_missing}）"
                        f"—— 训练集必须包含全部三个类别")
                else:
                    # 验证集缺类别只作警告。本数据集 cola 只出现在 img01/img02、
                    # obstacle 只出现在 img03/img04，无法让三类同时进入验证集，
                    # 这是数据量的客观限制，不是标注错误。
                    warn(f"{split}: 缺少类别 {missing}（{names_missing}）"
                         f"—— 该类的指标将无法评估（本数据集为已知限制，见 report.md 2.4 节）")
            else:
                ok(f"{split}: 三个类别全部覆盖 {sorted(seen)}")

    # ---------------- 报告 ----------------
    print()
    print("=" * 78)
    print(f" 通过 {len(passed)} 项")
    for m in passed:
        print(f"   [OK]   {m}")
    if warnings:
        print(f"\n 警告 {len(warnings)} 项")
        for m in warnings:
            print(f"   [warn] {m}")
    if errors:
        print(f"\n 错误 {len(errors)} 项")
        for m in errors:
            print(f"   [FAIL] {m}")
    else:
        print("\n 未发现错误。")
    print("=" * 78)

    if errors:
        sys.exit(1)
    if warnings and args.strict:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
