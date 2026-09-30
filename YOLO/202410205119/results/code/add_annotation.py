#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
add_annotation.py —— 把像素坐标换算成 YOLO 归一化坐标并写入标签文件
================================================================================
项目位置: 2026XXXXXX/add_annotation.py（辅助工具，非提交必需）

【作用】
输入"图片文件名 + 类别编号 + 像素框(x_min, y_min, x_max, y_max)"，
脚本用该图片的【真实分辨率】换算成 YOLO 归一化格式，直接覆盖写入对应的
 dataset/labels/<split>/<同名>.txt。

【为什么要有这个工具】
手工做 4 次除法很容易出错（这也是任务里 Make-Sense 导出 bug 后手工补标签的步骤）。
把换算交给脚本，可以保证：
  - 用的是图片真实宽高，而不是假设值；
  - 坐标一定落在 0~1 之间；
  - 同一个目标的 5 个字段顺序正确（class_id cx cy w h）。

【两种用法】
  1) 从清单文件批量标注（推荐）
     python add_annotation.py --from-file annotations.txt

     清单格式（每行一个目标，# 开头为注释）：
         <图片文件名> <类别编号> <x_min> <y_min> <x_max> <y_max>
     例：
         img01.jpg 0 150 184 291 416
         img01.jpg 1 101 228 241 388
     图片文件名只需 basename（img01.jpg）；脚本会自动在 train/val 里找它。

  2) 命令行单条追加
     python add_annotation.py --image img01.jpg --cls 1 --box 101 228 241 388 --append

     默认是覆盖该图片的标签；加 --append 则在已有内容后追加一个目标。

【安全措施】
  - 覆盖前自动把原标签备份为 <name>.txt.bak（已存在则不覆盖备份）；
  - 参数校验：类别编号必须是 0/1/2；坐标必须满足 x_min<x_max、y_min<y_max；
  - 越界检查：框超出图片范围会报警并跳过该条（可用 --allow-out-of-range 强制写入）；
  - 写入后立即回读校验，并打印换算明细，方便逐条核对。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 本文件已收纳到 results/code/ 下，因此项目根目录需要上溯两级：
# results/code/xxx.py -> results/code -> results -> 项目根（你的学号/）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_CLASSES = {0, 1, 2}
CLASS_NAME = {0: "obstacle", 1: "cola", 2: "football"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="像素坐标 -> YOLO 归一化坐标，并写入 dataset/labels/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--from-file", default="", help="标注清单文件路径（批量模式）")
    p.add_argument("--image", default="", help="图片文件名，如 img01.jpg（单条模式）")
    p.add_argument("--cls", type=int, default=None, help="类别编号 0/1/2（单条模式）")
    p.add_argument("--box", nargs=4, type=float, default=None,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="像素框（单条模式）")
    p.add_argument("--append", action="store_true",
                   help="追加到已有标签之后（默认覆盖该图片的标签）")
    p.add_argument("--allow-out-of-range", action="store_true",
                   help="允许框超出图片范围（默认跳过并警告）")
    p.add_argument("--dry-run", action="store_true", help="只打印换算结果，不写文件")
    p.add_argument("--no-backup", action="store_true", help="不备份原标签")
    return p.parse_args()


def find_image(name: str) -> tuple[Path, str] | None:
    """在 dataset/images/{train,val} 里找图片，返回 (图片路径, split)。"""
    stem = Path(name).stem
    for split in ("train", "val"):
        d = DATASET / "images" / split
        if not d.is_dir():
            continue
        for ext in IMAGE_EXTS:
            cand = d / f"{stem}{ext}"
            if cand.is_file():
                return cand, split
    return None


def parse_manifest(path: Path) -> list[tuple[str, int, float, float, float, float]]:
    """解析标注清单，返回 [(image_name, class_id, xmin, ymin, xmax, ymax), ...]。

    注意用 utf-8-sig 读取：Windows 上很多编辑器（含 PowerShell 的
    Set-Content -Encoding UTF8）会在文件开头写入 BOM。若用 utf-8 读，
    BOM 会变成看不见的 \\ufeff 粘在第一个文件名前面，
    导致"图片明明存在却提示找不到"。
    """
    rows = []
    for ln, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 6:
            print(f"[!] 清单第 {ln} 行字段数为 {len(parts)}，应为 6，已跳过: {raw.strip()!r}")
            continue
        name = parts[0]
        try:
            cid = int(parts[1])
            coords = [float(x) for x in parts[2:6]]
        except ValueError:
            print(f"[!] 清单第 {ln} 行含非数字，已跳过: {raw.strip()!r}")
            continue
        rows.append((name, cid, *coords))
    return rows


def write_labels(txt: Path, lines: list[str], dry_run: bool, no_backup: bool) -> None:
    """写入标签文件（覆盖）。写前备份，写后回读校验。"""
    if dry_run:
        print(f"      [dry-run] 将写入 {txt.name}:")
        for l in lines:
            print(f"          {l}")
        return

    # 备份原文件，避免误操作丢失已有标注
    if txt.is_file() and not no_backup:
        bak = txt.with_suffix(".txt.bak")
        if not bak.exists():
            shutil.copy2(txt, bak)
            print(f"      已备份原标签 -> {bak.name}")

    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 回读校验：确保真的写进去了（Make-Sense 那个 0 字节 bug 的教训）
    back = txt.read_text(encoding="utf-8").strip().splitlines()
    if len(back) != len(lines):
        print(f"      [X] 回读校验失败：期望 {len(lines)} 行，实际 {len(back)} 行")
        sys.exit(1)
    print(f"      ✅ 已写入 {txt.name}（{len(lines)} 个目标，回读校验通过）")


def main() -> None:
    args = parse_args()

    if not args.from_file and not (args.image and args.cls is not None and args.box):
        print(__doc__)
        print("[X] 请指定 --from-file，或同时指定 --image/--cls/--box")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("[X] 未安装 Pillow，请先执行: pip install pillow")
        sys.exit(1)

    # 收集待处理条目：(image_name, class_id, xmin, ymin, xmax, ymax)
    if args.from_file:
        mf = Path(args.from_file)
        if not mf.is_file():
            print(f"[X] 找不到清单文件: {mf}")
            sys.exit(1)
        entries = parse_manifest(mf)
        print(f"[i] 从清单读入 {len(entries)} 条标注：{mf}")
    else:
        entries = [(args.image, args.cls, *args.box)]

    if not entries:
        print("[X] 没有任何有效的标注条目。")
        sys.exit(1)

    print("=" * 78)
    print(" 像素坐标 -> YOLO 归一化坐标")
    print("=" * 78)

    # 按图片分组（同一张图的多个目标要写进同一个 txt）
    by_image: dict[str, list[tuple[int, float, float, float, float]]] = {}
    order: list[str] = []
    for name, cid, x1, y1, x2, y2 in entries:
        if cid not in VALID_CLASSES:
            print(f"[!] 跳过：类别编号 {cid} 非法（必须是 0/1/2），来自 {name}")
            continue
        if not (x1 < x2 and y1 < y2):
            print(f"[!] 跳过：{name} 的框不合法 (xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2})，"
                  f"必须满足 xmin<xmax 且 ymin<ymax")
            continue
        if name not in by_image:
            by_image[name] = []
            order.append(name)
        by_image[name].append((cid, x1, y1, x2, y2))

    ok_count = 0
    failed: list[str] = []          # 记录失败的图片，最后据此决定退出码
    for name in order:
        boxes = by_image[name]
        found = find_image(name)
        if not found:
            print(f"[X] 在 dataset/images/{{train,val}} 里找不到图片: {name}")
            print("    请先确认图片已放入，且文件名与清单一致（含扩展名）。")
            failed.append(name)
            continue
        img_path, split = found
        with Image.open(img_path) as im:
            W, H = im.size
        print(f"\n  {name}  (split={split}, 真实分辨率 {W} x {H})")

        lines: list[str] = []
        if args.append:
            txt = DATASET / "labels" / split / f"{Path(name).stem}.txt"
            if txt.is_file():
                lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]

        for cid, x1, y1, x2, y2 in boxes:
            # 越界检查：框必须落在图片内
            if not args.allow_out_of_range and (x1 < 0 or y1 < 0 or x2 > W or y2 > H):
                print(f"      [!] 跳过越界框 cls={cid} ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
                      f"，图片范围 0~{W} x 0~{H}。"
                      f"确认无误可加 --allow-out-of-range 强制写入。")
                continue

            # ---- 核心换算：像素 -> 归一化 ----
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            line = f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            lines.append(line)
            ok_count += 1
            print(f"      cls={cid}({CLASS_NAME[cid]})  像素 ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            print(f"          -> x_center=({x1:.0f}+{x2:.0f})/2/{W}={cx:.6f}  "
                  f"y_center=({y1:.0f}+{y2:.0f})/2/{H}={cy:.6f}")
            print(f"          -> width=({x2:.0f}-{x1:.0f})/{W}={bw:.6f}  "
                  f"height=({y2:.0f}-{y1:.0f})/{H}={bh:.6f}")
            print(f"          -> 写入行: {line}")

        if not lines:
            print("      [!] 该图片没有任何有效目标，未写入文件。")
            continue

        txt = DATASET / "labels" / split / f"{Path(name).stem}.txt"
        write_labels(txt, lines, args.dry_run, args.no_backup)

    print()
    print("=" * 78)
    print(f" 完成：成功换算 {ok_count} 个目标框"
          + ("（dry-run，未写文件）" if args.dry_run else ""))
    if failed:
        print(f" 失败：{len(failed)} 张图片未处理 -> {failed}")
    print("=" * 78)
    if failed:
        # 明确以非 0 退出，避免"部分成功"被误当成全部成功。
        # （本项目就吃过"看起来成功、其实内容为空"的亏，所以这里必须显式失败。）
        print(" [X] 存在未处理的图片，请修正清单或补齐图片后重跑。")
        sys.exit(1)
    print(" 下一步建议：")
    print("   python check_labels.py           # 校验图/标签配对与坐标合法性")
    print("   python train.py --check-only     # 确认配对数量为 6 后再训练")


if __name__ == "__main__":
    main()
