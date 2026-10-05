# -*- coding: utf-8 -*-
"""数据体检脚本（Level 1 步骤2）
检查 obstacle / cola / football 三个原始数据文件夹：
- 图片数量与格式分布
- 分辨率分布
- 无法正常读取的损坏图片
用法: python scripts/check_data.py
"""
import os
from collections import Counter
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["obstacle", "cola", "football"]


def main():
    print(f"{'类别':<10}{'数量':>6}{'损坏':>6}  格式分布 / 分辨率范围")
    for cls in CLASSES:
        folder = os.path.join(ROOT, cls)
        files = sorted(os.listdir(folder))
        bad = []
        exts = Counter()
        sizes = []
        for f in files:
            path = os.path.join(folder, f)
            if not os.path.isfile(path):
                continue
            exts[os.path.splitext(f)[1].lower()] += 1
            try:
                with Image.open(path) as im:
                    im.verify()  # 校验文件完整性
                with Image.open(path) as im:
                    sizes.append(im.size)  # (宽, 高)
            except Exception as e:
                bad.append((f, str(e)))

        if sizes:
            ws, hs = zip(*sizes)
            res_info = f"宽 {min(ws)}-{max(ws)}, 高 {min(hs)}-{max(hs)}"
        else:
            res_info = "无有效图片"
        print(f"{cls:<10}{len(sizes):>6}{len(bad):>6}  {dict(exts)} | {res_info}")
        for name, err in bad:
            print(f"  [损坏] {name}: {err}")

    # 统计唯一文件名（跨类别重名时后续需要加前缀区分）
    names = set()
    dup_cross = 0
    for cls in CLASSES:
        folder = os.path.join(ROOT, cls)
        for f in os.listdir(folder):
            if f in names:
                dup_cross += 1
            names.add(f)
    print(f"\n跨类别重名文件数: {dup_cross}（>0 则划分时需为文件名加类别前缀）")


if __name__ == "__main__":
    main()
