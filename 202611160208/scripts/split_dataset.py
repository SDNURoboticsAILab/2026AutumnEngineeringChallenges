# -*- coding: utf-8 -*-
"""训练/验证集划分脚本（Level 1 步骤3）
将 obstacle / cola / football 原始图片复制到 YOLO 数据集结构：
  yolo_project/dataset/images/{train,val}/
  yolo_project/dataset/labels/{train,val}/   (标签 Phase 2 生成)

划分方式：每个类别内部独立按 8:2 随机划分（固定随机种子 42，可复现），
保证 train/val 中三类比例一致。
文件名加类别前缀避免跨类别重名冲突。
用法: python scripts/split_dataset.py
"""
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["obstacle", "cola", "football"]
SEED = 42
VAL_RATIO = 0.2
DST = os.path.join(ROOT, "yolo_project", "dataset")


def main():
    random.seed(SEED)
    # 创建目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", split), exist_ok=True)

    manifest_lines = ["image,split"]
    for cls in CLASSES:
        src = os.path.join(ROOT, cls)
        files = sorted(
            f for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
        )
        random.shuffle(files)
        n_val = int(round(len(files) * VAL_RATIO))
        splits = {"val": files[:n_val], "train": files[n_val:]}
        for split, fl in splits.items():
            for f in fl:
                # 统一命名: <类别>_<原名去扩展名>.jpg
                stem = os.path.splitext(f)[0]
                new_name = f"{cls}_{stem}.jpg"
                dst_img = os.path.join(DST, "images", split, new_name)
                if not os.path.exists(dst_img):
                    shutil.copy2(os.path.join(src, f), dst_img)
                manifest_lines.append(f"{new_name},{split}")
            print(f"{cls:<10} {split:<6} {len(fl):>4} 张")

    manifest = os.path.join(ROOT, "yolo_project", "split_manifest.csv")
    with open(manifest, "w", encoding="utf-8") as fp:
        fp.write("\n".join(manifest_lines))
    print(f"\n划分清单已保存: {manifest}")
    print("目录结构:")
    for split in ["train", "val"]:
        n = len(os.listdir(os.path.join(DST, "images", split)))
        print(f"  dataset/images/{split}: {n} 张")


if __name__ == "__main__":
    main()
