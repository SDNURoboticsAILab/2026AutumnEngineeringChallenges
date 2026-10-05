# -*- coding: utf-8 -*-
"""提示词实验：对抽样图片用不同文本提示跑 YOLO-World，输出可视化图供人工评估"""
import os
import random
import sys

from ultralytics import YOLOWorld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "yolo_project", "dataset")
OUT = os.path.join(ROOT, "yolo_project", "prompt_test")
os.makedirs(OUT, exist_ok=True)

# 候选提示词方案
PROMPTS = {
    "A": ["chair", "desk", "table", "stool", "bottle", "soccer ball"],
    "B": ["wooden chair", "stacked chairs", "table", "cola bottle", "soccer ball", "orange ball"],
}


def main():
    random.seed(0)
    model = YOLOWorld(os.path.join(ROOT, "downloads", "yolov8s-worldv2.pt"))

    samples = []
    for cls in ["obstacle", "cola", "football"]:
        files = [f for f in os.listdir(os.path.join(DATASET, "images", "train"))
                 if f.startswith(cls)]
        samples += random.sample(files, 4)

    for pid, classes in PROMPTS.items():
        model.set_classes(classes)
        for f in samples:
            img = os.path.join(DATASET, "images", "train", f)
            r = model.predict(img, conf=0.2, verbose=False)[0]
            stem = os.path.splitext(f)[0]
            r.save(os.path.join(OUT, f"{pid}_{stem}.jpg"))
            det = [f"{classes[int(c)]}x{conf:.2f}" for c, conf in
                   zip(r.boxes.cls, r.boxes.conf)]
            print(f"[{pid}] {stem[:34]:<34} -> {len(det)}: {det}")


if __name__ == "__main__":
    main()
