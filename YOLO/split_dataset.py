import os
import shutil
import random

categories = ["cola", "football", "obstacle"]
train_dir = "dataset/images/train"
val_dir = "dataset/images/val"
random.seed(42)

for category in categories:
    source_folder = os.path.join(".", category)
    images = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]
    random.shuffle(images)

    split_index = int(len(images) * 0.8)

    for i, image_name in enumerate(images):
        if i < split_index:
            dest_folder = train_dir
        else:
            dest_folder = val_dir

        new_name = f"{category}_{i+1:03d}.jpg"

        src_path = os.path.join(source_folder, image_name)
        dst_path = os.path.join(dest_folder, new_name)

        shutil.copy(src_path, dst_path)