from ultralytics import YOLO

# 加载训练好的模型
model = YOLO(r"D:\deeplearn\ultralytics-8.3.163\yolo_project\xz_datasset\runs\detect\train6\weights\best.pt")

# 对整个文件夹做识别
# 默认保存到 runs/detect/predict/，不用手动 r.save()
results = model(
    r"D:\deeplearn\ultralytics-8.3.163\yolo_project\finaly\images\new train",
    save=True,
    project=r"D:\deeplearn\ultralytics-8.3.163\yolo_project\finaly\results",
    name="predict",
)
for r in results:
    print(r.path, r.boxes.cls.tolist(), r.boxes.conf.tolist())