from ultralytics import YOLO

# 加载自己训练的最优权重
model = YOLO("runs/detect/runs/train_exp/weights/best.pt")

# 用验证集里的一张图片测试（训练时没见过）
results = model(
    "dataset/images/val/image_rgb_20260727_094949.jpg",
    save=True,
    project="runs",
    name="predict_test"
)

# 打印检测结果
for r in results:
    print(f"\n检测到 {len(r.boxes)} 个目标:")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = r.names[cls_id]
        print(f"  {cls_name}: 置信度 {conf:.2f}")
