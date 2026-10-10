from ultralytics import YOLO

# 加载预训练模型（第一次运行会自动下载）
model = YOLO("yolov8n.pt")

# 开始训练
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    project="runs",
    name="train",
)