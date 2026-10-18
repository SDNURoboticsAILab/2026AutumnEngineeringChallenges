from ultralytics import YOLO

# 1. 加载预训练模型 (yolov8n.pt 是最小最快的模型)
model = YOLO("yolov8n.pt")

# 2. 开始训练
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    device="cpu",
    name="train_v2"
)