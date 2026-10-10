from ultralytics import YOLO

# 加载YOLOv8n预训练模型
model = YOLO("yolov8n.pt")

# 开始训练，使用你自己的数据集yaml配置
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="train-3"
)
