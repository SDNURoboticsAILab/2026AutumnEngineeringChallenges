from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 启动训练 (与命令行执行的参数保持一致)
model.train(data='data.yaml', epochs=1, imgsz=320)