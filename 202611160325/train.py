from ultralytics import YOLO

if __name__ == '__main__':
    # 加载预训练的 YOLOv8n 模型
    model = YOLO('yolov8n.pt')
    # 使用 GPU 0 设备训练，设置为 100 个 epoch，图像大小 640
    model.train(data='data.yaml', epochs=100, imgsz=640, batch=8, device=0)