from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml', epochs=50, imgsz=320, batch=4, device='cpu')