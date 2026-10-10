from ultralytics import YOLO

if __name__ == "__main__":
    # 加载 YOLOv8n 预训练模型
    model = YOLO("yolov8n.pt")

    # 开始训练
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,       # RTX 5070Ti GPU
        workers=0,
        project="runs",
        name="train_exp"
    )
