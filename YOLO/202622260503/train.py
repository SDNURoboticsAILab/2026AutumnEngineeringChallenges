from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"data.yaml",
        epochs=5,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=1,
    )