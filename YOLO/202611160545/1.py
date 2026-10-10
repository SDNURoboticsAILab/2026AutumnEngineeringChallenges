from ultralytics import YOLO

def main():
    model = YOLO(r"D:\YOLO-Challenge\yolo11n.pt")
    model.train(
        data=r"D:\YOLO-Challenge\数据集配置.yaml",
        epochs=60,
        imgsz=640,
        batch=8,
        device=0,
        project=r"D:\YOLO-Challenge\runs",
        name="train",
        exist_ok=True,
    )
    print("训练结束")

if __name__ == "__main__":
    main()
