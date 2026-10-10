from ultralytics import YOLO

if __name__ == "__main__":
    # 加载你自己训练出来的模型
    model = YOLO("runs/detect/runs/train/exp/weights/best.pt")

    # 对 test_images 里的新图片进行推理
    results = model.predict(
        source="test_images",
        conf=0.25,
        iou=0.45,
        save=True,
        project="results",
        name="predict",
        exist_ok=True
    )

    for r in results:
        print(r.path)
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(model.names[cls_id], conf)