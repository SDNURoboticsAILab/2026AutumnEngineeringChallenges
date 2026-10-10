from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,
        workers=8,
        cache=True,
        device=0,
        project='runs/train',
        name='exp',
        exist_ok=True,
        amp=False,
    )

    print("训练结束！请去 runs/train/exp 查看结果。")