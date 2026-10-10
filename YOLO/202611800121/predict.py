from ultralytics import YOLO

# 加载你自己训练的模型（路径要和你复制过来的路径完全一致）
model = YOLO("runs/detect/runs/train/weights/best.pt")

# 对新图片进行检测
results = model.predict(
    source="test_images/",
    save=True,          # 保存画了框的图片
    conf=0.25,          # 置信度阈值，低于0.25的不要
    imgsz=640,
)

# 在终端打印检测结果
for r in results:
    print(f"图片: {r.path}")
    for box in r.boxes:
        cls = int(box.cls[0])          # 类别编号
        conf = float(box.conf[0])      # 置信度
        name = model.names[cls]        # 类别名称
        print(f"  检测到: {name}, 置信度: {conf:.2f}")