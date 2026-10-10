from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/train-3/weights/best.pt")

# 推理测试图片
results = model.predict(source="test_img", save=True)

# 打印检测结果
for res in results:
    print(res.boxes)
