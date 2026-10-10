from ultralytics import YOLO

# 加载自己训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 对图片进行推理 (可根据实际情况修改source路径)
model.predict(source='test.jpg', save=True)