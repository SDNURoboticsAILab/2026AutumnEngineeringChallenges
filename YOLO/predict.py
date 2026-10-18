from ultralytics import YOLO

# 1. 加载你刚才训练了100轮的模型权重 (注意路径变成了 train_v2)
model = YOLO("runs/detect/train_v2/weights/best.pt")

# 2. 指定你要测试的图片路径（去验证集里找一张）
test_image_path = "dataset/images/val/cola_235.jpg"

# 3. 开始预测
results = model.predict(
    source=test_image_path,  # 图片来源
    conf=0.25,               # 置信度阈值
    save=True,               # 保存画好框的图片
    project="results",       # 保存到 results 文件夹
    name="predict_test_v2"   # 任务名字改成 v2，避免覆盖之前的结果
)

print("预测完成！请去 results/predict_test_v2/ 文件夹里看结果图。")