from ultralytics import YOLO

# 加载 YOLO11n 预训练模型（会自动下载）
model = YOLO("yolo11n.pt")

# 开始训练
results = model.train(
    data="data.yaml",        # 指向刚才建的配置文件
    epochs=150,              # 训练150轮，小数据集足够收敛
    imgsz=640,               # 图片大小
    batch=8,                 # 每次喂给模型8张图（32G内存够用）
    patience=30,             # 如果30轮没提升就自动停止（防止浪费时间）
    device="cpu",            # 强制使用CPU训练（你的核显不适配PyTorch）
    optimizer="AdamW",       # 优化器，适合小数据集
    lr0=0.001,               # 初始学习率，小数据集要调低
    lrf=0.01,
    warmup_epochs=3.0,
    mosaic=0.5,              # 数据增强，防止过拟合
    mixup=0.0,
    degrees=5.0,
    fliplr=0.5,
    project="runs",          # 结果保存在 runs/train 里
    name="train",
    exist_ok=True,
)