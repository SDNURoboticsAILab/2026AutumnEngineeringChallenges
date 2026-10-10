from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt

# 修复字体报错：强制指定系统自带英文字体，避开损坏的字体缓存
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 加载你训练好的最佳权重，路径和你报错里的保持一致
model = YOLO(r"C:\YOLO-homework\2026AutumnEngineeringChallenges-main\YOLO\yolo-project\runs\detect\train-4\weights\best.pt")

# 运行验证，自动生成所有指标图、混淆矩阵、批次预测图
results = model.val(
    data=r"C:\YOLO-homework\2026AutumnEngineeringChallenges-main\YOLO\yolo-project\data.yaml",
    plots=False,   # 开启绘图
    save_json=True
)

# 终端打印核心指标，方便直接记录
print("=== 验证核心指标 ===")
print(f"平均Precision(P): {results.box.mp:.4f}")
print(f"平均Recall(R): {results.box.mr:.4f}")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
