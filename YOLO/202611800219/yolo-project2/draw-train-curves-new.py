import csv
import matplotlib
matplotlib.use('Agg')
# 不调用系统字体，避开Arial字体报错！！！
import matplotlib.pyplot as plt

csv_path = r"runs\detect\train-6\results.csv"

epochs = []
train_loss_box = []
val_loss_box = []
p = []
r = []

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epo = int(row['epoch'])
        epochs.append(epo)
        train_loss_box.append(float(row['train/box_loss']))
        val_loss_box.append(float(row['val/box_loss']))
        p.append(float(row['metrics/precision(B)']))
        r.append(float(row['metrics/recall(B)']))


fig, axes = plt.subplots(2,1, figsize=(10, 8))

axes[0].plot(epochs, train_loss_box, label='train box loss')
axes[0].plot(epochs, val_loss_box, label='val box loss')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs, p, label='Precision')
axes[1].plot(epochs, r, label='Recall')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Value")
axes[1].legend()

plt.tight_layout()
plt.savefig("new_train_curves.png", dpi=150)
plt.close()
print("已生成 new_train_curves.png！这个是新版正确曲线，可以用于报告")
