"""从 runs/yolo11n_3classes/weights/last.pt 断点续训。

用法（先切到本项目目录）：

    python resume_train.py

训练被中断时（例如电脑休眠、误关终端）用它可以接上剩余的 epoch。
ultralytics 会从 last.pt 里恢复优化器状态、数据集参数和 epochs，
并在跑完后照常生成 results.png、混淆矩阵、P/R 曲线等收尾图表。
"""
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    last = root / "runs" / "yolo11n_3classes" / "weights" / "last.pt"
    if not last.exists():
        raise FileNotFoundError(f"未找到断点权重：{last}")
    print(f"从断点续训：{last}")
    model = YOLO(str(last))
    model.train(resume=True)


if __name__ == "__main__":
    main()
