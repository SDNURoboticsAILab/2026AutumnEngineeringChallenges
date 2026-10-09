# -*- coding: utf-8 -*-
"""
train.py —— YOLO 模型训练入口
================================================================================
本文件是提交规范要求的训练脚本入口，实际训练流程位于 scripts/step3_train.py。

用法:
    python train.py                       # 使用默认参数(100 epochs, yolo11n)
    python train.py --epochs 150 --batch 16
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # 把命令行参数原样透传给真正的训练脚本
    target = os.path.join(ROOT, "scripts", "step3_train.py")
    sys.argv = [target] + sys.argv[1:]
    with open(target, "r", encoding="utf-8") as fh:
        code = compile(fh.read(), target, "exec")
    exec(code, {"__name__": "__main__", "__file__": target})
