# -*- coding: utf-8 -*-
"""
predict.py —— YOLO 推理测试与评估入口
================================================================================
本文件是提交规范要求的推理脚本入口，实际推理流程位于 scripts/step4_predict_evaluate.py。

用法:
    python predict.py                          # 对 test 测试集推理并评估
    python predict.py --conf 0.25 --imgsz 640
    python predict.py --source 某张新图片.jpg   # 对单张新图片推理
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    target = os.path.join(ROOT, "scripts", "step4_predict_evaluate.py")
    sys.argv = [target] + sys.argv[1:]
    with open(target, "r", encoding="utf-8") as fh:
        code = compile(fh.read(), target, "exec")
    exec(code, {"__name__": "__main__", "__file__": target})
