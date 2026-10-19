# -*- coding: utf-8 -*-
"""环境与写权限自检（用 C:\\Users\\dell\\yolo-env 里的 python 运行）。"""
import sys
from pathlib import Path

print('PY        ', sys.version.split()[0])

try:
    import torch
    print('TORCH     ', torch.__version__)
    print('CUDA      ', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU       ', torch.cuda.get_device_name(0))
        print('VRAM_GB   ', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1))
except Exception as exc:
    print('TORCH_ERR ', exc)

try:
    import ultralytics
    print('ULTRALYTICS', ultralytics.__version__)
except Exception as exc:
    print('ULTRALYTICS_ERR', exc)

try:
    import cv2
    print('OPENCV    ', cv2.__version__)
except Exception as exc:
    print('OPENCV_ERR', exc)

# 测试能否把文件写进工作区里的 dataset/labels/train
proj = Path(__file__).resolve().parent
target = proj / 'dataset' / 'labels' / 'train' / '_write_test.txt'
try:
    target.write_text('ok', encoding='utf-8')
    print('WRITE_OK  ', target.exists(), target.stat().st_size)
    target.unlink()
    print('DELETE_OK ', not target.exists())
except Exception as exc:
    print('WRITE_ERR ', type(exc).__name__, exc)
