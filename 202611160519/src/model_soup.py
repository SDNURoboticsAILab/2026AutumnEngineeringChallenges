# model_soup.py
# 模型加权融合（Model Soup）
import shutil
from pathlib import Path

import torch


def soup_models(model_paths, weights=None, output_path="soup.pt"):
    """
    多个 YOLO 模型加权融合。
    - model_paths: 权重文件路径列表
    - weights: 每个模型的权重（None 表示等权）
    - output_path: 输出路径
    """
    if not model_paths:
        raise ValueError("模型列表不能为空")
    if len(model_paths) == 1:
        shutil.copy2(model_paths[0], output_path)
        return output_path

    if weights is None:
        weights = [1.0] * len(model_paths)
    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("权重和为 0")
    weights = [w / total_w for w in weights]

    # 加载骨架
    base_ckpt = torch.load(model_paths[0], map_location="cpu", weights_only=False)
    base_state = base_ckpt["model"].state_dict()

    # 累加
    avg_state = {k: torch.zeros_like(v, dtype=torch.float32)
                 for k, v in base_state.items()}

    for path, w in zip(model_paths, weights):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model"].state_dict()
        for k in avg_state:
            if k in state:
                avg_state[k] += w * state[k].float()

    # 写回
    for k in base_state:
        base_state[k] = avg_state[k].to(base_state[k].dtype)

    base_ckpt["model"].load_state_dict(base_state)
    torch.save(base_ckpt, output_path)
    return output_path