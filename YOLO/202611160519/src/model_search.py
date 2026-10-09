# model_search.py
# 智能融合权重搜索（贝叶斯优化）
import tempfile
from dataclasses import dataclass
from pathlib import Path

from model_soup import soup_models


@dataclass
class Candidate:
    path: str
    score: float
    class_scores: dict


def evaluate_fused(model_paths, weights, gt_data, detector_cls,
                   iou_thr=0.5, area_weight=True, required_enforce=True):
    """融合 + 评估，返回平均 F1"""
    from core import evaluate_one
    from PIL import Image

    tmp_file = Path(tempfile.gettempdir()) / "soup_search_tmp.pt"
    try:
        soup_models(model_paths, weights=weights, output_path=str(tmp_file))
        detector = detector_cls(str(tmp_file))
        total_f1 = 0.0
        n = 0
        for key, rec in gt_data:
            img_path = Path(rec.get("_img_dir", "")) / rec["file"]
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    img_size = im.size
            except Exception:
                continue
            try:
                preds = detector.predict(str(img_path), conf=0.3)
            except Exception:
                continue
            m = evaluate_one(rec["boxes"], preds, iou_thr=iou_thr,
                             img_size=img_size, use_area_weight=area_weight,
                             enforce_required=required_enforce)
            total_f1 += m["f1"]
            n += 1
        return total_f1 / n if n > 0 else 0.0
    finally:
        try: tmp_file.unlink()
        except Exception: pass


def smart_soup(candidates, gt_data, detector_cls,
               n_calls=30, n_initial=10, top_k=5, verbose_cb=None):
    """
    智能融合：先验初始化 + 贝叶斯优化搜索权重。
    返回: (best_model_paths, best_weights, best_score)
    """
    def log(msg):
        if verbose_cb:
            try: verbose_cb(msg)
            except Exception: pass

    if not candidates:
        raise ValueError("候选模型为空")

    # 1. 筛选 top-K
    sorted_cand = sorted(candidates, key=lambda c: -c.score)
    top = sorted_cand[:min(top_k, len(sorted_cand))]
    log(f"[搜索] 保留 top-{len(top)} 候选:")
    for c in top:
        log(f"    {Path(c.path).name}  F1={c.score:.3f}")

    if len(top) == 1:
        return [top[0].path], [1.0], top[0].score

    # 2. 先验权重（Softmax over scores）
    import math
    scores = [c.score for c in top]
    exp_s = [math.exp(s * 5.0) for s in scores]
    sum_e = sum(exp_s)
    prior = [e / sum_e for e in exp_s]
    log(f"[搜索] 先验权重: {[round(w, 3) for w in prior]}")

    # 3. 贝叶斯优化
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        use_bayes = True
    except ImportError:
        log("[搜索] 未安装 scikit-optimize，退回随机搜索")
        use_bayes = False

    paths = [c.path for c in top]
    n_models = len(paths)
    cache = {}

    def objective(weights_raw):
        s = sum(weights_raw)
        if s <= 0:
            return 1.0
        w = [x / s for x in weights_raw]
        key = tuple(round(x, 3) for x in w)
        if key in cache:
            return cache[key]
        try:
            f1 = evaluate_fused(paths, w, gt_data, detector_cls)
        except Exception as e:
            log(f"[评估失败] {e}")
            f1 = 0.0
        cache[key] = -f1
        log(f"    评估 w={[round(x, 3) for x in w]}  F1={f1:.4f}")
        return -f1

    if use_bayes:
        log(f"[搜索] 贝叶斯优化，{n_calls} 次采样")
        try:
            result = gp_minimize(
                objective,
                dimensions=[Real(0.0, 1.0) for _ in range(n_models)],
                x0=[prior],
                n_calls=n_calls,
                n_initial_points=n_initial,
                random_state=42,
            )
            best_raw = result.x
            best_f1 = -result.fun
        except Exception as e:
            log(f"[搜索] 贝叶斯失败: {e}，退回随机搜索")
            use_bayes = False

    if not use_bayes:
        import random
        random.seed(42)
        log(f"[搜索] 随机搜索，{n_calls} 次采样")
        best_f1 = -1
        best_raw = None
        for i in range(n_calls):
            if i == 0:
                w = prior
            else:
                w = [random.random() for _ in range(n_models)]
            f1 = -objective(w)
            if f1 > best_f1:
                best_f1 = f1
                best_raw = w

    s = sum(best_raw)
    best_w = [x / s for x in best_raw]
    log(f"[搜索] 最优权重: {[round(x, 3) for x in best_w]}")
    log(f"[搜索] 最优 F1: {best_f1:.4f}")

    return paths, best_w, best_f1