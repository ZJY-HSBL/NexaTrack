from __future__ import annotations

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_error(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5])
    cb = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])
    return float(np.linalg.norm(ca - cb))


def success_auc(ious: list[float], thresholds: int = 101) -> float:
    th = np.linspace(0.0, 1.0, thresholds)
    curve = np.array([(np.asarray(ious) >= t).mean() for t in th], dtype=np.float64)
    return float(curve.mean())


def precision_at_20(errors: list[float]) -> float:
    return float((np.asarray(errors) <= 20.0).mean())
