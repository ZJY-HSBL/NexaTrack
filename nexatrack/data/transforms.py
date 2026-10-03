from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class CropMeta:
    x0: float
    y0: float
    side: float
    output_size: int


def crop_target_square(
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray | list[float],
    area_scale: float,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, CropMeta]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    x1, y1, x2, y2 = bbox.tolist()
    w = max(x2 - x1, 2.0)
    h = max(y2 - y1, 2.0)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(math.sqrt(w * h * area_scale), max(w, h) * 1.05)
    x0, y0 = cx - side * 0.5, cy - side * 0.5
    scale = output_size / side

    matrix = np.array([
        [scale, 0.0, -x0 * scale],
        [0.0, scale, -y0 * scale],
    ], dtype=np.float32)
    border = tuple(float(v) for v in image_rgb.reshape(-1, 3).mean(axis=0))
    crop = cv2.warpAffine(
        image_rgb,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

    norm = np.array([
        (x1 - x0) / side,
        (y1 - y0) / side,
        (x2 - x0) / side,
        (y2 - y0) / side,
    ], dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    return crop, norm, CropMeta(x0=x0, y0=y0, side=side, output_size=output_size)


def normalized_box_to_image(box: np.ndarray | torch.Tensor, meta: CropMeta) -> np.ndarray:
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy()
    box = np.asarray(box, dtype=np.float32)
    x1 = meta.x0 + box[0] * meta.side
    y1 = meta.y0 + box[1] * meta.side
    x2 = meta.x0 + box[2] * meta.side
    y2 = meta.y0 + box[3] * meta.side
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
