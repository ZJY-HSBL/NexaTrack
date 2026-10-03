from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return inter / (union + eps)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2, eps=eps)
    lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    closure = wh[..., 0] * wh[..., 1]

    inter_lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = box_area(boxes1) + box_area(boxes2) - inter
    return iou - (closure - union) / (closure + eps)


def clip_boxes(boxes: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0) -> torch.Tensor:
    boxes = boxes.clamp(min=min_value, max=max_value)
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)
