from __future__ import annotations

import torch
import torch.nn.functional as F

from nexatrack.utils.box_ops import generalized_box_iou


def tracking_loss(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    score_logits: torch.Tensor | None = None,
    lambda_iou: float = 2.0,
    lambda_l1: float = 5.0,
) -> dict[str, torch.Tensor]:
    l1 = F.l1_loss(pred_boxes, gt_boxes)
    giou = generalized_box_iou(pred_boxes, gt_boxes)
    iou_loss = (1.0 - giou).mean()
    total = lambda_iou * iou_loss + lambda_l1 * l1

    score_loss = pred_boxes.new_tensor(0.0)
    if score_logits is not None:
        with torch.no_grad():
            target = giou.detach().clamp(0.0, 1.0)
        score_loss = F.binary_cross_entropy_with_logits(score_logits, target)
        total = total + 0.5 * score_loss

    return {
        "loss": total,
        "loss_iou": iou_loss,
        "loss_l1": l1,
        "loss_score": score_loss,
        "mean_giou": giou.mean(),
    }
