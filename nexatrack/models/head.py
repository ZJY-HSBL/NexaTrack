from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CornerPredictionHead(nn.Module):
    """Predict top-left and bottom-right probability maps and decode by expectation."""

    def __init__(self, d_model: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(d_model, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 2, 1),
        )

    @staticmethod
    def _soft_argmax(prob_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, w = prob_logits.shape
        prob = F.softmax(prob_logits.flatten(1), dim=-1).view(b, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=prob.device, dtype=prob.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        x = (prob * xx).sum(dim=(1, 2))
        y = (prob * yy).sum(dim=(1, 2))
        return x, y, prob

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.fcn(feat)
        xtl, ytl, ptl = self._soft_argmax(logits[:, 0])
        xbr, ybr, pbr = self._soft_argmax(logits[:, 1])
        boxes = torch.stack([xtl, ytl, xbr, ybr], dim=-1)
        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return {
            "boxes": boxes,
            "corner_logits": logits,
            "prob_tl": ptl,
            "prob_br": pbr,
        }


class ReliabilityHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.net(query.squeeze(1)).squeeze(-1)
