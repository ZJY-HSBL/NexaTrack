from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from nexatrack.data.transforms import (
    crop_target_square,
    image_to_tensor,
    normalized_box_to_image,
)


@dataclass
class TrackResult:
    bbox: np.ndarray
    score: float
    updated_template: bool


class NexaTracker:
    """Stateful single-object tracker around a trained NexaTrack model."""

    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any], device: torch.device) -> None:
        self.model = model.eval()
        self.cfg = cfg
        self.device = device
        tr = cfg.get("tracking", {})
        mo = cfg.get("model", {})
        self.template_size = int(mo.get("template_size", 180))
        self.search_size = int(mo.get("search_size", 320))
        self.template_area_scale = float(tr.get("template_area_scale", 22.0))
        self.search_area_scale = float(tr.get("search_area_scale", 52.0))
        self.update_threshold = float(tr.get("update_threshold", 0.65))
        self.update_interval = int(tr.get("update_interval", 10))
        self.frame_index = 0
        self.static_template: torch.Tensor | None = None
        self.dynamic_template: torch.Tensor | None = None
        self.static_bbox: torch.Tensor | None = None
        self.dynamic_bbox: torch.Tensor | None = None
        self.last_bbox: np.ndarray | None = None

    def initialize(self, frame_rgb: np.ndarray, bbox_xyxy: np.ndarray | list[float]) -> None:
        crop, norm_box, _ = crop_target_square(
            frame_rgb, bbox_xyxy, self.template_area_scale, self.template_size
        )
        tensor = image_to_tensor(crop).unsqueeze(0).to(self.device)
        box = torch.from_numpy(norm_box).unsqueeze(0).to(self.device)
        self.static_template = tensor
        self.dynamic_template = tensor.clone()
        self.static_bbox = box
        self.dynamic_bbox = box.clone()
        self.last_bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        self.frame_index = 0

    @torch.no_grad()
    def track(self, frame_rgb: np.ndarray) -> TrackResult:
        if self.last_bbox is None or self.static_template is None or self.dynamic_template is None:
            raise RuntimeError("Tracker is not initialized")

        search_crop, _, meta = crop_target_square(
            frame_rgb, self.last_bbox, self.search_area_scale, self.search_size
        )
        search_tensor = image_to_tensor(search_crop).unsqueeze(0).to(self.device)
        out = self.model(
            self.static_template,
            self.dynamic_template,
            search_tensor,
            self.static_bbox,
            self.dynamic_bbox,
        )
        pred_norm = out["boxes"][0]
        pred_img = normalized_box_to_image(pred_norm, meta)
        h, w = frame_rgb.shape[:2]
        pred_img[[0, 2]] = np.clip(pred_img[[0, 2]], 0, w - 1)
        pred_img[[1, 3]] = np.clip(pred_img[[1, 3]], 0, h - 1)
        score = float(out["score"][0].item())

        self.frame_index += 1
        updated = False
        if score >= self.update_threshold and self.frame_index % max(self.update_interval, 1) == 0:
            crop, norm_box, _ = crop_target_square(
                frame_rgb, pred_img, self.template_area_scale, self.template_size
            )
            self.dynamic_template = image_to_tensor(crop).unsqueeze(0).to(self.device)
            self.dynamic_bbox = torch.from_numpy(norm_box).unsqueeze(0).to(self.device)
            updated = True

        self.last_bbox = pred_img
        return TrackResult(bbox=pred_img, score=score, updated_template=updated)
