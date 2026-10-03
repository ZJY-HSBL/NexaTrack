from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import crop_target_square, image_to_tensor, read_rgb


class TripletTrackingDataset(Dataset):
    """
    Generic JSONL triplet dataset.

    Each line contains:
    {
      "static_template": "path/to/frame_a.jpg",
      "dynamic_template": "path/to/frame_b.jpg",
      "search": "path/to/frame_c.jpg",
      "static_bbox": [x1,y1,x2,y2],
      "dynamic_bbox": [x1,y1,x2,y2],
      "search_bbox": [x1,y1,x2,y2]
    }
    """

    def __init__(
        self,
        manifest: str | Path,
        template_size: int = 180,
        search_size: int = 320,
        template_area_scale: float = 22.0,
        search_area_scale: float = 52.0,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_scale = template_area_scale
        self.search_area_scale = search_area_scale
        self.records: list[dict[str, Any]] = []
        with self.manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if not self.records:
            raise ValueError(f"No samples in {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, p: str) -> str:
        path = Path(p)
        if not path.is_absolute():
            path = self.root / path
        return str(path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self.records[index]
        s_img = read_rgb(self._resolve(rec["static_template"]))
        d_img = read_rgb(self._resolve(rec["dynamic_template"]))
        x_img = read_rgb(self._resolve(rec["search"]))

        s_crop, s_box, _ = crop_target_square(
            s_img, rec["static_bbox"], self.template_area_scale, self.template_size
        )
        d_crop, d_box, _ = crop_target_square(
            d_img, rec["dynamic_bbox"], self.template_area_scale, self.template_size
        )
        x_crop, x_box, _ = crop_target_square(
            x_img, rec["search_bbox"], self.search_area_scale, self.search_size
        )

        return {
            "static_template": image_to_tensor(s_crop),
            "dynamic_template": image_to_tensor(d_crop),
            "search": image_to_tensor(x_crop),
            "static_bbox": torch.from_numpy(s_box),
            "dynamic_bbox": torch.from_numpy(d_box),
            "search_bbox": torch.from_numpy(x_box),
        }
