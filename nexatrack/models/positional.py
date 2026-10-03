from __future__ import annotations

import math
import torch
from torch import nn


class SinePositionEncoding2D(nn.Module):
    """DETR-style 2D sine/cosine positional encoding."""

    def __init__(self, d_model: int = 256, temperature: float = 10000.0) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sine encoding")
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, h: int, w: int, device=None, dtype=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        y, x = torch.meshgrid(y, x, indexing="ij")
        if h > 1:
            y = y / (h - 1) * 2 * math.pi
        if w > 1:
            x = x / (w - 1) * 2 * math.pi

        dim = self.d_model // 4
        omega = torch.arange(dim, device=device, dtype=dtype)
        omega = self.temperature ** (2 * torch.div(omega, 2, rounding_mode="floor") / dim)

        px = x[..., None] / omega
        py = y[..., None] / omega
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1).flatten(-2)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((py, px), dim=-1)
        if pos.shape[-1] < self.d_model:
            pad = self.d_model - pos.shape[-1]
            pos = torch.nn.functional.pad(pos, (0, pad))
        return pos.reshape(1, h * w, self.d_model)
