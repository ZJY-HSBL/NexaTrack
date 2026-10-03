from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Partition NHWC tensor into windows, padding spatial dimensions if needed."""
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows, (h, w, hp, wp)


def _window_reverse(windows: torch.Tensor, window_size: int, meta: tuple[int, int, int, int], batch: int) -> torch.Tensor:
    h, w, hp, wp = meta
    c = windows.shape[-1]
    x = windows.view(batch, hp // window_size, wp // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, hp, wp, c)
    return x[:, :h, :w, :]


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x * rand / keep


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        index = relative.sum(-1)
        self.register_buffer("relative_position_index", index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        bias = bias.permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if attn_mask is not None:
            nw = attn_mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def _build_mask(self, hp: int, wp: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, hp, wp, 1), device=device, dtype=dtype)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hs in h_slices:
            for wslice in w_slices:
                img_mask[:, hs, wslice, :] = cnt
                cnt += 1
        mask_windows, _ = _window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows, meta = _window_partition(x, self.window_size)
        _, _, hp, wp = meta
        attn_mask = self._build_mask(hp, wp, x.device, x.dtype)
        windows = self.attn(windows, attn_mask)
        x = _window_reverse(windows, self.window_size, meta, b)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, drop_path: Sequence[float]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                drop_path=drop_path[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CrossScaleSwinBackbone(nn.Module):
    """Three-stage shifted-window backbone with stage-2/stage-3 cross-scale fusion."""

    def __init__(
        self,
        dims: Sequence[int] = (96, 192, 384),
        depths: Sequence[int] = (2, 2, 6),
        d_model: int = 256,
        window_size: int = 7,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dims) != 3 or len(depths) != 3:
            raise ValueError("dims and depths must contain exactly three stages")
        if dims[1] != 2 * dims[0] or dims[2] != 2 * dims[1]:
            raise ValueError("backbone dims must double at each stage")

        self.patch_embed = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)
        self.patch_norm = nn.LayerNorm(dims[0])

        total = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total).tolist()
        p = 0
        heads = [max(1, dims[0] // 32), max(1, dims[1] // 32), max(1, dims[2] // 32)]
        self.stage1 = SwinStage(dims[0], depths[0], heads[0], window_size, dpr[p:p + depths[0]])
        p += depths[0]
        self.merge1 = PatchMerging(dims[0])
        self.stage2 = SwinStage(dims[1], depths[1], heads[1], window_size, dpr[p:p + depths[1]])
        p += depths[1]
        self.merge2 = PatchMerging(dims[1])
        self.stage3 = SwinStage(dims[2], depths[2], heads[2], window_size, dpr[p:p + depths[2]])

        self.fuse = nn.Sequential(
            nn.Conv2d(dims[1] + dims[2], d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).permute(0, 2, 3, 1).contiguous()
        x = self.patch_norm(x)
        x = self.stage1(x)
        x = self.merge1(x)
        f2 = self.stage2(x)
        x = self.merge2(f2)
        f3 = self.stage3(x)

        f2_nchw = f2.permute(0, 3, 1, 2).contiguous()
        f3_nchw = f3.permute(0, 3, 1, 2).contiguous()
        f3_up = F.interpolate(f3_nchw, size=f2_nchw.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([f2_nchw, f3_up], dim=1))
