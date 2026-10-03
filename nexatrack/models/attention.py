from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class ContextAwareAttention(nn.Module):
    """
    Multi-head attention with a nested correlation-refinement stage.

    The first correlation map is constructed from QK^T. Its key-wise correlation
    vectors are compressed into low-dimensional descriptors, refined by a second
    attention operation, expanded back to the original map size, and added as a
    residual before the final softmax. A continuous target mask can weight the
    nested value descriptors on template tokens while search tokens remain 1.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        inner_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.inner_dim = inner_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.inner_q = nn.Linear(inner_dim, inner_dim)
        self.inner_k = nn.Linear(inner_dim, inner_dim)
        self.inner_v = nn.Linear(inner_dim, inner_dim)
        self.inner_gate = nn.Parameter(torch.zeros(num_heads))
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _correlation_refinement(
        self,
        corr: torch.Tensor,
        key_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        b, h, nq, nk = corr.shape

        # Treat each key column as a correlation vector over all queries.
        columns = corr.transpose(-2, -1).contiguous()  # [B, H, Nk, Nq]
        desc = columns.view(b * h * nk, 1, nq)
        desc = F.interpolate(desc, size=self.inner_dim, mode="linear", align_corners=False)
        desc = desc.view(b, h, nk, self.inner_dim)
        desc = F.layer_norm(desc, (self.inner_dim,))

        q2 = self.inner_q(desc)
        k2 = self.inner_k(desc)
        v2 = self.inner_v(desc)

        if key_weights is not None:
            weights = key_weights[:, None, :, None].to(dtype=v2.dtype)
            v2 = v2 * weights

        inner_logits = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.inner_dim)
        inner_attn = inner_logits.softmax(dim=-1)
        inner_attn = self.dropout(inner_attn)
        refined = torch.matmul(inner_attn, v2)  # [B, H, Nk, inner_dim]

        refined = refined.view(b * h * nk, 1, self.inner_dim)
        refined = F.interpolate(refined, size=nq, mode="linear", align_corners=False)
        refined = refined.view(b, h, nk, nq).transpose(-2, -1).contiguous()

        gate = torch.tanh(self.inner_gate).view(1, h, 1, 1)
        return refined * (1.0 + gate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_weights: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        corr = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        residual = self._correlation_refinement(corr, key_weights)
        logits = corr + residual

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                logits = logits + attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3:
                logits = logits + attn_mask[:, None, :, :]
            else:
                logits = logits + attn_mask

        attn = logits.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.d_model)
        out = self.out_proj(out)
        if return_attention:
            return out, attn
        return out
