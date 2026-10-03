from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .attention import ContextAwareAttention


class ContextEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        inner_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ContextAwareAttention(d_model, num_heads, inner_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        key_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        y = self.norm1(x)
        qk = y + pos
        y = self.attn(qk, qk, y, key_weights=key_weights)
        x = x + self.drop(y)
        x = x + self.ffn(self.norm2(x))
        return x


class ContextEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        inner_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ContextEncoderLayer(d_model, num_heads, ffn_dim, inner_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, key_weights: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos, key_weights)
        return self.norm(x)


class DualBranchDecoderLayer(nn.Module):
    """Decoder layer without self-attention, using two context-aware cross-attention branches."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        inner_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm_a = nn.LayerNorm(d_model)
        self.memory_norm_b = nn.LayerNorm(d_model)
        self.branch_a = ContextAwareAttention(d_model, num_heads, inner_dim, dropout)
        self.branch_b = ContextAwareAttention(d_model, num_heads, inner_dim, dropout)
        self.merge = nn.Linear(d_model * 2, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory_encoded: torch.Tensor,
        memory_raw: torch.Tensor,
        memory_pos: torch.Tensor,
        key_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.query_norm(query)
        ma = self.memory_norm_a(memory_encoded)
        mb = self.memory_norm_b(memory_raw)

        a = self.branch_a(q, ma + memory_pos, ma, key_weights=key_weights)
        b = self.branch_b(q, mb + memory_pos, mb, key_weights=key_weights)
        fused = self.merge(torch.cat([a, b], dim=-1))
        query = query + self.drop(fused)
        query = query + self.ffn(self.norm2(query))
        return query


class DualBranchDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        inner_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DualBranchDecoderLayer(d_model, num_heads, ffn_dim, inner_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        memory_encoded: torch.Tensor,
        memory_raw: torch.Tensor,
        memory_pos: torch.Tensor,
        key_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, memory_encoded, memory_raw, memory_pos, key_weights)
        return self.norm(query)
