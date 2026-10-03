from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import CrossScaleSwinBackbone
from .head import CornerPredictionHead, ReliabilityHead
from .positional import SinePositionEncoding2D
from .transformer import ContextEncoder, DualBranchDecoder


def _as_pair(value: Sequence[int] | int) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


class NexaTrack(nn.Module):
    """Context-aware cross-scale Transformer tracker."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"] if "model" in cfg else cfg
        self.cfg = cfg
        self.d_model = int(m.get("d_model", 256))
        self.template_grid = _as_pair(m.get("token_grid_template", [12, 12]))
        self.search_grid = _as_pair(m.get("token_grid_search", [20, 20]))
        self.template_work_size = int(m.get("template_work_size", 192))
        self.search_work_size = int(m.get("search_work_size", 320))

        self.backbone = CrossScaleSwinBackbone(
            dims=m.get("backbone_dims", [96, 192, 384]),
            depths=m.get("backbone_depths", [2, 2, 6]),
            d_model=self.d_model,
            window_size=int(m.get("window_size", 7)),
        )
        self.pos2d = SinePositionEncoding2D(self.d_model)
        self.segment_embed = nn.Parameter(torch.zeros(3, 1, self.d_model))
        nn.init.trunc_normal_(self.segment_embed, std=0.02)

        self.encoder = ContextEncoder(
            num_layers=int(m.get("encoder_layers", 6)),
            d_model=self.d_model,
            num_heads=int(m.get("num_heads", 4)),
            ffn_dim=int(m.get("ffn_dim", 2048)),
            inner_dim=int(m.get("inner_dim", 64)),
            dropout=float(m.get("dropout", 0.1)),
        )
        self.decoder = DualBranchDecoder(
            num_layers=int(m.get("decoder_layers", 1)),
            d_model=self.d_model,
            num_heads=int(m.get("num_heads", 4)),
            ffn_dim=int(m.get("ffn_dim", 2048)),
            inner_dim=int(m.get("inner_dim", 64)),
            dropout=float(m.get("dropout", 0.1)),
        )
        self.target_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.target_query, std=0.02)

        hidden = int(m.get("score_hidden", self.d_model))
        self.corner_head = CornerPredictionHead(self.d_model, hidden=self.d_model)
        self.score_head = ReliabilityHead(self.d_model, hidden=hidden)
        self.sim_temperature = nn.Parameter(torch.tensor(0.07))

    def _extract(self, image: torch.Tensor, grid: tuple[int, int], work_size: int) -> torch.Tensor:
        if image.shape[-2:] != (work_size, work_size):
            image = F.interpolate(image, size=(work_size, work_size), mode="bilinear", align_corners=False)
        feat = self.backbone(image)
        feat = F.adaptive_avg_pool2d(feat, grid)
        return feat

    def _flatten_with_pos(self, feat: torch.Tensor, segment: int) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        pos = self.pos2d(h, w, device=feat.device, dtype=feat.dtype).expand(b, -1, -1)
        seg = self.segment_embed[segment].view(1, 1, -1)
        return tokens, pos + seg

    @staticmethod
    def gaussian_target_mask(
        boxes: torch.Tensor,
        grid: tuple[int, int],
        sigma: float = 0.05,
    ) -> torch.Tensor:
        """Build a normalized Gaussian mask around the target center on a feature grid."""
        b = boxes.shape[0]
        h, w = grid
        cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).clamp(0, 1)
        cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5).clamp(0, 1)
        xs = torch.linspace(0, 1, w, device=boxes.device, dtype=boxes.dtype)
        ys = torch.linspace(0, 1, h, device=boxes.device, dtype=boxes.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx = xx[None] - cx[:, None, None]
        dy = yy[None] - cy[:, None, None]
        mask = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma * sigma))
        mask = mask.flatten(1)
        # Keep tiny non-zero weights for numerical stability while preserving emphasis.
        return mask.clamp_min(1e-4)

    def forward(
        self,
        static_template: torch.Tensor,
        dynamic_template: torch.Tensor,
        search: torch.Tensor,
        static_bbox: torch.Tensor | None = None,
        dynamic_bbox: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = search.shape[0]
        device, dtype = search.device, search.dtype
        if static_bbox is None:
            static_bbox = torch.tensor([0.25, 0.25, 0.75, 0.75], device=device, dtype=dtype).repeat(b, 1)
        if dynamic_bbox is None:
            dynamic_bbox = static_bbox

        fs = self._extract(static_template, self.template_grid, self.template_work_size)
        fd = self._extract(dynamic_template, self.template_grid, self.template_work_size)
        fx = self._extract(search, self.search_grid, self.search_work_size)

        ts, ps = self._flatten_with_pos(fs, 0)
        td, pd = self._flatten_with_pos(fd, 1)
        tx, px = self._flatten_with_pos(fx, 2)

        raw = torch.cat([ts, td, tx], dim=1)
        pos = torch.cat([ps, pd, px], dim=1)

        ms = self.gaussian_target_mask(static_bbox, self.template_grid)
        md = self.gaussian_target_mask(dynamic_bbox, self.template_grid)
        mx = torch.ones(b, self.search_grid[0] * self.search_grid[1], device=device, dtype=dtype)
        key_weights = torch.cat([ms, md, mx], dim=1)

        encoded = self.encoder(raw, pos, key_weights)
        query = self.target_query.expand(b, -1, -1)
        query = self.decoder(query, encoded, raw, pos, key_weights)

        ns = self.search_grid[0] * self.search_grid[1]
        search_tokens = encoded[:, -ns:, :]
        q = F.normalize(query.squeeze(1), dim=-1)
        st = F.normalize(search_tokens, dim=-1)
        temperature = self.sim_temperature.abs().clamp_min(1e-3)
        similarity = torch.sigmoid(torch.einsum("bnd,bd->bn", st, q) / temperature)
        enhanced = search_tokens * (1.0 + similarity.unsqueeze(-1))
        search_map = enhanced.transpose(1, 2).reshape(
            b, self.d_model, self.search_grid[0], self.search_grid[1]
        )

        out = self.corner_head(search_map)
        out["score_logits"] = self.score_head(query)
        out["score"] = torch.sigmoid(out["score_logits"])
        out["similarity"] = similarity
        out["query"] = query.squeeze(1)
        out["encoded"] = encoded
        return out
