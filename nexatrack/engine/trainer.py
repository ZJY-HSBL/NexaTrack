from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import tracking_loss


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    scaler: GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    train_cfg = cfg["train"]
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = scaler or GradScaler(enabled=use_amp)
    sums = defaultdict(float)
    steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            out = model(
                batch["static_template"],
                batch["dynamic_template"],
                batch["search"],
                batch["static_bbox"],
                batch["dynamic_bbox"],
            )
            losses = tracking_loss(
                out["boxes"],
                batch["search_bbox"],
                out["score_logits"],
                lambda_iou=float(train_cfg.get("lambda_iou", 2.0)),
                lambda_l1=float(train_cfg.get("lambda_l1", 5.0)),
            )

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        steps += 1
        for key, value in losses.items():
            sums[key] += float(value.detach().item())
        pbar.set_postfix(loss=f"{sums['loss'] / steps:.4f}")

    return {k: v / max(steps, 1) for k, v in sums.items()}
