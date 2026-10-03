from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from nexatrack.config import load_config
from nexatrack.data import TripletTrackingDataset
from nexatrack.engine import train_one_epoch
from nexatrack.models import NexaTrack
from nexatrack.utils.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NexaTrack on a JSONL triplet manifest")
    p.add_argument("--config", default="configs/nexatrack.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="runs/nexatrack")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["train"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = args.device or cfg.get("runtime", {}).get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = TripletTrackingDataset(
        args.manifest,
        template_size=int(cfg["model"].get("template_size", 180)),
        search_size=int(cfg["model"].get("search_size", 320)),
        template_area_scale=float(cfg["tracking"].get("template_area_scale", 22.0)),
        search_area_scale=float(cfg["tracking"].get("search_area_scale", 52.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = NexaTrack(cfg).to(device)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(cfg["train"].get("lr_backbone", 1e-5))},
            {"params": other_params, "lr": float(cfg["train"].get("lr_main", 1e-4))},
        ],
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(cfg["train"].get("lr_drop_epoch", 400))],
        gamma=float(cfg["train"].get("lr_drop_gamma", 0.1)),
    )
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    epochs = args.epochs or int(cfg["train"].get("epochs", 500))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, loader, optimizer, device, cfg, scaler)
        scheduler.step()
        print(f"epoch={epoch:03d} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint(
                output / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                extra={"config": cfg},
            )
            save_checkpoint(output / "latest.pt", model, epoch=epoch, extra={"config": cfg})


if __name__ == "__main__":
    main()
