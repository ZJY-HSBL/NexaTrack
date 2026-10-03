from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from nexatrack.config import load_config
from nexatrack.engine.metrics import center_error, iou_xyxy, precision_at_20, success_auc
from nexatrack.models import NexaTrack
from nexatrack.tracking import NexaTracker
from nexatrack.utils.checkpoint import load_checkpoint


def parse_gt(path: Path) -> list[np.ndarray]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.replace("\t", ",").replace(" ", ",").split(",") if v]
        if len(vals) >= 4:
            x, y, w, h = vals[:4]
            boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
    return boxes


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a folder of tracking sequences")
    p.add_argument("--config", default="configs/nexatrack.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True,
                   help="Root containing sequence/img/*.jpg and sequence/groundtruth.txt")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device_name = args.device or cfg.get("runtime", {}).get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = NexaTrack(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    all_ious: list[float] = []
    all_errors: list[float] = []
    root = Path(args.root)
    seq_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for seq in seq_dirs:
        gt_path = seq / "groundtruth.txt"
        img_dir = seq / "img"
        if not gt_path.exists() or not img_dir.exists():
            continue
        frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        frames = sorted(set(frames))
        gt = parse_gt(gt_path)
        if not frames or not gt:
            continue
        n = min(len(frames), len(gt))
        first = cv2.imread(str(frames[0]))
        tracker = NexaTracker(model, cfg, device)
        tracker.initialize(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), gt[0])
        seq_ious = [1.0]
        seq_errors = [0.0]
        for i in range(1, n):
            frame = cv2.imread(str(frames[i]))
            result = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seq_ious.append(iou_xyxy(result.bbox, gt[i]))
            seq_errors.append(center_error(result.bbox, gt[i]))
        all_ious.extend(seq_ious)
        all_errors.extend(seq_errors)
        print(f"{seq.name}: AUC={success_auc(seq_ious):.4f} P20={precision_at_20(seq_errors):.4f}")

    print(f"OVERALL AUC={success_auc(all_ious):.4f} P20={precision_at_20(all_errors):.4f}")


if __name__ == "__main__":
    main()
