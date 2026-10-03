from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from nexatrack.config import load_config
from nexatrack.models import NexaTrack
from nexatrack.tracking import NexaTracker
from nexatrack.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NexaTrack on a video")
    p.add_argument("--config", default="configs/nexatrack.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="runs/demo.mp4")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device_name = args.device or cfg.get("runtime", {}).get("device", "cuda")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = NexaTrack(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=True)
    tracker = NexaTracker(model, cfg, device)

    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read video: {args.video}")
    init = cv2.selectROI("NexaTrack - select target", first, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("NexaTrack - select target")
    x, y, w, h = init
    if w <= 0 or h <= 0:
        raise RuntimeError("No initial ROI selected")
    bbox = [x, y, x + w, y + h]
    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    tracker.initialize(first_rgb, bbox)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )

    frame = first
    current_bbox = bbox
    score = 1.0
    while True:
        x1, y1, x2, y2 = map(int, current_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"score {score:.3f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)
        cv2.imshow("NexaTrack", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = tracker.track(rgb)
        current_bbox = result.bbox
        score = result.score

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
