"""Train FPO-YOLO with a configurable training batch size."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="s")
    parser.add_argument("--data", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/fpo_yolo")
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (reported VisDrone FPO n/s:12; UAVDT:16)",
    )
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    return args


def main():
    args = parse_args()
    model = YOLO(f"yolo11{args.scale}-fpo.yaml")
    model.train(
        data=str(args.data),
        epochs=200,
        patience=100,
        imgsz=640,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=False,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        amp=True,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name or f"fpo_yolo_{args.scale}_e200",
    )


if __name__ == "__main__":
    main()
