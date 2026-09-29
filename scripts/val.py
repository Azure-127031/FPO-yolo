"""Evaluate a validation-selected FPO-YOLO checkpoint."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--project", default="runs/fpo_yolo_val")
    parser.add_argument("--name", default=None)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(args.weights))
    model.val(
        data=str(args.data),
        split=args.split,
        imgsz=640,
        batch=16,
        half=True,
        iou=0.7,
        max_det=300,
        augment=False,
        device=args.device,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
