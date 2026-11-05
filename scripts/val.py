"""Evaluate a selected checkpoint with explicit split, batch size, and precision."""

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size; AI-TOD-v2 uses 15")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                        help="Evaluation precision; AI-TOD-v2 uses fp32")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--project", default="runs/fpo_yolo_val")
    parser.add_argument("--name", default=None)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    return args


def main():
    args = parse_args()
    from ultralytics import YOLO

    # Load the selected checkpoint; the data YAML maps val/test to image folders.
    model = YOLO(str(args.weights))
    model.val(
        data=str(args.data),
        split=args.split,
        imgsz=640,
        batch=args.batch,
        half=args.precision == "fp16",
        # Keep low-score candidates for AP evaluation, unlike display filtering.
        conf=0.001,
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
