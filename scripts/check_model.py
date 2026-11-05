"""Build an FPO-YOLO model and print its summary."""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="s")
    args = parser.parse_args()
    from ultralytics import YOLO

    # Report the requested architecture at the same 640-pixel input size.
    model = YOLO(f"yolo11{args.scale}-fpo.yaml")
    model.info(detailed=False, verbose=True, imgsz=640)


if __name__ == "__main__":
    main()
