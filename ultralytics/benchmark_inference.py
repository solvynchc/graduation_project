from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

import env_setup  # noqa: F401
from ultralytics import YOLO

import project_config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO segmentation inference speed.")
    parser.add_argument("--model", default=str(cfg.FINAL_MODEL), help="Model weight path relative to project root.")
    parser.add_argument("--source", default=str(cfg.DATA_ROOT / "images" / "val"), help="Image file or folder.")
    parser.add_argument("--imgsz", type=int, default=cfg.IMGSZ, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=cfg.CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=cfg.DEVICE, help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup image count.")
    parser.add_argument(
        "--save-file",
        default="metric_results/inference_benchmark.json",
        help="Result json path relative to project root.",
    )
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted([p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def run_once(model: YOLO, image_path: Path, imgsz: int, conf: float, iou: float, device: str):
    model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=False,
        verbose=False,
    )


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    images = iter_images(source)
    if not images:
        raise RuntimeError("no images found for benchmarking")

    model = YOLO(str(root / args.model))

    for image_path in images[: min(args.warmup, len(images))]:
        run_once(model, image_path, args.imgsz, args.conf, args.iou, args.device)

    if torch.cuda.is_available() and str(args.device).lower() != "cpu":
        torch.cuda.synchronize()
    start = time.time()

    for image_path in images:
        run_once(model, image_path, args.imgsz, args.conf, args.iou, args.device)

    if torch.cuda.is_available() and str(args.device).lower() != "cpu":
        torch.cuda.synchronize()
    end = time.time()

    total_images = len(images)
    total_seconds = end - start
    avg_ms = total_seconds / total_images * 1000.0
    fps = total_images / total_seconds if total_seconds > 0 else 0.0

    result = {
        "num_images": total_images,
        "total_seconds": total_seconds,
        "avg_inference_ms": avg_ms,
        "fps": fps,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    save_path = root / args.save_file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved benchmark to: {save_path}")


if __name__ == "__main__":
    main()
