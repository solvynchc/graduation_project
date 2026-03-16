from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import env_setup  # noqa: F401
from ultralytics import YOLO

import project_config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation metrics for a YOLO segmentation model.")
    parser.add_argument("--model", default=str(cfg.FINAL_MODEL), help="Model weight path relative to project root.")
    parser.add_argument("--data-root", default=str(cfg.DATA_ROOT), help="YOLO_Dataset_Ready path.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=cfg.IMGSZ, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=cfg.CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=cfg.DEVICE, help="Inference device, e.g. 0 or cpu.")
    parser.add_argument(
        "--save-dir",
        default="metric_results",
        help="Directory under project root used to save csv/json summaries.",
    )
    return parser.parse_args()


def load_gt_mask(label_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            points = np.stack([coords[:, 0] * w, coords[:, 1] * h], axis=1)
            points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
    return mask


def build_pred_mask(result, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return mask

    pred_masks = result.masks.data.detach().cpu().numpy()
    for pred in pred_masks:
        resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.maximum(mask, (resized > 0.5).astype(np.uint8))
    return mask


def calc_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    eps = 1e-7

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
        "accuracy": float(accuracy),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split

    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label directory not found: {label_dir}")

    model = YOLO(str(root / args.model))
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    rows: list[dict[str, float | str]] = []

    for image_path in image_paths:
        gt_label = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=False,
            verbose=False,
        )
        pred_mask = build_pred_mask(results[0], (h, w))
        gt_mask = load_gt_mask(gt_label, (h, w))
        metrics = calc_metrics(pred_mask, gt_mask)
        metrics["image"] = image_path.name
        rows.append(metrics)

    if not rows:
        raise RuntimeError("no images were evaluated")

    df = pd.DataFrame(rows)
    summary = {
        "num_images": int(len(df)),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "iou": float(df["iou"].mean()),
        "dice": float(df["dice"].mean()),
        "accuracy": float(df["accuracy"].mean()),
    }

    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / f"{args.split}_seg_metrics_per_image.csv", index=False)
    with (save_dir / f"{args.split}_seg_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved per-image metrics to: {save_dir / f'{args.split}_seg_metrics_per_image.csv'}")
    print(f"saved summary to: {save_dir / f'{args.split}_seg_metrics_summary.json'}")


if __name__ == "__main__":
    main()
