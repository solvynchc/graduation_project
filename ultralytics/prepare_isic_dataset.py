from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    root_dir = Path(__file__).resolve().parent
    database_dir = root_dir.parent / "database"
    image_default = database_dir / "ISIC2018_Task1-2_Training_Input"
    mask_default = database_dir / "ISIC2018_Task1_Training_GroundTruth"
    output_default = database_dir / "YOLO_Dataset_Ready"

    parser = argparse.ArgumentParser(description="Convert ISIC masks to YOLO segmentation dataset.")
    parser.add_argument(
        "--image-dir",
        default=str(image_default),
        help=f"Original ISIC jpg image directory. Default: {image_default}",
    )
    parser.add_argument(
        "--mask-dir",
        default=str(mask_default),
        help=f"Original ISIC png mask directory. Default: {mask_default}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(output_default),
        help=f"YOLO dataset output directory. Default: {output_default}",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio, default: 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default: 42")
    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum contour area to keep, default: 100",
    )
    return parser.parse_args()


def ensure_dirs(output_dir: Path):
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def mask_to_yolo_lines(mask_path: Path, min_area: float) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue

        normalized = []
        for x, y in points:
            nx = min(max(float(x) / w, 0.0), 1.0)
            ny = min(max(float(y) / h, 0.0), 1.0)
            normalized.extend([nx, ny])

        lines.append("0 " + " ".join(f"{v:.6f}" for v in normalized))

    return lines


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask directory not found: {mask_dir}")

    ensure_dirs(output_dir)

    all_images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() == ".jpg"])
    random.seed(args.seed)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * args.train_ratio)
    train_names = {p.name for p in all_images[:split_idx]}

    print(
        f"total images: {len(all_images)} | "
        f"train: {len(train_names)} | "
        f"val: {len(all_images) - len(train_names)}"
    )

    converted = 0
    skipped = 0

    for img_path in tqdm(all_images, desc="Converting"):
        mask_name = f"{img_path.stem}_segmentation.png"
        mask_path = mask_dir / mask_name
        if not mask_path.exists():
            skipped += 1
            continue

        yolo_lines = mask_to_yolo_lines(mask_path, args.min_area)
        if not yolo_lines:
            skipped += 1
            continue

        split = "train" if img_path.name in train_names else "val"
        dst_img = output_dir / "images" / split / img_path.name
        dst_txt = output_dir / "labels" / split / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_txt.write_text("\n".join(yolo_lines), encoding="utf-8")
        converted += 1

    print(f"done: converted={converted}, skipped={skipped}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
