from __future__ import annotations

import argparse
from pathlib import Path

import env_setup  # noqa: F401
import yaml
from ultralytics import YOLO

import project_config as cfg


def build_data_yaml(
    data_root: Path,
    save_path: Path,
    raw_mask_dir: Path | None = None,
    raw_mask_zip: Path | None = None,
    raw_mask_zip_root: str = "ISIC2018_Task1_Training_GroundTruth",
    residual_band_kernel: int = 5,
) -> Path:
    data = {
        "path": str(data_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lesion"},
        "raw_mask_suffix": "_segmentation.png",
        "residual_band_kernel": int(residual_band_kernel),
    }

    if raw_mask_dir and raw_mask_dir.exists():
        data["raw_mask_dir"] = str(raw_mask_dir.resolve()).replace("\\", "/")
    elif raw_mask_zip and raw_mask_zip.exists():
        data["raw_mask_zip"] = str(raw_mask_zip.resolve()).replace("\\", "/")
        data["raw_mask_zip_root"] = raw_mask_zip_root

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(cfg.DATA_ROOT), help="YOLO_Dataset_Ready absolute path")
    parser.add_argument("--pretrained", default=cfg.PRETRAINED)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=cfg.IMGSZ)
    parser.add_argument("--batch", type=int, default=cfg.BATCH)
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--workers", type=int, default=cfg.WORKERS)
    parser.add_argument("--optimizer", default=cfg.OPTIMIZER)
    parser.add_argument("--lr0", type=float, default=cfg.LR0)
    parser.add_argument("--lrf", type=float, default=cfg.LRF)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--project", default=cfg.TRAIN_PROJECT)
    parser.add_argument("--name", default=cfg.TRAIN_NAME_RESIDUAL_BAND)
    parser.add_argument("--raw-mask-dir", default=str(cfg.RAW_MASK_DIR))
    parser.add_argument("--raw-mask-zip", default=str(cfg.RAW_MASK_ZIP))
    parser.add_argument("--raw-mask-zip-root", default="ISIC2018_Task1_Training_GroundTruth")
    parser.add_argument("--residual-band-kernel", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"dataset path not found: {data_root}")

    raw_mask_dir = Path(args.raw_mask_dir) if args.raw_mask_dir else None
    raw_mask_zip = Path(args.raw_mask_zip) if args.raw_mask_zip else None
    data_yaml = build_data_yaml(
        data_root,
        root / "generated" / "skin_data_residual_band.yaml",
        raw_mask_dir=raw_mask_dir,
        raw_mask_zip=raw_mask_zip,
        raw_mask_zip_root=args.raw_mask_zip_root,
        residual_band_kernel=args.residual_band_kernel,
    )
    model_yaml = root / cfg.MODEL_YAML_RESIDUAL_BAND

    print(f"data_yaml: {data_yaml}")
    print(f"model_yaml: {model_yaml}")

    model = YOLO(str(model_yaml))
    model.load(args.pretrained)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        deterministic=False,
        pretrained=args.pretrained,
        project=str((root / args.project).resolve()),
        name=args.name,
        mosaic=0.0,
        copy_paste=0.0,
        mixup=0.0,
        cutmix=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
    )


if __name__ == "__main__":
    main()
