from __future__ import annotations

import argparse
from pathlib import Path

import env_setup  # noqa: F401
import yaml
from ultralytics import YOLO

import project_config as cfg


def build_data_yaml(data_root: Path, save_path: Path) -> Path:
    data = {
        "path": str(data_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lesion"},
    }
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
    parser.add_argument("--name", default=cfg.TRAIN_NAME)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"dataset path not found: {data_root}")

    data_yaml = build_data_yaml(data_root, root / "generated" / "skin_data_local.yaml")
    model_yaml = root / cfg.MODEL_YAML

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
    )


if __name__ == "__main__":
    main()
