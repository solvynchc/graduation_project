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
    parser.add_argument("--model", default=str(cfg.FINAL_MODEL))
    parser.add_argument("--split", default="val")
    parser.add_argument("--workers", type=int, default=0, help="Validation dataloader workers on local Windows.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"dataset path not found: {data_root}")

    model_path = root / args.model
    data_yaml = build_data_yaml(data_root, root / "generated" / "skin_data_local.yaml")

    print(f"model: {model_path}")
    print(f"data_yaml: {data_yaml}")
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split=args.split, workers=args.workers)
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
