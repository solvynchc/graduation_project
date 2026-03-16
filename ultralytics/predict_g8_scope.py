from __future__ import annotations

import argparse
from pathlib import Path

import env_setup  # noqa: F401
from ultralytics import YOLO

import project_config as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(cfg.PREDICT_SOURCE), help="image file or folder")
    parser.add_argument("--model", default=str(cfg.FINAL_MODEL))
    parser.add_argument("--project", default=cfg.PREDICT_PROJECT)
    parser.add_argument("--name", default=cfg.PREDICT_NAME)
    parser.add_argument("--conf", type=float, default=cfg.CONF)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    model = YOLO(str(root / args.model))
    model.predict(
        source=str(source),
        conf=args.conf,
        save=True,
        project=str((root / args.project).resolve()),
        name=args.name,
    )
    print(f"saved to: {root / args.project / args.name}")


if __name__ == "__main__":
    main()
