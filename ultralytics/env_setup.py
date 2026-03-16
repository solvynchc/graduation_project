from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def resolve_config_dir() -> Path:
    configured = os.environ.get("YOLO_CONFIG_DIR")
    if configured:
        return Path(configured)
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / ".yolo_config"
    return ROOT_DIR / ".yolo_config"


LOCAL_CONFIG_DIR = resolve_config_dir()
LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_CONFIG_DIR)
