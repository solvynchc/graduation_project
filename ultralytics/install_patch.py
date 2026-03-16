from __future__ import annotations

import shutil
from pathlib import Path

import env_setup  # noqa: F401
import ultralytics


def main():
    root = Path(__file__).resolve().parent
    patch_root = root / "ultralytics"
    target_root = Path(ultralytics.__file__).resolve().parent

    files = [
        ("nn/tasks.py", "nn/tasks.py"),
        ("nn/modules/block.py", "nn/modules/block.py"),
        ("nn/modules/head.py", "nn/modules/head.py"),
        ("nn/modules/dysample.py", "nn/modules/dysample.py"),
        ("nn/modules/__init__.py", "nn/modules/__init__.py"),
        ("data/augment.py", "data/augment.py"),
        ("data/build.py", "data/build.py"),
        ("data/dataset.py", "data/dataset.py"),
        ("utils/loss.py", "utils/loss.py"),
    ]

    for src_rel, dst_rel in files:
        src = patch_root / src_rel
        dst = target_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"patched: {dst}")

    print("patch complete")


if __name__ == "__main__":
    main()
