"""Check the minimum packages needed by the WallTopo-UQ experiments."""

from __future__ import annotations

import importlib
import sys


REQUIRED = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "cv2",
    "skimage",
    "yaml",
    "tqdm",
    "seaborn",
    "pytest",
]


def main() -> None:
    print("python", sys.version)
    missing: list[str] = []
    for module in REQUIRED:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "unknown")
            print(f"OK {module}: {version}")
        except Exception as error:  # noqa: BLE001 - diagnostic script
            print(f"MISSING {module}: {error}")
            missing.append(module)
    if missing:
        print("missing:", ", ".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
