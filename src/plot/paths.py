from __future__ import annotations

import os
from pathlib import Path

# .../src/plot/paths.py -> project root is two levels up from src/plot
PLOT_PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLOT_PKG_DIR.parents[1]


def data_root() -> Path:
    override = os.environ.get("PLMCALIPER_PLOT_DATA")
    if override:
        return Path(override)
    return PROJECT_ROOT / "data" / "plot_data"


def experiment_dir(name: str) -> Path:
    return data_root() / name
