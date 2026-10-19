"""Filesystem locations. Override the root with MOTIONCOACH_ROOT (defaults to the repository root)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOTIONCOACH_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("MOTIONCOACH_DATA", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("MOTIONCOACH_MODELS", ROOT / "models"))
REFERENCES_DIR = Path(os.environ.get("MOTIONCOACH_REFERENCES", ROOT / "references"))
ASSETS_DIR = ROOT / "assets"

# Developer mode: exposes algorithm selection and the comparison page in the web UI.
# End users always run the default MediaPipe backend; comparisons are run by developers (CLI `motioncoach bench`).
DEV_MODE = os.environ.get("MOTIONCOACH_DEV", "0") == "1"
