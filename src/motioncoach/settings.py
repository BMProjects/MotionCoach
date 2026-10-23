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

# Concurrent analysis jobs (the GPU server runs 2). Per-video pose extraction stays one process: decoding
# is already multi-threaded and was the bottleneck; chunked multi-process extraction measured no gain on the GPU server.
JOB_WORKERS = max(1, int(os.environ.get("MOTIONCOACH_JOB_WORKERS", "1")))
