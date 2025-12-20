"""
Entry point for DanceBattle GUI application.

Usage:
    python main_gui.py
"""

from pathlib import Path
import sys

# Ensure local src layout is importable without installation.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dancebattle.gui import run_gui  # noqa: E402


if __name__ == "__main__":
    run_gui()
