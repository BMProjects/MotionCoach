"""GPU availability per runtime, checked against each vendor's documented requirements.

- MediaPipe Tasks (Python): GPU delegate via OpenGL ES on EGL; Google documents it for Ubuntu only.
  On glvnd systems with a desktop session, EGL may pick the Mesa vendor and fail
  ("eglChooseConfig() returned no matching EGL configuration"); pointing glvnd at the NVIDIA
  vendor file fixes it, so we set ``__EGL_VENDOR_LIBRARY_FILENAMES`` when it is unset.
- ONNX Runtime (rtmlib): ``onnxruntime-gpu`` >= 1.27 on PyPI is built for CUDA 13.0 + cuDNN 9.x;
  ``onnxruntime.preload_dlls()`` loads CUDA/cuDNN from the NVIDIA pip wheels (installed with torch cu13x).
- Ultralytics: PyTorch CUDA build (``torch.cuda.is_available()``).

``MOTIONCOACH_DEVICE=cpu`` forces CPU everywhere (reproducibility checks, machines without GPU).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

NVIDIA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")


def forced_cpu() -> bool:
    return os.environ.get("MOTIONCOACH_DEVICE", "auto").lower() == "cpu"


def prepare_mediapipe_gpu() -> None:
    """Must run before the first EGL use in this process."""
    if "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ and NVIDIA_EGL_VENDOR.exists():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(NVIDIA_EGL_VENDOR)


@functools.lru_cache(maxsize=1)
def ort_cuda() -> bool:
    if forced_cpu():
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            return False
    return True


@functools.lru_cache(maxsize=1)
def torch_cuda() -> bool:
    if forced_cpu():
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
