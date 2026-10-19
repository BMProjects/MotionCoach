"""Registry of pose backends.

Production default is ``mediapipe-full`` only (``DEFAULT_BACKEND``); every other entry exists for
developer-side comparison (``motioncoach bench``) and is hidden from the end-user UI.
``status="candidate"`` marks researched 3D methods that are catalogued with their requirements
but not runnable until their gated weights / code are installed.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BACKEND = "mediapipe-full"
LOCATE_BACKEND = "mediapipe-lite"


@dataclass(frozen=True)
class BackendSpec:
    name: str
    family: str
    label: str
    keypoints: int
    requires: tuple[str, ...]
    license: str
    note: str = ""
    dims: str = "2D"                 # 2D | 2D+3D (what the backend outputs)
    gpu: str = ""                    # documented GPU condition
    status: str = "available"        # available | candidate
    links: tuple[str, ...] = field(default=())


_MP_GPU = "MediaPipe GPU 委托（OpenGL ES/EGL）；官方仅声明 Ubuntu，本机 Debian 需指向 NVIDIA EGL vendor（自动设置）"
_ORT_GPU = "onnxruntime-gpu CUDA 执行器；≥1.27 的 PyPI 包基于 CUDA 13.0 + cuDNN 9（随 torch cu13x 的 NVIDIA 轮子提供）"
_TORCH_GPU = "PyTorch CUDA 版本（torch.cuda.is_available()）"

SPECS: dict[str, BackendSpec] = {s.name: s for s in [
    BackendSpec("mediapipe-lite", "mediapipe", "MediaPipe Pose Landmarker lite", 33, ("mediapipe",), "Apache-2.0",
                "区间定位粗扫用", dims="2D+3D", gpu=_MP_GPU),
    BackendSpec("mediapipe-full", "mediapipe", "MediaPipe Pose Landmarker full", 33, ("mediapipe",), "Apache-2.0",
                "默认算法；含足跟/脚尖；另输出髋部为原点的 3D 估计（米）", dims="2D+3D", gpu=_MP_GPU),
    BackendSpec("mediapipe-heavy", "mediapipe", "MediaPipe Pose Landmarker heavy", 33, ("mediapipe",), "Apache-2.0",
                dims="2D+3D", gpu=_MP_GPU),
    BackendSpec("rtmpose-m", "rtmlib", "RTMPose-m + YOLOX-m (COCO-17)", 17, ("rtmlib", "onnxruntime"), "Apache-2.0",
                "无足部关键点", gpu=_ORT_GPU),
    BackendSpec("rtmpose-x", "rtmlib", "RTMPose-x 384×288 + YOLOX-x", 17, ("rtmlib", "onnxruntime"), "Apache-2.0",
                "无足部关键点", gpu=_ORT_GPU),
    BackendSpec("rtmo-m", "rtmlib", "RTMO-m 单阶段多人", 17, ("rtmlib", "onnxruntime"), "Apache-2.0", "无足部关键点", gpu=_ORT_GPU),
    BackendSpec("rtmw-l", "rtmlib", "RTMW-l 全身 133 点", 133, ("rtmlib", "onnxruntime"), "Apache-2.0", "含足、手", gpu=_ORT_GPU),
    BackendSpec("rtmw-x", "rtmlib", "RTMW-x 全身 133 点", 133, ("rtmlib", "onnxruntime"), "Apache-2.0", "含足、手", gpu=_ORT_GPU),
    BackendSpec("rtmw3d-x", "rtmlib", "RTMW3D-x 全身 133 点 + 3D", 133, ("rtmlib", "onnxruntime"), "Apache-2.0",
                "3D 为裁剪框像素 x/y + 相对深度 z，非米制；暂不参与评分", dims="2D+3D", gpu=_ORT_GPU),
    BackendSpec("yolo26n-pose", "ultralytics", "YOLO26n-pose", 17, ("ultralytics",), "AGPL-3.0", "商用需授权", gpu=_TORCH_GPU),
    BackendSpec("yolo26m-pose", "ultralytics", "YOLO26m-pose", 17, ("ultralytics",), "AGPL-3.0", "商用需授权", gpu=_TORCH_GPU),
    BackendSpec("yolo11m-pose", "ultralytics", "YOLO11m-pose（旧版本对照）", 17, ("ultralytics",), "AGPL-3.0", "商用需授权", gpu=_TORCH_GPU),
    # ---- researched 3D methods: catalogued, not yet runnable --------------------------------------
    BackendSpec("gvhmr", "external", "GVHMR 世界坐标人体运动恢复（视频级，SMPL）", 24, (), "研究用途；SMPL/SMPL-X 需注册",
                "按视频片段整体推理（需目标跟踪 + ViTPose 2D + 时序模型），输出重力对齐的世界坐标 SMPL 关节；"
                "适合转体、前屈等正面 2D 不可测的指标", dims="3D", status="candidate",
                gpu="PyTorch CUDA；官方安装为 Python 3.10 conda 环境，可选 DPVO 需 CUDA 12.1 编译（官方不建议追求速度时使用）",
                links=("https://github.com/zju3dv/GVHMR", "https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md")),
    BackendSpec("sam3d-body", "external", "SAM 3D Body 单图全身网格恢复（MHR）", 0, (), "SAM License；权重需在 Hugging Face 申请",
                "单张图像推理，可接受 2D 关键点/掩码提示；视频需逐帧运行并另做时序平滑", dims="3D", status="candidate",
                gpu="PyTorch CUDA；官方 INSTALL 为 Python 3.11 + Detectron2，未给出显存要求（主干为 ViT-H / DINOv3，4 GB 笔记本显卡未验证）",
                links=("https://github.com/facebookresearch/sam-3d-body",
                       "https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md")),
]}

_CANDIDATE_REQUIREMENTS = {
    "gvhmr": ("GVHMR_ROOT", "GVHMR 代码库与 inputs/checkpoints（gvhmr_siga24_release.ckpt、vitpose、hmr2、yolo）及 "
                            "body_models/smpl(x)（需在 SMPL/SMPL-X 官网注册下载）"),
    "sam3d-body": ("SAM3D_BODY_ROOT", "sam-3d-body 代码库、Detectron2，以及经 Hugging Face 批准下载的 "
                                      "facebook/sam-3d-body-dinov3 或 -vith 权重"),
}


def availability(name: str) -> tuple[bool, str]:
    spec = SPECS[name]
    if spec.status == "candidate":
        env, what = _CANDIDATE_REQUIREMENTS[name]
        root = os.environ.get(env)
        if not root or not Path(root).exists():
            return False, f"候选技术，尚未接入：需要 {what}，并设置环境变量 {env}"
        return False, f"已检测到 {env}，但适配器尚未实现（按 docs/ARCHITECTURE.md 的片段级接口接入）"
    missing = [m for m in spec.requires if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"缺少依赖 {', '.join(missing)}（uv sync --extra {'rtm' if spec.family == 'rtmlib' else 'yolo'}）"
    return True, ""


def list_backends() -> list[dict]:
    out = []
    for s in SPECS.values():
        ok, why = availability(s.name)
        out.append({**s.__dict__, "requires": list(s.requires), "links": list(s.links), "available": ok,
                    "reason": why, "default": s.name == DEFAULT_BACKEND})
    return out


def get_backend(name: str, models_dir: Path, device: str | None = None):
    if name not in SPECS:
        raise ValueError(f"Unknown pose backend '{name}'. Known: {', '.join(SPECS)}")
    ok, why = availability(name)
    if not ok:
        raise RuntimeError(f"Backend {name} unavailable: {why}")
    fam = SPECS[name].family
    if fam == "mediapipe":
        from .mediapipe_tasks import MediaPipeTasksBackend

        return MediaPipeTasksBackend(name.split("-", 1)[1], models_dir, device=device)
    if fam == "rtmlib":
        from .rtm import RtmlibBackend

        return RtmlibBackend(name, device=device)
    from .yolo import UltralyticsBackend

    return UltralyticsBackend(name, f"{name}.pt", models_dir, device)
