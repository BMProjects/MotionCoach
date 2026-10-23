"""E4 first trial: ViTPose++ against RTMW-l on the re-capture video's sections (docs/EXPERIMENT_PLAN.md E4).

Same frames (the sampling grid), same person boxes (rtmlib's YOLOX detector, as the RTM models use), the same
13 body joints and the same cleaning; only the keypoint model differs. ViTPose++ comes from Hugging Face
Transformers, installed outside the service environment:

    PYTHONPATH=<dir with transformers> python -m motioncoach.research.vitpose_trial OUT_DIR '<gt json>' RECAPTURE_VIDEO

This is a trial on the development video pair, not the acceptance test (that needs the E3 validation set).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import settings
from ..apps.assets import get_reference, load_profile
from ..engine.normalize import bone_lengths
from ..media.extract import extract_poses, prepared_video
from ..perception.base import select_target
from ..perception.registry import get_backend
from .cleaning_ablation import section_scores, stages

MODEL = "usyd-community/vitpose-plus-base"


class VitPoseBackend:
    """Top-down: YOLOX person boxes -> ViTPose++ (COCO head, dataset index 0) -> one target person."""
    skeleton_name = "coco17"
    conf_threshold = 0.3

    def __init__(self, model: str = MODEL, device: str = "cuda"):
        self.name, self.model_name, self.device = "vitpose++", model, device
        self.model_id = f"hf/{model}/{device}"
        self._det = self._proc = self._net = None
        self._prev_box = None

    def open(self) -> None:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        if self._net is None:
            rtm = get_backend("rtmpose-m", settings.MODELS_DIR)         # its YOLOX person detector
            rtm.open()
            self._det = rtm._model.det_model
            self._proc = AutoProcessor.from_pretrained(self.model_name)
            self._net = VitPoseForPoseEstimation.from_pretrained(self.model_name).to(self.device).eval()
            self._torch = torch
        self._prev_box = None

    def estimate(self, frame_bgr: np.ndarray, ts_ms: int):
        boxes = np.asarray(self._det(frame_bgr), dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return None, None, None
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        coco = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        inputs = self._proc(rgb, boxes=[coco.tolist()], return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            out = self._net(**inputs, dataset_index=self._torch.zeros(len(coco), dtype=self._torch.long, device=self.device))
        res = self._proc.post_process_pose_estimation(out, boxes=[coco.tolist()])[0]
        kpts = np.stack([r["keypoints"].cpu().numpy() for r in res]).astype(np.float32)
        scores = np.stack([r["scores"].cpu().numpy() for r in res]).astype(np.float32)
        i, self._prev_box = select_target(kpts, scores, self._prev_box, self.conf_threshold)
        if i < 0:
            return None, None, None
        return kpts[i], np.clip(scores[i], 0, 1), None

    def close(self) -> None:
        self._prev_box = None


def section_errors(backend, gt: dict, recapture: Path) -> dict:
    ref = get_reference("baduanjin_demo_a", settings.REFERENCES_DIR, settings.ROOT)
    prof = load_profile(ref.raw["profile"])
    names, segs = prof.features, ref.segments(prof)
    tol = np.array([prof.tolerance(n)["good"] for n in names])
    rid, rpx = prepared_video(ref.video, settings.DATA_DIR)
    uid, upx = prepared_video(recapture, settings.DATA_DIR)
    R, U = extract_poses(rpx, rid, backend, 10.0, settings.DATA_DIR), extract_poses(upx, uid, backend, 10.0, settings.DATA_DIR)
    SR, SU = stages(R), stages(U)
    out = {"ms_per_frame": [R.meta.get("ms_per_frame"), U.meta.get("ms_per_frame")]}
    rb, ub = bone_lengths(SR["full"]), bone_lengths(SU["full"])
    for st in ("raw", "conf", "full"):
        out[st] = section_scores(SR[st], SU[st], rb, ub, gt, segs, names, tol)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gt")
    ap.add_argument("recapture")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = json.loads(a.gt)
    res = {"vitpose++": section_errors(VitPoseBackend(), gt, Path(a.recapture)),
           "rtmw-l": section_errors(get_backend("rtmw-l", settings.MODELS_DIR), gt, Path(a.recapture))}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for m, r in res.items():
        print(m, r["ms_per_frame"], {s: (r["full"][s]["joint_med"], r["full"][s]["joint_p90"], r["full"][s]["feat_med"]) for s in ("s4", "s5", "s6", "s7")})


if __name__ == "__main__":
    main()
