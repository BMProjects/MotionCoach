"""Run SAM 3D Body on sampled frames with given person boxes; export plain arrays (runs in the
sam-3d-body environment, not the app's).

    python sam3db_export.py BOXES.npz OUT.npz --repo <sam-3d-body checkout> --checkpoint DIR/model.ckpt
        [--mhr DIR/assets/mhr_model.pt] [--inference-type body]

``BOXES.npz`` comes from ``offline_trial --dump-boxes`` (t, boxes xyxy per 10 fps frame, video path), so the
model sees the same crops as the other candidates. No detector, segmentor or FOV estimator is loaded (the
model's default camera is used), which keeps detectron2 and MoGe out of the environment.

Writes ``t``, ``kp2d`` (T, 70, 3: MHR70 2D keypoints, confidence 1 where a box existed), ``kp3d``
(T, 70, 3, camera metres), ``valid`` (T,) and ``ms_per_frame`` (docs/EXPERIMENT_PLAN.md E4).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("boxes")
    ap.add_argument("out")
    ap.add_argument("--repo", required=True, help="sam-3d-body checkout")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mhr", default="")
    ap.add_argument("--inference-type", default="body", choices=("body", "full"))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.repo).expanduser()))
    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.build_models import load_sam_3d_body

    ckpt = Path(a.checkpoint).expanduser()
    mhr = Path(a.mhr).expanduser() if a.mhr else ckpt.parent / "assets" / "mhr_model.pt"
    model, cfg = load_sam_3d_body(str(ckpt), device="cuda", mhr_path=str(mhr))
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    b = np.load(a.boxes)
    ts, boxes, video = b["t"], b["boxes"], str(b["video"])
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    want = {int(round(t * fps)): i for i, t in enumerate(ts)}
    kp2d = np.full((len(ts), 70, 3), np.nan, np.float32)
    kp3d = np.full((len(ts), 70, 3), np.nan, np.float32)
    valid = np.zeros(len(ts), bool)
    n, spent, f = 0, 0.0, 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        i = want.pop(f, None)
        f += 1
        if i is None or not np.isfinite(boxes[i]).all():
            continue
        s = time.perf_counter()
        out = est.process_one_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bboxes=boxes[i][None].astype(np.float32),
                                    inference_type=a.inference_type)
        spent += time.perf_counter() - s
        n += 1
        if out:
            kp2d[i, :, :2] = out[0]["pred_keypoints_2d"][:, :2]
            kp2d[i, :, 2] = 1.0
            kp3d[i] = out[0]["pred_keypoints_3d"]
            valid[i] = True
        if n % 500 == 0:
            print(f"{n} frames, {1000 * spent / n:.0f} ms/frame", flush=True)
    cap.release()
    np.savez_compressed(a.out, t=ts, kp2d=kp2d, kp3d=kp3d, valid=valid.astype(np.float32),
                        ms_per_frame=1000 * spent / max(1, n))
    print(a.out, f"{valid.mean():.3f} valid", f"{1000 * spent / max(1, n):.0f} ms/frame")


if __name__ == "__main__":
    main()
