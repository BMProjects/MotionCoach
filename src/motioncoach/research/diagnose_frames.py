"""Diagnostic panels for hard poses (docs/EXPERIMENT_PLAN.md E2): for chosen moments, the same frame three times -
person box (from the detector, flagged when a limb leaves it), the model's raw skeleton (joint colour =
confidence), and the cleaned skeleton (joints removed by the bone check in red, filled ones hollow). One look
tells whether an error comes from the box, the model, or the cleaning.

    python -m motioncoach.research.diagnose_frames OUT.png VIDEO t1,t2,... [--backend rtmw-l] [--label 第五式,...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .. import settings
from ..engine.filters import one_euro_clean, reject_long_bones
from ..media.extract import extract_poses, prepared_video
from ..perception.registry import get_backend

BODY = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
EDGES = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_hip", "left_knee"),
         ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"))


def _conf_colour(c: float) -> tuple[int, int, int]:
    c = float(np.clip(c, 0, 1))                 # BGR: red (low) -> yellow -> green (high)
    return (0, int(255 * min(1, 2 * c)), int(255 * min(1, 2 * (1 - c))))


def _draw(img, xy, idx, ok, colour, filled=None, removed=None, conf=None):
    for a, b in EDGES:
        ia, ib = idx[a], idx[b]
        if ok[ia] and ok[ib]:
            cv2.line(img, tuple(int(v) for v in xy[ia]), tuple(int(v) for v in xy[ib]), colour, 3, cv2.LINE_AA)
    for j in BODY:
        i = idx[j]
        if removed is not None and removed[i]:
            p = tuple(int(v) for v in xy[i])
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
        elif ok[i]:
            p = tuple(int(v) for v in xy[i])
            col = _conf_colour(conf[i]) if conf is not None else colour
            if filled is not None and filled[i]:
                cv2.circle(img, p, 7, col, 2, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 6, col, -1, cv2.LINE_AA)


def panel(video: Path, times: list[float], labels: list[str], backend: str, out: Path, crop_pad: float = 0.25) -> Path:
    be = get_backend(backend, settings.MODELS_DIR)
    vid, proxy = prepared_video(Path(video), settings.DATA_DIR)
    raw = extract_poses(proxy, vid, be, 10.0, settings.DATA_DIR)
    thr = float(raw.meta.get("conf_thr", 0.3))
    bone, _ = reject_long_bones(raw)
    clean = one_euro_clean(raw, conf_thr=thr)
    idx = {j: raw.skeleton.index[j] for j in BODY}
    det = None
    try:                                         # the detector's own person box (top-down models)
        be.open()
        det = getattr(be._model, "det_model", None)
    except Exception:
        det = None
    cap = cv2.VideoCapture(str(proxy))
    rows = []
    for t, lab in zip(times, labels):
        k = int(np.argmin(np.abs(raw.t - t)))
        cap.set(cv2.CAP_PROP_POS_MSEC, raw.t[k] * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        xy_r, cf = raw.xy[k], raw.conf[k]
        okr = np.isfinite(xy_r).all(axis=1) & (cf >= thr)
        removed = okr & ~np.isfinite(bone.xy[k]).all(axis=1)
        xy_c = clean.xy[k]
        okc = np.isfinite(xy_c).all(axis=1)
        fill = clean.meta["filled"][k]
        pts = xy_r[[idx[j] for j in BODY]]
        pts = pts[np.isfinite(pts).all(axis=1)]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        boxes = np.asarray(det(frame)).reshape(-1, 4) if det is not None else np.zeros((0, 4))
        box = None
        if len(boxes):                           # the detection that contains the skeleton's centre
            c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            inside = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes if b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]]
            if inside:
                box = boxes[[i for i, b in enumerate(boxes) if b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]][int(np.argmax(inside))]]
        w, h = x1 - x0, y1 - y0
        X0, Y0 = int(max(0, x0 - crop_pad * w)), int(max(0, y0 - crop_pad * h))
        X1, Y1 = int(min(frame.shape[1], x1 + crop_pad * w)), int(min(frame.shape[0], y1 + crop_pad * h))
        tiles = []
        for mode in ("box", "raw", "clean"):
            img = frame.copy()
            if mode == "box" and box is not None:
                cut = (x0 < box[0] - 2) or (x1 > box[2] + 2) or (y0 < box[1] - 2) or (y1 > box[3] + 2)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255) if cut else (255, 200, 0), 3)
                cv2.putText(img, "box cuts limbs" if cut else "box ok", (int(box[0]), max(20, int(box[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if cut else (255, 200, 0), 2)
            if mode == "raw":
                _draw(img, xy_r, idx, okr, (200, 200, 200), conf=cf)
            if mode == "clean":
                _draw(img, xy_c, idx, okc, (255, 120, 40), filled=fill, removed=removed)
            tile = cv2.resize(img[Y0:Y1, X0:X1], (300, int(300 * (Y1 - Y0) / max(1, X1 - X0))))
            tiles.append(tile)
        hmax = max(t_.shape[0] for t_ in tiles)
        tiles = [cv2.copyMakeBorder(t_, 0, hmax - t_.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t_ in tiles]
        row = np.hstack(tiles)
        bar = np.full((34, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{lab}  t={raw.t[k]:.1f}s   box | raw (colour = confidence) | cleaned (x removed, o filled)",
                    (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))
    cap.release()
    wmax = max(r.shape[1] for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(r, 0, 8, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("video")
    ap.add_argument("times")
    ap.add_argument("--labels", default="")
    ap.add_argument("--backend", default="rtmw-l")
    a = ap.parse_args(argv)
    times = [float(x) for x in a.times.split(",")]
    labels = a.labels.split(",") if a.labels else [""] * len(times)
    print(panel(Path(a.video), times, labels, a.backend, Path(a.out)))


if __name__ == "__main__":
    main()
