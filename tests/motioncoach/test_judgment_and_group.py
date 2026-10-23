"""Measurement-uncertainty judgment, practitioner period, and multi-person identities."""

import numpy as np

from motioncoach.apps.assets import load_profile
from motioncoach.apps.rules import evaluate
from motioncoach.engine.group import GroupParams, build_identities
from motioncoach.engine.skeleton import COCO17
from motioncoach.engine.templates import practice_period
from motioncoach.perception.multi import APP_DIM, MultiPoses


def _cmp(seg_id: str, knee: float) -> dict:
    cell = {"coverage": 1.0, "mean_delta": knee, "median_abs": knee, "p90_abs": knee, "hold_coverage": 1.0,
            "hold_valid": 6, "hold_n": 6, "hold_mean_delta": knee, "hold_median_abs": knee,
            "per_rep": [knee] * 6, "ref_mean": 170.0, "user_mean": 170.0 + knee}
    return {"segments": [{"id": seg_id, "name": seg_id, "ref_start": 0.0, "ref_end": 10.0, "align_cost": 0.5,
                          "hold_times": [5.0], "features": {"l_knee": cell}}]}


def test_marginal_deviation_is_imprecise_only_where_measurement_is_noisy():
    prof = load_profile("baduanjin")
    # 16 deg vs a 15 deg tolerance: beyond it by more than the noise in an easy section ...
    easy = evaluate(_cmp("s1", 16.0), prof, backend="rtmw-l")["segments"][0]["features"]["l_knee"]
    assert easy["status"] == "warn" and easy["margin"] < 1.0
    # ... but within the measurement error in the hard section (deep squat with trunk swing)
    hard = evaluate(_cmp("s5", 16.0), prof, backend="rtmw-l")["segments"][0]["features"]["l_knee"]
    assert hard["status"] == "imprecise" and "text" not in hard and "测量误差" in hard["reason"]
    # a clear deviation stays a deviation; a less precise model widens the margin
    clear = evaluate(_cmp("s5", 40.0), prof, backend="mediapipe-full")["segments"][0]["features"]["l_knee"]
    assert clear["status"] == "bad" and clear["margin"] > hard["margin"]


def test_practice_period_prefers_one_repetition_over_a_left_right_cycle():
    names = ["l_elbow", "r_elbow"]
    t = np.arange(0, 60, 0.1)
    left = 0.5 + 0.5 * np.sin(2 * np.pi * t / 16) > 0.5          # left and right repetitions alternate
    bump = np.sin(np.pi * (t % 8) / 8) ** 2 * 60                  # one repetition = 8 s
    F = np.stack([150 - bump * left, 150 - bump * ~left], axis=1)
    p = practice_period(F, names, np.array([15.0, 15.0]), 37, 200, prior=75)
    assert p is not None and abs(p - 80) <= 5
    assert practice_period(F[:150], names, np.array([15.0, 15.0]), 37, 200, prior=75) is None   # < 3 periods


def _person(cx: float, foot: float, height: float, phase: float, t: float, moving: bool) -> np.ndarray:
    """COCO-17 keypoints of a front-facing figure; arms swing when ``moving``."""
    ix = COCO17.index
    s = height / 8.0
    raise_ = (0.5 + 0.5 * np.sin(2 * np.pi * (t - phase) / 6.0)) if moving else 0.0
    pts = {"nose": (0, 7.5), "left_eye": (0.2, 7.6), "right_eye": (-0.2, 7.6), "left_ear": (0.4, 7.5),
           "right_ear": (-0.4, 7.5), "left_shoulder": (1, 6.5), "right_shoulder": (-1, 6.5),
           "left_elbow": (1.2 + raise_, 5.2 + 1.5 * raise_), "right_elbow": (-1.2 - raise_, 5.2 + 1.5 * raise_),
           "left_wrist": (1.3 + 2 * raise_, 4.0 + 3 * raise_), "right_wrist": (-1.3 - 2 * raise_, 4.0 + 3 * raise_),
           "left_hip": (0.7, 4), "right_hip": (-0.7, 4), "left_knee": (0.8, 2), "right_knee": (-0.8, 2),
           "left_ankle": (0.8, 0), "right_ankle": (-0.8, 0)}
    out = np.zeros((len(COCO17.joints), 2))
    for n, (x, y) in pts.items():
        out[ix[n]] = (cx + x * s, foot - y * s)
    return out


def _scene(T=400, fps=10.0):
    """Three look-alike performers in a row (one occluded for 4 s), one small static onlooker at the back."""
    people = [(300, 900, 600, 0.0, True), (700, 900, 600, 0.5, True), (1100, 900, 600, 1.0, True),
              (1600, 420, 220, 0.0, False)]
    rows = {k: [] for k in ("frame", "xy", "conf", "box", "score", "app")}
    app = np.full(APP_DIM, 1.0 / (APP_DIM // 2), np.float32)        # identical clothing for everybody
    rng = np.random.default_rng(0)
    for f in range(T):
        t = f / fps
        for k, (cx, foot, h, ph, mv) in enumerate(people):
            if k == 1 and 15.0 <= t < 19.0:                         # occluded longer than the tracker's buffer
                continue
            xy = _person(cx, foot, h, ph, t, mv) + rng.normal(0, 1.5, (17, 2))
            rows["frame"].append(f)
            rows["xy"].append(xy)
            rows["conf"].append(np.full(17, 0.9))
            rows["box"].append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
            rows["score"].append(0.9)
            rows["app"].append(app)
    a = {k: np.asarray(v, np.float32) for k, v in rows.items()}
    return MultiPoses(t=np.arange(T) / fps, frame=a["frame"].astype(np.int32), xy=a["xy"], conf=a["conf"], box=a["box"],
                      score=a["score"], app=a["app"], frame_size=(1920, 1080), skeleton_name="coco17",
                      meta={"sample_fps": fps, "conf_thr": 0.3})


def test_identities_survive_occlusion_and_onlooker_is_screened_out():
    mp = _scene()
    ids, diag = build_identities(mp, GroupParams(baseline_s=10.0))
    perf = sorted((i for i in ids if i.performer), key=lambda i: i.foot[0])
    assert len(perf) == 3
    xs = [i.foot[0] for i in perf]
    assert np.allclose(xs, [300, 700, 1100], atol=20)
    # the occluded performer comes back as the same identity (two tracklets stitched), no new one created
    assert len(perf[1].tracklets) == 2 and len(perf[1].frames) == 400 - 40
    assert all(len(i.frames) == 400 for i in (perf[0], perf[2]))
    onlooker = [i for i in ids if not i.performer]
    assert len(onlooker) == 1 and onlooker[0].screening["reasons"]
    # layout: same row for the three, the onlooker much farther back
    assert max(abs(i.layout["z_m"] - perf[0].layout["z_m"]) for i in perf) < 0.2
    assert onlooker[0].layout["z_m"] > 2 * perf[0].layout["z_m"]


def _pinhole_scene(people, T=200, f=1400.0, cx=960.0, y_h=380.0, h_cam=1.3, fps=10.0):
    """People of different real heights (same head size) seen by a level pinhole camera.
    ``people``: (x_m, z_m, stature_m); returns MultiPoses with COCO-17 keypoints."""
    ix = COCO17.index
    rows = {k: [] for k in ("frame", "xy", "conf", "box", "score", "app")}
    rng = np.random.default_rng(1)
    for fr in range(T):
        t = fr / fps
        for k, (X, Z, H) in enumerate(people):
            def P(dx, h):                      # world point at (X + dx, height h above floor, depth Z) -> pixels
                return (cx + f * (X + dx) / Z, y_h + f * (h_cam - h) / Z)
            sw = 0.25 * np.sin(2 * np.pi * (t - 0.3 * k) / 6.0)
            pts = {"nose": (0, H - 0.12), "left_eye": (0.0317, H - 0.09), "right_eye": (-0.0317, H - 0.09),
                   "left_ear": (0.075, H - 0.11), "right_ear": (-0.075, H - 0.11),
                   "left_shoulder": (0.2 * H / 1.7, 0.818 * H), "right_shoulder": (-0.2 * H / 1.7, 0.818 * H),
                   "left_elbow": (0.25 * H / 1.7 + sw, 0.63 * H), "right_elbow": (-0.25 * H / 1.7 - sw, 0.63 * H),
                   "left_wrist": (0.27 * H / 1.7 + 2 * sw, 0.48 * H), "right_wrist": (-0.27 * H / 1.7 - 2 * sw, 0.48 * H),
                   "left_hip": (0.1 * H / 1.7, 0.53 * H), "right_hip": (-0.1 * H / 1.7, 0.53 * H),
                   "left_knee": (0.1 * H / 1.7, 0.285 * H), "right_knee": (-0.1 * H / 1.7, 0.285 * H),
                   "left_ankle": (0.1 * H / 1.7, 0.0), "right_ankle": (-0.1 * H / 1.7, 0.0)}
            xy = np.zeros((17, 2))
            for n, (dx, h) in pts.items():
                xy[ix[n]] = P(dx, h)
            xy += rng.normal(0, 0.8, xy.shape)
            rows["frame"].append(fr)
            rows["xy"].append(xy)
            rows["conf"].append(np.full(17, 0.9))
            rows["box"].append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
            rows["score"].append(0.9)
            app = np.zeros(APP_DIM, np.float32)
            app[k] = app[APP_DIM // 2 + k] = 1.0                       # different clothing
            rows["app"].append(app)
    a = {k: np.asarray(v, np.float32) for k, v in rows.items()}
    return MultiPoses(t=np.arange(T) / fps, frame=a["frame"].astype(np.int32), xy=a["xy"], conf=a["conf"], box=a["box"],
                      score=a["score"], app=a["app"], frame_size=(1920, 1080), skeleton_name="coco17",
                      meta={"sample_fps": fps, "conf_thr": 0.3})


def test_depth_does_not_assume_equal_heights():
    people = [(-1.5, 5.0, 1.50), (0.0, 5.0, 1.85), (1.5, 7.0, 1.68)]   # a short and a tall person side by side
    f = (1920 / 2) / np.tan(np.radians(GroupParams().hfov_deg) / 2)
    ids, _ = build_identities(_pinhole_scene(people, f=f), GroupParams(baseline_s=5.0))
    ids = sorted(ids, key=lambda i: i.foot[0])
    for i, (_, z, h) in zip(ids, people):
        assert abs(i.layout["z_m"] / z - 1) < 0.05                     # combined depth: within 5 %
        assert abs(i.layout["height_m"] / h - 1) < 0.06                # and each person's real height
    # the equal-height assumption alone would put the short and the tall person ~10 % off in opposite directions
    st = [i.layout["cues"]["stature"] for i in ids[:2]]
    assert st[0] > 5.4 and st[1] < 4.6
    assert all(i.performer for i in ids)


def test_identity_edits_replay_in_order():
    mp = _scene()
    base, _ = build_identities(mp, GroupParams())
    perf = sorted((i for i in base if i.performer), key=lambda i: i.foot[0])
    a, b = perf[0].id, perf[1].id
    edits = ({"op": "split", "id": a, "at": 20.0}, {"op": "merge", "from": max(i.id for i in base) + 1, "into": a},
             {"op": "role", "id": b, "performer": False})
    ids, diag = build_identities(mp, GroupParams(edits=edits))
    by = {i.id: i for i in ids}
    assert len(diag["edits_applied"]) == 3
    assert len(by[a].frames) == len(perf[0].frames)                    # split then merged back: unchanged
    assert not by[b].performer and "人工排除" in by[b].screening["reasons"]


def test_sync_from_motion_signals():
    from motioncoach.engine.group import signal_sync
    mp = _scene()                                     # performers 0.5 s apart: x 300 earliest, 1100 latest
    ids, _ = build_identities(mp, GroupParams())
    perf = sorted((i for i in ids if i.performer), key=lambda i: i.foot[0])
    sync = signal_sync(mp, perf)["sync"]
    got = [sync[i.id]["offset_median_s"] for i in perf]
    assert np.allclose(got, [-0.5, 0.0, 0.5], atol=0.15)


def test_camera_motion_recovers_a_pan(tmp_path):
    import cv2
    from motioncoach.media.camera import apply, camera_motion
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur((rng.random((900, 1600)) * 255).astype(np.uint8), (0, 0), 3)
    tex = cv2.cvtColor(cv2.equalizeHist(tex), cv2.COLOR_GRAY2BGR)
    path = tmp_path / "pan.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for k in range(30):                                    # camera pans right: content moves 6 px left per frame
        vw.write(tex[100:820, 100 + 6 * k:1380 + 6 * k])
    vw.release()
    A = camera_motion(path, 10.0, [np.zeros((0, 4))] * 30, (1280, 720))
    assert A.shape == (30, 2, 3)
    assert abs(A[-1, 0, 2] - 6 * 29) < 6 and abs(A[-1, 1, 2]) < 4     # last frame maps back by +174 px
    p = apply(A, np.array([29]), np.array([[[100.0, 200.0]]]))
    assert abs(p[0, 0, 0] - 274) < 6
