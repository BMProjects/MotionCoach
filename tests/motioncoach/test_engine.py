"""Regression tests for the comparison engine (the failure modes found in the legacy analyzer)."""

import numpy as np
import pytest

from motioncoach.engine.align import align
from motioncoach.engine.compare import Segment, compare_segments
from motioncoach.engine.features import FEATURES, compute_features, lr_swap_features
from motioncoach.engine.reps import compare_by_reps
from motioncoach.engine.templates import compare_by_templates
from motioncoach.engine.sequence import PoseSequence, clean
from motioncoach.engine.skeleton import MEDIAPIPE33

J = len(MEDIAPIPE33.joints)


def _standing_pose(w=400.0, h=800.0, arm_raise=0.0):
    """Synthetic front-facing pose in pixels; arm_raise in [0,1] lifts the performer's left arm."""
    ix = MEDIAPIPE33.index
    p = np.zeros((J, 2))
    cx = w / 2
    # performer faces the camera: performer's left appears on image right
    pts = {
        "nose": (cx, 120), "left_shoulder": (cx + 60, 200), "right_shoulder": (cx - 60, 200),
        "left_elbow": (cx + 70, 300), "right_elbow": (cx - 70, 300),
        "left_wrist": (cx + 75, 390), "right_wrist": (cx - 75, 390),
        "left_hip": (cx + 40, 420), "right_hip": (cx - 40, 420),
        "left_knee": (cx + 45, 570), "right_knee": (cx - 45, 570),
        "left_ankle": (cx + 50, 720), "right_ankle": (cx - 50, 720),
    }
    for name, xy in pts.items():
        p[ix[name]] = xy
    for name in MEDIAPIPE33.joints:
        if name not in pts:
            p[ix[name]] = (cx, 400)
    if arm_raise:
        a = np.radians(180 * arm_raise)
        sh = p[ix["left_shoulder"]]
        p[ix["left_elbow"]] = sh + 100 * np.array([np.sin(a), np.cos(a)])
        p[ix["left_wrist"]] = sh + 190 * np.array([np.sin(a), np.cos(a)])
    return p


def _seq(poses, fps=10.0, size=(400, 800), conf=None):
    xy = np.stack(poses).astype(np.float32)
    T = len(poses)
    c = np.ones((T, J), np.float32) if conf is None else conf
    return PoseSequence(t=np.arange(T) / fps, xy=xy, conf=c, frame_size=size)


def test_angles_are_isotropic_across_frame_aspect():
    """Same body in pixels on a portrait vs landscape canvas must give identical angles."""
    pose = _standing_pose(arm_raise=0.3)
    a = _seq([pose] * 3, size=(400, 800))
    b = _seq([pose] * 3, size=(1600, 900))
    Fa, names, _ = compute_features(a)
    Fb, _, _ = compute_features(b)
    np.testing.assert_allclose(Fa, Fb, atol=1e-6)
    # elbow nearly straight in the synthetic arm
    assert Fa[0, names.index("l_elbow")] > 170


def test_unmirror_swaps_identity_and_is_involution():
    pose = _standing_pose(arm_raise=0.5)
    s = _seq([pose] * 2)
    m = s.unmirror()
    ix = MEDIAPIPE33.index
    # the raised (performer-left) arm becomes the right arm, x flipped
    np.testing.assert_allclose(m.xy[0, ix["right_wrist"], 0], 400 - s.xy[0, ix["left_wrist"], 0])
    np.testing.assert_allclose(m.unmirror().xy, s.xy)


def test_mirror_equals_feature_swap():
    s = _seq([_standing_pose(arm_raise=r) for r in np.linspace(0, 0.8, 5)])
    F, names, _ = compute_features(s)
    Fm, _, _ = compute_features(s.unmirror())
    np.testing.assert_allclose(lr_swap_features(F, names), Fm, atol=1e-6)


def test_mirrored_pose_is_not_a_perfect_match():
    """Legacy Procrustes scored a mirror image 100. Asymmetric poses must now differ."""
    s = _seq([_standing_pose(arm_raise=0.6)] * 3)
    F, names, _ = compute_features(s)
    Fm, _, _ = compute_features(s.unmirror())
    i = names.index("l_shoulder")
    assert abs(F[0, i] - Fm[0, i]) > 60


def test_low_confidence_joint_is_missing_not_zero():
    poses = [_standing_pose()] * 20
    conf = np.ones((20, J), np.float32)
    conf[:, MEDIAPIPE33.index["left_wrist"]] = 0.1
    s, _ = clean(_seq(poses, conf=conf))
    F, names, _ = compute_features(s)
    assert np.isnan(F[:, names.index("l_elbow")]).all()
    assert np.isfinite(F[:, names.index("r_elbow")]).all()   # other joints unaffected


def test_short_gap_interpolated_long_gap_kept():
    poses = [_standing_pose()] * 30
    conf = np.ones((30, J), np.float32)
    wi = MEDIAPIPE33.index["left_wrist"]
    conf[5:7, wi] = 0.0     # 0.2 s gap -> filled
    conf[15:25, wi] = 0.0   # 1.0 s gap -> stays missing
    s, interp = clean(_seq(poses, conf=conf), max_gap_s=0.4)
    assert np.isfinite(s.xy[5:7, wi]).all() and interp[5:7, wi].all()
    assert np.isnan(s.xy[15:25, wi]).all()


def _wave(t):
    return np.c_[60 * np.sin(t), 40 * np.cos(0.5 * t), 30 * np.sin(1.7 * t)]


def test_dtw_recovers_tempo_and_skips_lead_in():
    t_ref = np.linspace(0, 12, 120)
    R = _wave(t_ref)
    # user: 3 s of unrelated motion, then the same movement 2x slower
    lead = np.c_[np.full(30, 200.0), np.full(30, -200.0), np.full(30, 200.0)]
    t_u = np.linspace(0, 12, 240)
    U = np.vstack([lead, _wave(t_u)])
    al = align(R, U, scales=np.array([15.0, 15.0, 15.0]))
    assert al.user_idx[0] >= 28                       # lead-in skipped
    mid = al.user_idx[al.ref_idx == 60].mean()
    assert abs(mid - (30 + 120)) < 6                  # ref t=6 s <-> user 30 + 2*60 samples


def test_compare_reports_signed_delta_and_timing():
    t = np.arange(50) / 10.0
    R = np.c_[np.full(50, 150.0), np.full(50, 170.0)]
    U = np.c_[np.full(100, 135.0), np.full(100, 170.0)]   # user elbow 15 deg more bent, 2x slower
    names = ["l_elbow", "r_elbow"]
    scales = np.array([15.0, 15.0])
    al = align(R + np.linspace(0, 1, 50)[:, None], U + np.linspace(0, 1, 100)[:, None], scales)
    out = compare_segments(t, R, np.arange(100) / 10.0, U, al, [Segment("a", "A", 0, 5)], names, scales)
    f = out["segments"][0]["features"]
    assert f["l_elbow"]["mean_delta"] == pytest.approx(-15, abs=0.5)
    assert f["r_elbow"]["median_abs"] == pytest.approx(0, abs=0.5)
    assert FEATURES["l_elbow"].less == "更弯曲"
    assert out["tempo_ratio"] == pytest.approx(2.0, rel=0.1)


def test_reps_pair_one_demo_with_several_practice_cycles():
    """1 reference cycle vs 3 slower practitioner cycles: each cycle is paired interval-to-interval, not
    piled onto single reference frames, and the key-moment difference is measured at the same phase."""
    def cycles(n_cycles, period_s, rest_s, offset=0.0):
        t, ys = [], []
        T = n_cycles * (period_s + rest_s) + rest_s
        tt = np.arange(0, T, 0.1)
        y = np.full(len(tt), 20.0)
        for k in range(n_cycles):
            a = rest_s + k * (period_s + rest_s)
            m = (tt >= a) & (tt < a + period_s)
            y[m] = 20 + (140 + offset) * np.sin(np.pi * (tt[m] - a) / period_s) ** 2
        return tt, y
    t_r, yr = cycles(1, 6.0, 2.0)
    t_u, yu = cycles(3, 9.0, 3.0, offset=-20.0)            # practitioner raises 20 deg less, 1.5x slower
    names = ["l_shoulder", "r_shoulder"]
    F_r, F_u = np.c_[yr, yr], np.c_[yu, yu]
    scales = np.array([15.0, 15.0])
    out = compare_by_reps(t_r, F_r, t_u, F_u, [Segment("a", "A", 0, t_r[-1] + 0.1)], [(0, len(t_u))],
                          names, scales, neutral=np.array([20.0, 20.0]))
    seg = out["segments"][0]
    assert (seg["reps"]["count_ref"], seg["reps"]["count_user"], seg["reps"]["count_paired"]) == (1, 3, 3)
    assert all(1.2 < u["duration_ratio"] < 1.8 for u in seg["reps"]["user"])
    assert seg["features"]["l_shoulder"]["hold_mean_delta"] == pytest.approx(-20, abs=3)
    assert len(out["user_map"]) == 3


def test_templates_find_every_practice_cycle():
    """Annotated single reference cycle vs 3 slower practitioner cycles: each is found and compared on its hold."""
    def cycles(n_cycles, period_s, rest_s, offset=0.0):
        tt = np.arange(0, n_cycles * (period_s + rest_s) + rest_s, 0.1)
        y = np.full(len(tt), 20.0)
        for k in range(n_cycles):
            a = rest_s + k * (period_s + rest_s)
            m = (tt >= a) & (tt < a + period_s)
            y[m] = 20 + (140 + offset) * np.sin(np.pi * (tt[m] - a) / period_s) ** 2
        return tt, y
    t_r, yr = cycles(1, 6.0, 2.0)
    t_u, yu = cycles(3, 9.0, 3.0, offset=-20.0)            # 20 deg lower at the top, 1.5x slower
    names = ["l_shoulder", "r_shoulder"]
    out = compare_by_templates(t_r, np.c_[yr, yr], t_u, np.c_[yu, yu], [Segment("a", "A", 0, t_r[-1] + 0.1)],
                               [(0, len(t_u))], names, np.array([15.0, 15.0]), ref_reps={"a": [[2.0, 8.0, 4.5, 5.5]]})
    reps = out["segments"][0]["reps"]
    assert (reps["count_ref"], reps["count_user"]) == (1, 3)
    starts = sorted(u["start"] for u in reps["user"])
    assert all(abs(s - e) < 1.5 for s, e in zip(starts, [3.0, 15.0, 27.0]))
    assert out["segments"][0]["features"]["l_shoulder"]["hold_mean_delta"] == pytest.approx(-20, abs=3)
    pb = out["playback"]
    assert len(pb["ref_t_for_user"]) == len(t_u) and len(pb["user_t_for_ref"]) == len(t_r)
    # practitioner-led playback loops the template: the mapped reference hold appears once per repetition
    shown = np.array(pb["ref_t_for_user"])
    assert sum(1 for a, b in zip(shown[:-1], shown[1:]) if b < a - 1.0) == 2


def test_timestamps_must_increase():
    with pytest.raises(ValueError):
        PoseSequence(t=np.array([0.0, 0.0]), xy=np.zeros((2, J, 2), np.float32),
                     conf=np.ones((2, J), np.float32), frame_size=(10, 10))


def test_normalized_skeleton_removes_body_size_and_distance():
    """Same posture, different limb proportions and camera distance -> zero difference vectors."""
    from motioncoach.engine.normalize import bone_lengths, normalized

    ref = _seq([_standing_pose(arm_raise=0.4)] * 5)
    pose = _standing_pose(arm_raise=0.4)
    ix = MEDIAPIPE33.index
    # user: longer forearms (same direction), then filmed from twice as close
    for side in ("left", "right"):
        e, w = ix[f"{side}_elbow"], ix[f"{side}_wrist"]
        pose[w] = pose[e] + 1.3 * (pose[w] - pose[e])
    pose = (pose - pose[ix["left_hip"]]) * 2.0 + 300
    usr = _seq([pose] * 5, size=(1200, 1600))
    L_ref = bone_lengths(ref)
    unit = L_ref[("hip_mid", "shoulder_mid")]
    d = normalized(usr, L_ref, bone_lengths(usr), unit) - normalized(ref, L_ref, L_ref, unit)
    assert np.nanmax(np.abs(d)) < 1e-6


def test_patterns_separate_near_reference_and_deviation_groups():
    from motioncoach.engine.patterns import cluster_deviations

    rng = np.random.default_rng(0)
    tol = np.array([15.0, 15.0, 0.15])
    near = rng.normal(0, 3, (300, 3)) * [1, 1, 0.01]
    bent = np.c_[rng.normal(-40, 3, 80), rng.normal(0, 3, 80), rng.normal(0, 0.02, 80)]
    narrow = np.c_[rng.normal(0, 3, 60), rng.normal(0, 3, 60), rng.normal(-0.5, 0.03, 60)]
    delta = np.vstack([near, bent, narrow])
    out = cluster_deviations(delta, tol, np.ones(len(delta), bool))
    lab = out["labels"]
    assert (lab[:300] == 0).mean() > 0.95                      # within tolerance -> class 0
    assert out["k"] == 2
    assert len(set(lab[300:380])) == 1 and len(set(lab[380:])) == 1 and lab[300] != lab[380]


def test_scale_from_upright_stature():
    """Eyes-to-ankles stature with Drillis & Contini proportions -> cm per pixel; torso cross-check."""
    from motioncoach.engine.scale import ANKLE_H, EYE_H, HIP_H, SHOULDER_H, estimate_scale, to_cm
    from motioncoach.engine.skeleton import MEDIAPIPE33 as SK
    H_px, T = 800.0, 30
    floor, cx = 900.0, 300.0
    y = lambda frac: floor - frac * H_px   # noqa: E731
    pts = {"left_eye": (cx + 10, y(EYE_H)), "right_eye": (cx - 10, y(EYE_H)),
           "left_shoulder": (cx + 60, y(SHOULDER_H)), "right_shoulder": (cx - 60, y(SHOULDER_H)),
           "left_hip": (cx + 40, y(HIP_H)), "right_hip": (cx - 40, y(HIP_H)),
           "left_knee": (cx + 40, y(0.285)), "right_knee": (cx - 40, y(0.285)),
           "left_ankle": (cx + 40, y(ANKLE_H)), "right_ankle": (cx - 40, y(ANKLE_H))}
    xy = np.full((T, len(SK.joints), 2), np.nan, np.float32)
    conf = np.zeros((T, len(SK.joints)), np.float32)
    for j, p in pts.items():
        xy[:, SK.index[j]] = p
        conf[:, SK.index[j]] = 0.9
    seq = PoseSequence(t=np.arange(T) / 10.0, xy=xy, conf=conf, frame_size=(600, 1000))
    torso_px = (SHOULDER_H - HIP_H) * H_px
    sc = estimate_scale(seq, torso_px, height_cm=170.0)
    assert sc["reliable"] and sc["cm_per_px"] == pytest.approx(170.0 / H_px, rel=1e-3)
    assert sc["torso_cm"] == pytest.approx(0.288 * 170.0, rel=1e-3)
    assert to_cm("stance_width", 0.5, sc) == pytest.approx(0.5 * 0.288 * 170.0, rel=1e-3)
    assert to_cm("l_elbow", 10.0, sc) is None
    xy[:, SK.index["left_hip"], 1] -= 200      # torso no longer matches stature -> unreliable, no cm
    xy[:, SK.index["right_hip"], 1] -= 200
    bad = estimate_scale(PoseSequence(t=seq.t, xy=xy, conf=conf, frame_size=(600, 1000)), torso_px)
    assert not bad["reliable"] and to_cm("stance_width", 0.5, bad) is None


def test_bone_check_keeps_a_limb_turning_into_the_image_plane():
    """A correct limb whose projection doubles (50 -> 100 px, e.g. turning sideways) must not be deleted;
    a joint that jumps to three times the longest projection still is."""
    from motioncoach.engine.filters import reject_long_bones
    T = 100
    xy = np.full((T, J, 2), 400.0, np.float32)
    ix = MEDIAPIPE33.index
    for k in range(T):
        L = 50.0 if k < 80 else 100.0
        xy[k, ix["left_shoulder"]] = (400, 300)
        xy[k, ix["left_elbow"]] = (400, 300 + L)
        xy[k, ix["right_shoulder"]] = (300, 300)
        xy[k, ix["right_elbow"]] = (300, 350)
    xy[50, ix["left_elbow"]] = (400, 300 + 300)                   # a wrong detection: 3x the longest projection
    seq = PoseSequence(t=np.arange(T) / 10.0, xy=xy, conf=np.ones((T, J), np.float32), frame_size=(800, 800))
    out, n = reject_long_bones(seq)
    el = out.joint("left_elbow")[0]
    assert np.isfinite(el[80:]).all()                             # the turned limb is kept
    assert not np.isfinite(el[50]).any() and n >= 1               # the jump is removed
