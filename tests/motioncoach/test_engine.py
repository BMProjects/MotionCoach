"""Regression tests for the comparison engine (the failure modes found in the legacy analyzer)."""

import numpy as np
import pytest

from motioncoach.engine.align import align
from motioncoach.engine.compare import Segment, compare_segments
from motioncoach.engine.features import FEATURES, compute_features, lr_swap_features
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
