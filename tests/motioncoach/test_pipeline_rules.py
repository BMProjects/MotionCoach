"""Regression tests for review findings: mirroring, frame identity across cache paths, validity gating."""

import numpy as np
import pytest

from motioncoach.apps.analysis import resolve_swap_mode


@pytest.mark.parametrize("mirrored,practice,expected", [
    ("no", False, "no"), ("yes", False, "yes"), ("no", True, "yes"), ("yes", True, "no"), ("auto", True, "auto"),
])
def test_mirror_transforms_compose_as_xor(mirrored, practice, expected):
    assert resolve_swap_mode(mirrored, practice) == expected


class _Dummy:
    name, model_id, skeleton_name, device, conf_threshold = "dummy", "dummy", "coco17", "cpu", 0.5

    def open(self):
        pass

    def estimate(self, frame, ts_ms):
        return np.zeros((17, 2), np.float32), np.ones(17, np.float32), None

    def close(self):
        pass


def test_window_run_and_full_slice_sample_identical_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from motioncoach.media.extract import slice_window
    from motioncoach.perception.base import run_on_video

    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(30 * 6):
        w.write(np.full((48, 64, 3), i % 255, np.uint8))
    w.release()
    full = run_on_video(_Dummy(), path, 10, 0.0, None)
    win = run_on_video(_Dummy(), path, 10, 2.07, 4.5)
    sl = slice_window(full, 2.07, 4.5)
    np.testing.assert_allclose(win.t, sl.t, atol=1e-6)
    assert len(win.t) > 15


def _profile():
    from motioncoach.apps.assets import load_profile
    return load_profile("baduanjin")


def _cmp(hold_valid, hold_n=15, coverage=0.9, align_cost=0.5):
    feat = {"coverage": coverage, "mean_delta": 2.0, "median_abs": 2.0, "p90_abs": 3.0,
            "hold_coverage": hold_valid / hold_n, "hold_valid": hold_valid, "hold_n": hold_n,
            "hold_mean_delta": 2.0, "hold_median_abs": 2.0 if hold_valid else None, "ref_mean": 170.0, "user_mean": 172.0}
    seg = {"id": "s1", "name": "第一式", "ref_start": 0.0, "ref_end": 10.0, "user_start": 0.0, "user_end": 20.0,
           "duration_ratio": 2.0, "duration_ratio_vs_tempo": 1.0, "align_cost": align_cost, "hold_times": [5.0],
           "features": {"l_elbow": feat}}
    return {"segments": [seg]}


def test_few_valid_key_moments_are_not_judged():
    from motioncoach.apps.rules import evaluate

    out = evaluate(_cmp(hold_valid=1), _profile())
    cell = out["segments"][0]["features"]["l_elbow"]
    assert cell["status"] == "unmeasurable" and "1/15" in cell["reason"]
    assert out["pass_rate"] is None


def test_unreliable_alignment_withholds_judgement_and_direction():
    from motioncoach.apps.rules import evaluate

    out = evaluate(_cmp(hold_valid=15, align_cost=3.0), _profile())
    cell = out["segments"][0]["features"]["l_elbow"]
    assert cell["status"] == "uncertain" and "text" not in cell
    assert out["pass_rate"] is None and out["counts"]["uncertain"] == 1
    ok = evaluate(_cmp(hold_valid=15), _profile())
    assert ok["segments"][0]["features"]["l_elbow"]["status"] == "good" and ok["pass_rate"] == 1.0
