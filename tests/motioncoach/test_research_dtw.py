"""Kernels of the correspondence study: slope limits and template search."""

import numpy as np

from motioncoach.engine.dtw import dtw_slope, hsmm_segment, occurrence_path, subseq_profile


def _cost(a, b):
    return np.abs(a[:, None] - b[None]).astype(np.float32)


def test_dtw_slope_never_stalls():
    x = np.sin(np.linspace(0, 3, 40))
    y = np.r_[np.full(30, x[0]), x]          # long rest at the start would make plain DTW stall on one frame
    ri, ui, _ = dtw_slope(_cost(x, y))
    assert ri[0] == 0 and ui[0] == 0 and ri[-1] == 39 and ui[-1] == len(y) - 1
    run = np.diff(np.flatnonzero(np.r_[True, np.diff(ri) != 0, True]))
    assert run.max() <= 2                    # a reference sample takes at most 2 user samples (slope >= 1/2)


def test_template_found_in_stream():
    tpl = np.sin(np.linspace(0, np.pi, 30)) ** 2
    stream = np.r_[np.zeros(50), np.interp(np.linspace(0, 29, 45), np.arange(30), tpl), np.zeros(40)]   # 1.5x slower
    cost, start, B = subseq_profile(_cost(tpl, stream))
    end = int(np.argmin(cost))
    ti, ui = occurrence_path(B, end)
    assert abs(start[end] - 50) <= 3 and abs(end - 94) <= 3
    assert ti[0] == 0 and ti[-1] == 29 and np.all(np.diff(ui) >= 0)


def test_hsmm_orders_sections():
    emis = np.ones((3, 90))
    emis[0, :20], emis[1, 20:60], emis[2, 60:] = 0, 0, 0
    seg = hsmm_segment(emis, np.array([10.0, 20.0, 15.0]), tempo=2.0, skip=5.0)
    assert [s for s, _ in seg] == [0, 20, 60] and seg[-1][1] == 90
