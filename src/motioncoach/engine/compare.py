"""Numeric comparison at corresponding moments.

The engine reports *evidence* only: signed differences, coverage, timing and alignment
quality per segment and feature. Whether a difference is acceptable, how it is weighted and
how it is phrased belongs to the application layer (apps/*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import Alignment


@dataclass(frozen=True)
class Segment:
    id: str
    name: str
    start: float   # reference seconds
    end: float


def aligned_user_values(F_user: np.ndarray, matches: list[np.ndarray]) -> np.ndarray:
    """(N_ref, K): median of the user samples matched to each reference sample."""
    out = np.full((len(matches), F_user.shape[1]), np.nan)
    for i, js in enumerate(matches):
        if len(js):
            with np.errstate(all="ignore"):
                vals = F_user[js]
                ok = np.isfinite(vals).any(axis=0)
                if ok.any():
                    out[i, ok] = np.nanmedian(vals[:, ok], axis=0)
    return out


def _motion(F: np.ndarray, scales: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(F / scales, axis=0, prepend=np.nan))
    valid = np.isfinite(d)
    n = valid.sum(axis=1)
    m = np.where(valid, d, 0).sum(axis=1) / np.maximum(n, 1)
    return np.where(n > 0, m, np.inf)


def _pick_holds(idx: np.ndarray, motion: np.ndarray, F_ref: np.ndarray, scales: np.ndarray,
                neutral: np.ndarray | None, fraction: float) -> np.ndarray:
    k = max(1, int(round(fraction * len(idx))))
    margin = int(0.1 * len(idx))
    inner = idx[margin:len(idx) - margin] if len(idx) - 2 * margin >= k else idx
    calm = inner[motion[inner] <= np.median(motion[inner])]
    if neutral is None or len(calm) <= k:
        return calm[np.argsort(motion[calm])[:k]] if len(calm) >= k else inner[np.argsort(motion[inner])[:k]]
    d = np.abs((F_ref[calm] - neutral) / scales)
    valid = np.isfinite(d)
    dist = np.where(valid, d, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    return calm[np.argsort(-dist)[:k]]


def compare_segments(t_ref: np.ndarray, F_ref: np.ndarray, t_user: np.ndarray, F_user: np.ndarray,
                     alignment: Alignment, segments: list[Segment], names: list[str], scales: np.ndarray,
                     hold_fraction: float = 0.15, neutral: np.ndarray | None = None) -> dict:
    """Per-segment, per-feature statistics of delta = user - reference.

    "Hold" (key-pose) samples: interior of the segment, below-median motion, and — when a
    ``neutral`` standing pose is given — the most distinctive postures. Lowest motion alone
    picks rest/transition poses rather than the defining posture of a movement. Annotated key
    frames from a coach should replace this heuristic when available.
    """
    matches = alignment.user_for_ref(len(t_ref))
    U_al = aligned_user_values(F_user, matches)
    delta = U_al - F_ref
    user_t_for_ref = np.array([t_user[js].mean() if len(js) else np.nan for js in matches])
    cost_for_ref = np.full(len(t_ref), np.nan)
    for i in range(len(t_ref)):
        sel = alignment.local_cost[alignment.ref_idx == i]
        cost_for_ref[i] = sel.mean() if len(sel) else np.nan
    motion = _motion(F_ref, scales)

    user_span = float(t_user[alignment.user_idx[-1]] - t_user[alignment.user_idx[0]])
    ref_span = float(t_ref[-1] - t_ref[0])
    tempo = user_span / ref_span if ref_span > 0 else float("nan")

    seg_out = []
    for seg in segments:
        m = (t_ref >= seg.start) & (t_ref < seg.end)
        if not m.any():
            continue
        ut = user_t_for_ref[m]
        u0, u1 = float(np.nanmin(ut)), float(np.nanmax(ut))
        ref_dur = float(seg.end - seg.start)
        user_dur = u1 - u0
        idx = np.flatnonzero(m)
        hold = _pick_holds(idx, motion, F_ref, scales, neutral, hold_fraction)
        feats = {}
        for c, n in enumerate(names):
            d = delta[m, c]
            ok = np.isfinite(d)
            dh = delta[hold, c]
            okh = np.isfinite(dh)
            feats[n] = {
                "coverage": float(ok.mean()),
                "mean_delta": float(np.mean(d[ok])) if ok.any() else None,
                "median_abs": float(np.median(np.abs(d[ok]))) if ok.any() else None,
                "p90_abs": float(np.percentile(np.abs(d[ok]), 90)) if ok.any() else None,
                "hold_coverage": float(okh.mean()) if len(okh) else 0.0,
                "hold_valid": int(okh.sum()),
                "hold_n": int(len(okh)),
                "hold_mean_delta": float(np.mean(dh[okh])) if okh.any() else None,
                "hold_median_abs": float(np.median(np.abs(dh[okh]))) if okh.any() else None,
                "ref_mean": float(np.nanmean(F_ref[m, c])) if np.isfinite(F_ref[m, c]).any() else None,
                "user_mean": float(np.nanmean(U_al[m, c])) if np.isfinite(U_al[m, c]).any() else None,
            }
        seg_out.append({
            "id": seg.id, "name": seg.name,
            "ref_start": seg.start, "ref_end": seg.end,
            "user_start": u0, "user_end": u1,
            "duration_ratio": user_dur / ref_dur if ref_dur > 0 else None,
            "duration_ratio_vs_tempo": (user_dur / ref_dur) / tempo if ref_dur > 0 and tempo > 0 else None,
            "align_cost": float(np.nanmean(cost_for_ref[m])),
            "hold_times": [float(t_ref[i]) for i in sorted(hold)],   # the full key-moment set used by the statistics
            "features": feats,
        })

    return {
        "tempo_ratio": tempo,
        "user_span": [float(t_user[alignment.user_idx[0]]), float(t_user[alignment.user_idx[-1]])],
        "align_cost": alignment.total_cost,
        "segments": seg_out,
        "series": {"user_aligned": U_al, "delta": delta, "user_t_for_ref": user_t_for_ref,
                   "align_cost": cost_for_ref},
    }
