"""Repetition-level correspondence: interval ↔ interval instead of frame ↔ frame.

Reference and practitioner often differ in tempo *and* in how many times a move is repeated
(e.g. 1 demonstration vs 6 practice repetitions). Frame-level DTW cannot map six cycles onto
one, so it piles five of them onto a single reference frame. Here, within each section:

1. one periodic signal = the reference section's first principal component (practitioner projected
   onto the same axis), smoothed over 1 s;
2. both sides are split into repetitions at the minima between prominent peaks;
3. every practitioner repetition is paired with the reference repetition it matches best and
   compared on normalised phase (both cycles stretched to 0–1), so durations stay balanced;
4. key moments are intervals: the plateau around the reference repetition's peak, and the same
   phase interval of each practitioner repetition, compared by the median over the interval.

Differences in repetition count are reported, not hidden in the time warp. Alternating
left/right moves are also matched against the mirrored template (symmetric limb compared), and
pairs with an implausible duration ratio are left unpaired instead of being stretched.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .compare import Segment
from .features import lr_swap_features


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def cycle_signals(Fr: np.ndarray, Fu: np.ndarray, scales: np.ndarray, neutral: np.ndarray | None):
    """First principal component of the standardised reference features; same axis for the practitioner."""
    Zr, Zu = Fr / scales, Fu / scales
    mu = np.nanmean(Zr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Zr0, Zu0 = np.nan_to_num(Zr - mu), np.nan_to_num(Zu - mu)
    if len(Zr0) < 3 or not np.any(Zr0):
        return np.zeros(len(Zr0)), np.zeros(len(Zu0))
    w = np.linalg.svd(Zr0, full_matrices=False)[2][0]
    sr, su = Zr0 @ w, Zu0 @ w
    # orient so that "peak" = away from the neutral stance (or the larger excursion when no neutral is known)
    if neutral is not None and np.isfinite(neutral).any():
        flip = np.nan_to_num((neutral / scales) - mu) @ w > 0
    else:
        flip = abs(sr.min()) > abs(sr.max())
    return (-sr, -su) if flip else (sr, su)


def split_reps(sig: np.ndarray, dt: float, min_period_s: float = 3.0, prom: float = 0.35,
               rel_height: float = 0.85) -> list[tuple[int, int, int]]:
    """[(start, end_exclusive, peak)] = the *active core* of each repetition.

    A core runs from where the smoothed cycle signal rises into a peak to where it falls back
    (``peak_widths`` at ``rel_height`` of the peak's prominence). Entry/return transitions and rest
    between repetitions lie outside the cores, so a single demonstration cycle is not stretched over
    a whole section. No clear periodicity, or a section too short to repeat -> one core spanning it.
    """
    n = len(sig)
    if n < 3:
        return [(0, n, max(0, n // 2))]
    if n * dt < 2 * min_period_s:   # too short to contain repetitions (e.g. a dance phrase): one interval
        return [(0, n, int(np.argmax(sig)))]
    x = _smooth(sig, max(1, int(round(1.0 / dt))))
    amp = np.percentile(x, 95) - np.percentile(x, 5)
    if amp < 0.5:   # less than half a tolerance of movement: treat as one block
        return [(0, n, int(np.argmax(x)))]
    peaks, _ = find_peaks(x, prominence=prom * amp, distance=max(1, int(min_period_s / dt)))
    if len(peaks) == 0:
        return [(0, n, int(np.argmax(x)))]
    _, _, left, right = peak_widths(x, peaks, rel_height=rel_height)
    reps, prev_end = [], 0
    for pk, a, b in zip(peaks, left, right):
        a, b = max(prev_end, int(np.floor(a))), min(n, int(np.ceil(b)) + 1)
        if b - a >= 3:
            reps.append((a, b, int(pk)))
            prev_end = b
    return reps or [(0, n, int(np.argmax(x)))]


def hold_interval(sig: np.ndarray, rep: tuple[int, int, int], dt: float, frac: float = 0.15) -> tuple[int, int]:
    """Contiguous plateau around the repetition's peak (within `frac` of the repetition's amplitude)."""
    a, b, p = rep
    x = _smooth(sig[a:b], max(1, int(round(0.5 / dt))))
    if len(x) == 0:
        return a, b
    p = int(np.clip(p - a, 0, len(x) - 1))
    thr = x[p] - frac * max(1e-6, x.max() - x.min())
    i0 = p
    while i0 > 0 and x[i0 - 1] >= thr:
        i0 -= 1
    i1 = p + 1
    while i1 < len(x) and x[i1] >= thr:
        i1 += 1
    if i1 - i0 < 3:   # at least ~0.3 s
        i0, i1 = max(0, p - 1), min(len(x), p + 2)
    return a + i0, a + i1


def _phase_resample(F: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    """Resample rows a..b-1 of F to n samples on normalised phase (NaN-aware, per column)."""
    src = np.linspace(0, 1, max(2, b - a)) if b - a > 1 else np.array([0.0, 1.0])
    blk = F[a:b] if b - a > 1 else np.repeat(F[a:a + 1], 2, axis=0)
    dst = np.linspace(0, 1, n)
    out = np.full((n, F.shape[1]), np.nan)
    for c in range(F.shape[1]):
        ok = np.isfinite(blk[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(dst, src[ok], blk[ok, c], left=np.nan, right=np.nan)
    return out


def _cost(A: np.ndarray, B: np.ndarray, scales: np.ndarray) -> float:
    d = np.abs((A - B) / scales)
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.inf


MIN_RATIO, MAX_RATIO, MAX_PAIR_COST = 0.4, 2.5, 2.5


def compare_by_reps(t_ref, F_ref, t_user, F_user, segments: list[Segment], user_bounds: list[tuple[int, int]],
                    names: list[str], scales: np.ndarray, neutral: np.ndarray | None = None) -> dict:
    """Same output shape as compare.compare_segments, plus per-repetition structure under seg['reps']."""
    dt_r = float(np.median(np.diff(t_ref))) if len(t_ref) > 1 else 0.1
    dt_u = float(np.median(np.diff(t_user))) if len(t_user) > 1 else 0.1
    N, K = F_ref.shape
    U_al = np.full((N, K), np.nan)
    user_t_for_ref = np.full(N, np.nan)
    user_idx_for_ref = np.full(N, -1, dtype=np.int64)
    weight = np.zeros(N)                      # practitioner samples represented by each reference sample
    cost_for_ref = np.full(N, np.nan)
    seg_out, user_map = [], []

    for seg, (u0, u1) in zip(segments, user_bounds):
        ridx = np.flatnonzero((t_ref >= seg.start) & (t_ref < seg.end))
        if len(ridx) < 3 or u1 - u0 < 3:
            continue
        r0, r1 = int(ridx[0]), int(ridx[-1]) + 1
        Fr, Fu = F_ref[r0:r1], F_user[u0:u1]
        sr, su = cycle_signals(Fr, Fu, scales, neutral)
        reps_r, reps_u = split_reps(sr, dt_r), split_reps(su, dt_u)
        holds_r = [hold_interval(sr, r, dt_r) for r in reps_r]
        # pair every practitioner repetition with its best-matching reference repetition (phase-normalised)
        Fu_sw = lr_swap_features(Fu, names)   # same practitioner, left/right exchanged (mirror-side repetition)
        pairs = []
        for k, (ua, ub, _) in enumerate(reps_u):
            best = None
            for j, (ra, rb, _) in enumerate(reps_r):
                for mirror, src in ((False, Fu), (True, Fu_sw)):
                    c = _cost(_phase_resample(src, ua, ub, rb - ra), Fr[ra:rb], scales)
                    if not mirror:
                        c_same = c
                    if best is None or c < best[0] - (0.05 if mirror else 0):   # prefer the same side on near-ties
                        best = (c, j, mirror)
            c, j, mirror = best
            ratio = ((ub - ua) * dt_u) / max(1e-6, (reps_r[j][1] - reps_r[j][0]) * dt_r)
            pairs.append({"user": k, "ref": j, "cost": c, "mirror": mirror, "ratio": ratio,
                          "paired": bool(MIN_RATIO <= ratio <= MAX_RATIO and c <= MAX_PAIR_COST)})
        # default user repetition per reference repetition: best cost, kept in temporal order
        default, last_end = {}, -1
        for j in range(len(reps_r)):
            cand = [p for p in pairs if p["ref"] == j and p["paired"] and reps_u[p["user"]][0] >= last_end]
            if cand:
                best = min(cand, key=lambda p: p["cost"])
                default[j] = best["user"]
                last_end = reps_u[best["user"]][1]
        rep_series, holds_u = [], []
        for p in pairs:
            ra, rb, _ = reps_r[p["ref"]]
            ua, ub, _ = reps_u[p["user"]]
            res = _phase_resample(Fu_sw if p["mirror"] else Fu, ua, ub, rb - ra)
            rep_series.append(res)
            # the practitioner's key moment = the reference hold's phase interval, mapped into this repetition
            # (each cycle's own peak can fall at a different phase, e.g. the cross-over instead of the full draw)
            f0, f1 = [(h - ra) / max(1, rb - ra - 1) for h in (holds_r[p["ref"]][0], holds_r[p["ref"]][1] - 1)]
            holds_u.append((ua + int(round(f0 * (ub - ua - 1))), ua + int(round(f1 * (ub - ua - 1))) + 1))
            if p["paired"]:
                user_map.append([float(t_user[u0 + ua]), float(t_user[min(u0 + ub, len(t_user)) - 1]),
                                 float(t_ref[r0 + ra]), float(t_ref[r0 + rb - 1])])
            if default.get(p["ref"]) == p["user"]:
                sl = slice(r0 + ra, r0 + rb)
                U_al[sl] = res
                ph = np.linspace(0, 1, rb - ra)
                user_t_for_ref[sl] = t_user[u0 + ua] + ph * (t_user[min(u0 + ub, len(t_user)) - 1] - t_user[u0 + ua])
                user_idx_for_ref[sl] = np.clip(np.round(u0 + ua + ph * (ub - ua - 1)).astype(int), 0, len(t_user) - 1)
                d = np.abs((res - Fr[ra:rb]) / scales)
                with np.errstate(all="ignore"):
                    cost_for_ref[sl] = np.nanmean(d, axis=1)
            if p["paired"]:
                weight[r0 + ra:r0 + rb] += (ub - ua) / max(1, rb - ra)

        # interval-to-interval key-moment comparison: median over the hold interval, same phase on both sides
        hold_deltas, hold_valid = [], []
        for p, res in zip(pairs, rep_series):
            ra = reps_r[p["ref"]][0]
            ha, hb = holds_r[p["ref"]]
            hu = res[ha - ra:hb - ra]
            with np.errstate(all="ignore"):
                mr = np.nanmedian(Fr[ha:hb], axis=0)
                mu = np.nanmedian(hu, axis=0) if p["paired"] else np.full(Fr.shape[1], np.nan)
            hold_deltas.append(mu - mr)
            hold_valid.append(np.isfinite(Fr[ha:hb]).mean(axis=0) * np.isfinite(hu).mean(axis=0))
        HD, HV = np.array(hold_deltas), np.array(hold_valid)

        feats = {}
        sl = slice(r0, r1)
        delta = U_al[sl] - F_ref[sl]
        core = np.zeros(r1 - r0, bool)          # samples inside a paired reference repetition core
        for j, ku in default.items():
            core[reps_r[j][0]:reps_r[j][1]] = True
        for c, n in enumerate(names):
            d = delta[:, c][core]
            ok = np.isfinite(d)
            hd = HD[:, c][np.isfinite(HD[:, c]) & (HV[:, c] >= 0.5)]
            feats[n] = {
                "coverage": float(ok.mean()) if len(ok) else 0.0,
                "mean_delta": float(np.mean(d[ok])) if ok.any() else None,
                "median_abs": float(np.median(np.abs(d[ok]))) if ok.any() else None,
                "p90_abs": float(np.percentile(np.abs(d[ok]), 90)) if ok.any() else None,
                "hold_coverage": float(len(hd) / max(1, sum(p["paired"] for p in pairs))),
                "hold_valid": int(len(hd)), "hold_n": int(sum(p["paired"] for p in pairs)),
                "hold_mean_delta": float(np.median(hd)) if len(hd) else None,   # median over repetitions
                "hold_median_abs": float(np.median(np.abs(hd))) if len(hd) else None,
                "per_rep": [None if not np.isfinite(v) else float(v) for v in HD[:, c]],
                "ref_mean": float(np.nanmean(Fr[:, c])) if np.isfinite(Fr[:, c]).any() else None,
                "user_mean": float(np.nanmean(U_al[sl][:, c])) if np.isfinite(U_al[sl][:, c]).any() else None,
            }
        ut0, ut1 = float(t_user[u0]), float(t_user[min(u1, len(t_user)) - 1])
        dur_ratio = (ut1 - ut0) / max(1e-6, seg.end - seg.start)
        seg_out.append({
            "id": seg.id, "name": seg.name, "ref_start": seg.start, "ref_end": seg.end,
            "user_start": ut0, "user_end": ut1, "duration_ratio": dur_ratio, "duration_ratio_vs_tempo": None,
            "align_cost": float(np.nanmean([p["cost"] for p in pairs if p["paired"]] or [np.inf])),
            "hold_times": [float(t_ref[r0 + (a + b) // 2]) for a, b in holds_r],
            "features": feats,
            "reps": {
                "ref": [{"start": float(t_ref[r0 + a]), "end": float(t_ref[r0 + b - 1]),
                         "hold": [float(t_ref[r0 + ha]), float(t_ref[r0 + hb - 1])]}
                        for (a, b, _), (ha, hb) in zip(reps_r, holds_r)],
                "user": [{"start": float(t_user[u0 + a]), "end": float(t_user[min(u0 + b, len(t_user)) - 1]),
                          "hold": [float(t_user[u0 + holds_u[k][0]]), float(t_user[min(u0 + holds_u[k][1], len(t_user)) - 1])],
                          "ref": pairs[k]["ref"], "cost": pairs[k]["cost"], "mirror": pairs[k]["mirror"],
                          "paired": pairs[k]["paired"], "duration_ratio": pairs[k]["ratio"],
                          "default": default.get(pairs[k]["ref"]) == k,
                          "series": {n: [None if not np.isfinite(v) else round(float(v), 3) for v in rep_series[k][:, c]]
                                     for c, n in enumerate(names)}}
                         for k, (a, b, _) in enumerate(reps_u)],
                "count_ref": len(reps_r), "count_user": len(reps_u),
                "count_paired": int(sum(p["paired"] for p in pairs)), "alternating": bool(any(p["mirror"] for p in pairs)),
            },
        })

    if seg_out:
        ref_span = seg_out[-1]["ref_end"] - seg_out[0]["ref_start"]
        user_span = seg_out[-1]["user_end"] - seg_out[0]["user_start"]
        tempo = user_span / ref_span if ref_span > 0 else float("nan")
        for s in seg_out:
            s["duration_ratio_vs_tempo"] = s["duration_ratio"] / tempo if tempo > 0 else None
    else:
        tempo, user_span = float("nan"), 0.0
    ok = np.isfinite(user_t_for_ref)
    if ok.sum() >= 2:   # outside repetition cores (transitions, rest): interpolate the time map linearly
        user_t_for_ref = np.interp(t_ref, t_ref[ok], np.maximum.accumulate(user_t_for_ref[ok]))
    finite = np.isfinite(cost_for_ref)
    return {
        "tempo_ratio": tempo,
        "user_span": [seg_out[0]["user_start"], seg_out[-1]["user_end"]] if seg_out else [0.0, 0.0],
        "align_cost": float(np.nanmean(cost_for_ref[finite])) if finite.any() else float("nan"),
        "segments": seg_out,
        "user_map": user_map,                      # [user_start, user_end, ref_start, ref_end] for every repetition
        "series": {"user_aligned": U_al, "delta": U_al - F_ref, "user_t_for_ref": user_t_for_ref,
                   "align_cost": cost_for_ref, "user_idx_for_ref": user_idx_for_ref, "weight": weight},
    }
