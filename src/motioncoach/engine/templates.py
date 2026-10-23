"""Repetition correspondence by reference-template search (chosen in docs/ALIGNMENT_STUDY.md, "M3fa").

Within each section:
1. the reference repetitions are templates: taken from the reference asset's annotation
   (start, end, hold per repetition), or detected from the distance-to-rest signal when absent;
2. every template (at several tempi) is searched in the practitioner's section with a slope-constrained
   subsequence DTW on standardised features + velocity; occurrences are taken greedily, best first,
   without overlapping (ends may share <= 15 % of a template), until the cost exceeds an adaptive limit;
   an occurrence must also last 0.6-1.6x the practitioner's own repetition period in the section
   (autocorrelation of a mirror-invariant signal), so a slow repetition is not taken as two half ones;
3. the search path is the in-repetition correspondence: local speed stays within 0.5-2x of the
   repetition's own tempo, so no frame absorbs a stretch of the other video;
4. key moments are intervals: the reference hold and the practitioner samples mapped onto it.

When the reference demonstrates both sides (two templates), a practitioner repetition is never matched
to the mirrored template, so a left repetition is shown against the reference's left repetition.
Sections annotated as not repeated (``reps: []``) are one interval aligned end to end.

Playback maps follow the "loop / jump" policy: inside paired repetitions the correspondence is used;
between them the shown video runs at ~1x and jumps (back to the template start, or over practitioner
repetitions that have no reference counterpart) instead of fast-forwarding or freezing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .align import mirror_invariant
from .compare import Segment
from .dtw import dtw_slope, occurrence_path, subseq_profile
from .features import lr_swap_features

DT = 0.1                       # analysis grid (10 fps); maps are computed on sample indices
SCALES = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0)
ALPHA = 0.5                    # weight of the velocity channels
MIRROR_PENALTY = 0.05
ABS_CAP, MASKED_COST = 2.5, 3.0
PERIOD_BAND = (0.6, 1.6)       # occurrence length / practitioner repetition period (profile: alignment.period_band)
MIN_PERIODS = 3                # the period is used only when the section holds at least this many of them


# ----------------------------------------------------------------------------- small helpers

def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def resample_rows(X: np.ndarray, n: int) -> np.ndarray:
    src, dst = np.linspace(0, 1, len(X)), np.linspace(0, 1, n)
    out = np.full((n, X.shape[1]), np.nan, dtype=np.float64)
    for c in range(X.shape[1]):
        ok = np.isfinite(X[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(dst, src[ok], X[ok, c])
    return out


def feat_cost(A: np.ndarray, B: np.ndarray, min_valid: int = 3) -> np.ndarray:
    """Mean |a-b| over mutually valid (standardised) columns, capped at 3; 2 if too few are valid."""
    Av, Bv = np.isfinite(A), np.isfinite(B)
    Az, Bz = np.nan_to_num(A), np.nan_to_num(B)
    both = Av[:, None] & Bv[None]
    n = both.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.minimum((np.abs(Az[:, None] - Bz[None]) * both).sum(axis=2) / n, 3.0)
    c[n < min(min_valid, A.shape[1])] = 2.0
    return c.astype(np.float32)


def velocity(Z: np.ndarray, sd: np.ndarray) -> np.ndarray:
    V = np.gradient(np.nan_to_num(Z, nan=0.0), axis=0) / DT
    V[~np.isfinite(Z)] = np.nan
    return V / sd


def to_path(u_float: np.ndarray, r_float: np.ndarray, ua: int, ub: int) -> tuple[np.ndarray, np.ndarray]:
    """For each practitioner sample ua..ub-1 the matched reference sample (non-decreasing)."""
    ui = np.arange(ua, ub)
    order = np.argsort(u_float, kind="stable")
    r = np.interp(ui, u_float[order], r_float[order])
    return ui, np.maximum.accumulate(np.round(r).astype(np.int64))


def segment_reps(sig: np.ndarray, prom: float = 0.3, min_period_s: float = 3.0,
                 min_amp: float = 0.5, dt: float = DT) -> list[tuple[int, int, int]]:
    """[(start, end_excl, peak)]: one repetition per prominent peak, split at the valleys between peaks."""
    n = len(sig)
    if n < 6 or n * dt < 2 * min_period_s:
        return [(0, n, int(np.nanargmax(sig)) if np.isfinite(sig).any() else n // 2)]
    x = _smooth(np.nan_to_num(sig, nan=np.nanmedian(sig)), int(round(1.0 / dt)))
    amp = np.percentile(x, 95) - np.percentile(x, 5)
    if amp < min_amp:
        return [(0, n, int(np.argmax(x)))]
    peaks, _ = find_peaks(x, prominence=prom * amp, distance=max(1, int(min_period_s / dt)))
    if len(peaks) == 0:
        return [(0, n, int(np.argmax(x)))]
    rest = np.percentile(x, 10)
    cuts = [int(a + np.argmin(x[a:b])) for a, b in zip(peaks[:-1], peaks[1:])]
    out = []
    for k, p in enumerate(peaks):
        lvl = rest + 0.2 * (x[p] - rest)
        if k == 0:
            s = p
            while s > 0 and x[s - 1] > lvl:
                s -= 1
        else:
            s = cuts[k - 1]
        if k == len(peaks) - 1:
            e = p
            while e < n - 1 and x[e + 1] > lvl:
                e += 1
            e += 1
        else:
            e = cuts[k]
        if e - s >= 5:
            out.append((s, e, int(p)))
    return out or [(0, n, int(np.argmax(x)))]


def hold_plateau(sig: np.ndarray, speed: np.ndarray, rep: tuple[int, int, int], frac: float = 0.25) -> tuple[int, int]:
    """Low-speed run near the repetition's peak (indices relative to ``sig``)."""
    a, b, p = rep
    x = _smooth(np.nan_to_num(sig[a:b], nan=np.nanmedian(sig[a:b])), 5)
    v = _smooth(np.nan_to_num(speed[a:b], nan=np.nanmedian(speed[a:b])), 5)
    p = int(np.clip(p - a, 0, len(x) - 1))
    thr_x = x[p] - frac * max(1e-6, x[p] - x.min())
    i0, i1 = p, p + 1
    while i0 > 0 and x[i0 - 1] >= thr_x:
        i0 -= 1
    while i1 < len(x) and x[i1] >= thr_x:
        i1 += 1
    thr_v = 0.6 * np.median(v)
    best, cur, bs = (0, 0), 0, i0
    for k, ok in enumerate(v[i0:i1] <= thr_v):
        if ok:
            if cur == 0:
                bs = i0 + k
            cur += 1
            if cur > best[1] - best[0]:
                best = (bs, bs + cur)
        else:
            cur = 0
    h0, h1 = best if best[1] - best[0] >= 3 else (i0, i1)
    if h1 - h0 < 3:
        h0, h1 = max(0, p - 1), min(len(x), p + 2)
    return a + h0, a + h1


# ----------------------------------------------------------------------------- representation + search

class FeatureRep:
    """Standardised features + velocity; the "mirror" variant is the practitioner with left/right exchanged."""
    mirror_penalty, abs_cap, masked_cost = MIRROR_PENALTY, ABS_CAP, MASKED_COST

    def __init__(self, F_ref: np.ndarray, F_user: np.ndarray, names: list[str], scales: np.ndarray, alpha: float = ALPHA):
        Zr = F_ref / scales
        sd = np.nanstd(np.gradient(np.nan_to_num(Zr), axis=0) / DT, axis=0) + 1e-3
        self.Zr = Zr
        self.R = np.concatenate([Zr, alpha * velocity(Zr, sd)], axis=1)
        self.Zu, self.U = {}, {}
        for mirror, F in ((False, F_user), (True, lr_swap_features(F_user, names))):
            Zu = F / scales
            self.Zu[mirror] = Zu
            self.U[mirror] = np.concatenate([Zu, alpha * velocity(Zu, sd)], axis=1)
        self.G, self.g_sc = mirror_invariant(F_ref, names, scales)

    def ref_rows(self, a, b, n):
        return resample_rows(self.R[a:b], n)

    def cost(self, X, u0, u1, mirror, resample=None):
        U = self.U[mirror][u0:u1]
        return feat_cost(X, resample_rows(U, resample) if resample else U)

    def rest_signal(self, a, b):
        """Distance from the section's starting pose (mirror invariant), and speed."""
        Z = self.G[a:b] / self.g_sc
        rest = np.nanmedian(Z[:10], axis=0)
        d = np.sqrt(np.nanmean((Z - rest) ** 2, axis=1))
        v = np.linalg.norm(np.nan_to_num(np.gradient(Z, axis=0)), axis=1) / DT
        return d, v


def practice_period(F: np.ndarray, names: list[str], scales: np.ndarray, lo: int, hi: int,
                    prior: float | None = None) -> float | None:
    """Practitioner repetition period (samples) in one section, or None when not clearly periodic.

    Shortest strong autocorrelation lag (lo..hi samples) of the mirror-invariant distance from the section's
    first pose: a left and a right repetition look alike in it, so one repetition is one period; when a full
    left+right cycle correlates as strongly, the lag closest to ``prior`` (template length) is taken. Sections
    shorter than ``MIN_PERIODS`` periods (e.g. two repetitions) give no period: autocorrelation is unreliable there."""
    G, sc = mirror_invariant(F, names, scales)
    if len(G) < max(12, 2 * lo) or hi <= lo:
        return None
    Z = G / sc
    with np.errstate(all="ignore"):
        d = np.sqrt(np.nanmean((Z - np.nanmedian(Z[:10], axis=0)) ** 2, axis=1))
    if not np.isfinite(d).any():
        return None
    x = _smooth(np.nan_to_num(d - np.nanmean(d)), int(round(1.0 / DT)))
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / max(ac[0], 1e-9)
    hi = min(hi, len(ac) - 1)
    pk, _ = find_peaks(ac[lo:hi + 1])
    if not len(pk):
        return None
    pk = lo + pk
    top = ac[pk].max()
    if top <= 0.2:
        return None
    strong = pk[ac[pk] >= 0.6 * top]
    # several strong lags (one repetition, a left+right cycle): the one closest to the template length
    period = float(strong[np.argmin(np.abs(np.log(strong / prior)))] if prior else strong[0])
    return period if len(G) >= MIN_PERIODS * period else None   # too few cycles to trust the estimate


COVERAGE_GUARD = 0.85          # period-constrained search must cover this share of the unconstrained one (I11)


def search_reps(rep, reps_r: list[tuple[int, int, int]], u0: int, u1: int, expected: int, F_user: np.ndarray,
                names: list[str], scales: np.ndarray,
                period_band: tuple[float, float] | None = PERIOD_BAND) -> tuple[list[dict], float | None]:
    """``search`` with the practitioner-period constraint, falling back to the unconstrained search when the
    constraint loses much of the section (a wrong period estimate). Returns (occurrences, period used)."""
    free = search(rep, reps_r, u0, u1, expected)
    if not period_band:
        return free, None
    n0 = [b - a for a, b, _ in reps_r]
    period = practice_period(F_user[u0:u1], names, scales, int(0.5 * min(n0)), int(2.5 * max(n0)),
                             prior=float(np.median(n0)))
    if period is None:
        return free, None
    held = search(rep, reps_r, u0, u1, expected, period, period_band)
    cover = lambda occ: sum(f["end"] - f["start"] for f in occ)   # noqa: E731
    if cover(held) < COVERAGE_GUARD * cover(free):
        return free, None
    return held, period


def search(rep, reps_r: list[tuple[int, int, int]], u0: int, u1: int, expected: int,
           period: float | None = None, period_band: tuple[float, float] | None = PERIOD_BAND) -> list[dict]:
    """Greedy non-overlapping template occurrences in practitioner samples u0..u1-1 (reps_r absolute).
    ``period``: practitioner repetition period (samples); occurrences outside ``period_band`` x period are refused."""
    m = u1 - u0
    masked = np.zeros(m, dtype=bool)
    found, first = [], None
    cache = []                                   # cost matrices do not change between iterations
    for j, (ra, rb, _) in enumerate(reps_r):
        for s in SCALES:
            n = max(4, int(round((rb - ra) * s)))
            if n <= m:
                X = rep.ref_rows(ra, rb, n)
                for mirror in ((False, True) if len(reps_r) == 1 else (False,)):   # both sides demonstrated: no mirroring
                    cache.append((j, s, n, mirror, rep.cost(X, u0, u1, mirror)))
    for _ in range(max(3, 3 * expected + 2)):
        best = None
        cm = np.r_[0, np.cumsum(masked)]
        for j, s, n, mirror, C0 in cache:
            n0 = reps_r[j][1] - reps_r[j][0]
            C = C0.copy()
            C[:, masked] = rep.masked_cost
            cost, start, B = subseq_profile(C)
            ends = np.arange(m)
            tol = int(0.15 * n0)                # rest-pose ends may overlap a neighbour by <= 15 % of the template
            lo, hi = np.minimum(start + tol, ends + 1), np.maximum(ends + 1 - tol, start)
            clean = ((cm[ends + 1] - cm[start]) <= 2 * tol) & ((cm[np.maximum(hi, lo)] - cm[lo]) == 0)
            length = ends - start + 1
            ok = clean & (length >= 0.5 * n0) & (length <= 2.5 * n0) & np.isfinite(cost)
            if period and period_band:
                ok &= (length >= period_band[0] * period) & (length <= period_band[1] * period)
            if not ok.any():
                continue
            c = np.where(ok, cost, np.inf) + (rep.mirror_penalty if mirror else 0)
            e = int(np.argmin(c))
            if best is None or c[e] < best[0]:
                best = (float(c[e]), j, mirror, s, n, int(start[e]), e, B)
        if best is None:
            break
        c, j, mirror, s, n, a, e, B = best
        if first is None:
            first = c
        if c > min(rep.abs_cap, max(1.6 * first, first + 0.08 * rep.abs_cap / 0.45)):
            break
        ti, ui = occurrence_path(B, e)
        ra, rb, _ = reps_r[j]
        found.append({"cost": c, "ref": j, "mirror": mirror, "scale": s, "start": a, "end": e + 1,
                      "u": ui.astype(np.float64), "r": ra + ti / max(1, n - 1) * (rb - ra - 1)})
        masked[a:e + 1] = True
    found.sort(key=lambda f: f["start"])
    for f0, f1 in zip(found[:-1], found[1:]):  # resolve end overlaps at their midpoint
        if f0["end"] > f1["start"]:
            mid = (f0["end"] + f1["start"]) // 2
            f0["end"], f1["start"] = mid, mid
    for f in found:
        keep = (f["u"] >= f["start"]) & (f["u"] < f["end"])
        f["u"], f["r"] = f["u"][keep], f["r"][keep]
    return [f for f in found if f["end"] - f["start"] >= 5 and len(f["u"]) >= 2]


# ----------------------------------------------------------------------------- playback maps

def fill_map(anchors: list[tuple[int, np.ndarray]], X0: int, X1: int, Y0: float, Y1: float) -> np.ndarray:
    """anchors: (x_start, y for x_start..). Gaps: linear when the implied speed is within [0.25, 4],
    otherwise ~1x out of the previous anchor, a jump, and ~1x into the next one."""
    y = np.full(X1 - X0, np.nan)
    pts = [(X0, np.array([Y0]))] + sorted(anchors, key=lambda a: a[0]) + [(X1 - 1, np.array([Y1]))]
    for xs, ys in pts:
        lo, hi = max(xs, X0), min(xs + len(ys), X1)
        if hi > lo:
            y[lo - X0:hi - X0] = ys[lo - xs:hi - xs]
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        g0, g1 = xa + len(ya), xb
        if g1 <= g0:
            continue
        y_from, y_to = float(ya[-1]), float(yb[0])
        speed = (y_to - y_from) / (g1 - g0 + 1)
        idx = np.arange(g0, g1)
        if 0.25 <= speed <= 4.0:
            y[idx - X0] = y_from + (idx - g0 + 1) * speed
        else:
            half = (g0 + g1) // 2
            y[np.arange(g0, half) - X0] = y_from + (np.arange(g0, half) - g0 + 1)
            y[np.arange(half, g1) - X0] = y_to - (g1 - np.arange(half, g1))
    return y


def _drop_overlaps(anchors):
    out, end = [], -1
    for xs, ys in sorted(anchors, key=lambda a: a[0]):
        if xs >= end:
            out.append((xs, ys))
            end = xs + len(ys)
    return out


def playback_maps(paths: list[tuple[np.ndarray, np.ndarray, bool]], U0: int, U1: int, R0: int, R1: int):
    """paths: (user idx, ref idx, is_default) per paired repetition. Returns (u2r over U0..U1-1, r2u over R0..R1-1)."""
    ua, ra = [], []
    for ui, ri, default in paths:
        ua.append((int(ui[0]), ri.astype(np.float64)))
        if default:
            rr = np.arange(int(ri.min()), int(ri.max()) + 1)
            order = np.argsort(ri, kind="stable")
            ra.append((int(rr[0]), np.interp(rr, ri[order].astype(np.float64), ui[order].astype(np.float64))))
    u2r = fill_map(_drop_overlaps(ua), U0, U1, R0, R1 - 1)
    r2u = fill_map(_drop_overlaps(ra), R0, R1, U0, U1 - 1)
    return np.clip(u2r, R0, R1 - 1), np.clip(r2u, U0, U1 - 1)


# ----------------------------------------------------------------------------- production entry point

def _template(seg_reps, r0: int, r1: int, rep: FeatureRep, t_ref: np.ndarray, dt: float):
    """(reps [(a, b, peak)] absolute, holds [(h0, h1)] absolute, repetitive, source)."""
    n = r1 - r0
    if seg_reps is not None and len(seg_reps) == 0:
        d, v = rep.rest_signal(r0, r1)
        whole = (0, n, int(np.argmax(np.nan_to_num(d))))
        h = hold_plateau(d, v, whole)
        return [(r0, r1, r0 + whole[2])], [(r0 + h[0], r0 + h[1])], False, "annotated"
    if seg_reps:
        idx = lambda t: int(np.clip(np.searchsorted(t_ref, t - 1e-6), r0, r1))   # noqa: E731
        reps = [(idx(a), idx(b), (idx(ha) + idx(hb)) // 2) for a, b, ha, hb in seg_reps]
        holds = [(idx(ha), max(idx(ha) + 1, idx(hb))) for _, _, ha, hb in seg_reps]
        return reps, holds, True, "annotated"
    d, v = rep.rest_signal(r0, r1)
    auto = segment_reps(d, dt=dt)
    return ([(r0 + a, r0 + b, r0 + p) for a, b, p in auto], [tuple(r0 + x for x in hold_plateau(d, v, r)) for r in auto],
            True, "auto")


def compare_by_templates(t_ref, F_ref, t_user, F_user, segments: list[Segment], user_bounds: list[tuple[int, int]],
                         names: list[str], scales: np.ndarray, ref_reps: dict | None = None,
                         period_band: tuple[float, float] | None = PERIOD_BAND) -> dict:
    """Same output shape as ``reps.compare_by_reps`` plus ``playback`` maps.

    ``ref_reps``: section id -> [[start, end, hold_start, hold_end], ...] seconds on the reference video
    ([] = not a repeated move); sections without an entry use detected templates.
    Both sides must be sampled at the same rate (the extractor's absolute k / fps grid).
    """
    ref_reps = ref_reps or {}
    dt = float(np.median(np.diff(t_ref))) if len(t_ref) > 1 else DT
    N, K = F_ref.shape
    rep = FeatureRep(F_ref, F_user, names, scales)
    U_al = np.full((N, K), np.nan)
    user_idx_for_ref = np.full(N, -1, dtype=np.int64)
    weight = np.zeros(N)
    cost_for_ref = np.full(N, np.nan)
    seg_out, user_map, paths = [], [], []
    ti = lambda i, t: float(t[min(max(i, 0), len(t) - 1)])   # noqa: E731
    for seg, (u0, u1) in zip(segments, user_bounds):
        ridx = np.flatnonzero((t_ref >= seg.start) & (t_ref < seg.end))
        if len(ridx) < 3 or u1 - u0 < 3:
            continue
        r0, r1 = int(ridx[0]), int(ridx[-1]) + 1
        reps_r, holds_r, repetitive, source = _template(ref_reps.get(seg.id), r0, r1, rep, t_ref, dt)
        occ, period = [], None
        if repetitive:
            expected = max(1, int(round((u1 - u0) / max(1, r1 - r0) * len(reps_r))))
            occ, period = search_reps(rep, reps_r, u0, u1, expected, F_user, names, scales, period_band)
        else:                                       # one interval, aligned end to end
            L = max(4, int(round((r1 - r0 + u1 - u0) / 2)))
            ri, ui, c = dtw_slope(rep.cost(rep.ref_rows(r0, r1, L), u0, u1, False, resample=L))
            occ = [{"cost": c, "ref": 0, "mirror": False, "start": 0, "end": u1 - u0,
                    "u": ui / (L - 1) * (u1 - u0 - 1), "r": r0 + ri / (L - 1) * (r1 - r0 - 1)}]
        users = []
        for f in occ:
            ua, ub = u0 + f["start"], u0 + f["end"]
            ui, ri = to_path(u0 + f["u"], f["r"], ua, ub)
            with np.errstate(all="ignore"):   # position-only cost along the path (tolerance units)
                pc = float(np.nanmean(np.minimum(np.abs(rep.Zr[ri] - rep.Zu[f["mirror"]][ui]), 3.0)))
            h0, h1 = holds_r[f["ref"]]
            inh = np.flatnonzero((ri >= h0) & (ri < h1))
            users.append({"ua": ua, "ub": ub, "ui": ui, "ri": ri, "ref": f["ref"], "mirror": f["mirror"], "cost": pc,
                          "hold": (int(ui[inh[0]]), int(ui[inh[-1]]) + 1) if len(inh) else None})
        # default practitioner repetition per reference repetition: lowest cost, temporal order
        default, last = {}, -1
        for j in range(len(reps_r)):
            cand = [k for k, u in enumerate(users) if u["ref"] == j and u["ua"] >= last]
            if cand:
                k = min(cand, key=lambda k: users[k]["cost"])
                default[j] = k
                last = users[k]["ub"]
        rep_series = []
        for k, u in enumerate(users):
            ui, ri = u["ui"], u["ri"]
            Fu_k = lr_swap_features(F_user, names) if u["mirror"] else F_user
            ra, rb = reps_r[u["ref"]][0], reps_r[u["ref"]][1]
            # practitioner values on the reference repetition's samples (inverse path, averaged)
            S = np.full((rb - ra, K), np.nan)
            for r in range(ra, rb):
                m = ri == r
                if m.any():
                    with np.errstate(all="ignore"):
                        S[r - ra] = np.nanmean(Fu_k[ui[m]], axis=0)
            if np.isfinite(S).any():
                for c in range(K):
                    ok = np.isfinite(S[:, c])
                    if 2 <= ok.sum() < len(S):
                        S[:, c] = np.interp(np.arange(len(S)), np.flatnonzero(ok), S[ok, c], left=np.nan, right=np.nan)
            rep_series.append(S)
            is_def = default.get(u["ref"]) == k
            paths.append((ui, ri, is_def))
            user_map.append([ti(u["ua"], t_user), ti(u["ub"] - 1, t_user), ti(ra, t_ref), ti(rb - 1, t_ref)])
            weight[ra:rb] += (u["ub"] - u["ua"]) / max(1, rb - ra)
            if is_def:
                U_al[ra:rb] = S
                inv = np.full(rb - ra, -1, dtype=np.int64)
                for r in range(ra, rb):
                    m = np.flatnonzero(ri == r)
                    if len(m):
                        inv[r - ra] = ui[m[len(m) // 2]]
                good = inv >= 0
                if good.any():
                    inv[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), inv[good]).astype(np.int64)
                user_idx_for_ref[ra:rb] = inv
                with np.errstate(all="ignore"):
                    cost_for_ref[ra:rb] = np.nanmean(np.abs(S - F_ref[ra:rb]) / scales, axis=1)

        # interval-to-interval key moments: median over the reference hold vs over the samples mapped onto it
        hold_d, hold_v, hold_r = [], [], []
        for u, S in zip(users, rep_series):
            ra = reps_r[u["ref"]][0]
            h0, h1 = holds_r[u["ref"]]
            hu = S[h0 - ra:h1 - ra]
            with np.errstate(all="ignore"):
                hold_r.append(np.nanmedian(F_ref[h0:h1], axis=0))
                hold_d.append(np.nanmedian(hu, axis=0) - hold_r[-1])
            hold_v.append(np.isfinite(F_ref[h0:h1]).mean(axis=0) * np.isfinite(hu).mean(axis=0))
        HD = np.array(hold_d) if hold_d else np.zeros((0, K))
        HV = np.array(hold_v) if hold_v else np.zeros((0, K))
        core = np.zeros(r1 - r0, bool)
        for j in default:
            core[reps_r[j][0] - r0:reps_r[j][1] - r0] = True
        delta = U_al[r0:r1] - F_ref[r0:r1]
        feats = {}
        for c, nme in enumerate(names):
            d = delta[:, c][core]
            ok = np.isfinite(d)
            hd = HD[:, c][np.isfinite(HD[:, c]) & (HV[:, c] >= 0.5)] if len(HD) else np.zeros(0)
            feats[nme] = {
                "coverage": float(ok.mean()) if len(ok) else 0.0,
                "mean_delta": float(np.mean(d[ok])) if ok.any() else None,
                "median_abs": float(np.median(np.abs(d[ok]))) if ok.any() else None,
                "p90_abs": float(np.percentile(np.abs(d[ok]), 90)) if ok.any() else None,
                "hold_coverage": float(len(hd) / max(1, len(users))),
                "hold_valid": int(len(hd)), "hold_n": int(len(users)),
                "hold_mean_delta": float(np.median(hd)) if len(hd) else None,
                "hold_median_abs": float(np.median(np.abs(hd))) if len(hd) else None,
                "per_rep": [None if not np.isfinite(v) else float(v) for v in (HD[:, c] if len(HD) else [])],
                # the reference value at the hold of each practitioner repetition's template (same order)
                "per_rep_ref": [None if not np.isfinite(r[c]) else float(r[c]) for r in hold_r],
                "ref_mean": float(np.nanmean(F_ref[r0:r1, c])) if np.isfinite(F_ref[r0:r1, c]).any() else None,
                "user_mean": float(np.nanmean(U_al[r0:r1, c])) if np.isfinite(U_al[r0:r1, c]).any() else None,
            }
        ut0, ut1 = ti(u0, t_user), ti(u1 - 1, t_user)
        seg_out.append({
            "id": seg.id, "name": seg.name, "ref_start": seg.start, "ref_end": seg.end,
            "user_start": ut0, "user_end": ut1, "duration_ratio": (ut1 - ut0) / max(1e-6, seg.end - seg.start),
            "duration_ratio_vs_tempo": None,
            "align_cost": float(np.mean([u["cost"] for u in users])) if users else float("inf"),
            "hold_times": [ti((a + b) // 2, t_ref) for a, b in holds_r],
            "practice_period_s": None if period is None else period * dt,
            "features": feats,
            "reps": {
                "ref": [{"start": ti(a, t_ref), "end": ti(b - 1, t_ref), "hold": [ti(h0, t_ref), ti(h1 - 1, t_ref)]}
                        for (a, b, _), (h0, h1) in zip(reps_r, holds_r)],
                "user": [{"start": ti(u["ua"], t_user), "end": ti(u["ub"] - 1, t_user),
                          "hold": None if u["hold"] is None else [ti(u["hold"][0], t_user), ti(u["hold"][1] - 1, t_user)],
                          "ref": u["ref"], "cost": u["cost"], "mirror": bool(u["mirror"]), "paired": True,
                          "duration_ratio": (u["ub"] - u["ua"]) / max(1, reps_r[u["ref"]][1] - reps_r[u["ref"]][0]),
                          "default": default.get(u["ref"]) == k,
                          "series": {n: [None if not np.isfinite(v) else round(float(v), 3) for v in S[:, c]]
                                     for c, n in enumerate(names)}}
                         for k, (u, S) in enumerate(zip(users, rep_series))],
                "count_ref": len(reps_r) if repetitive else 0, "count_user": len(users) if repetitive else 0,
                "count_paired": len(users) if repetitive else 0, "alternating": len(reps_r) == 2,
                "repeated": repetitive, "template": source,
            },
        })
    if seg_out:
        ref_span = seg_out[-1]["ref_end"] - seg_out[0]["ref_start"]
        user_span = seg_out[-1]["user_end"] - seg_out[0]["user_start"]
        tempo = user_span / ref_span if ref_span > 0 else float("nan")
        for s in seg_out:
            s["duration_ratio_vs_tempo"] = s["duration_ratio"] / tempo if tempo > 0 else None
        U0, U1 = user_bounds[0][0], user_bounds[len(seg_out) - 1][1]
        R0 = int(np.flatnonzero(t_ref >= segments[0].start)[0])
        R1 = int(np.flatnonzero(t_ref < segments[len(seg_out) - 1].end)[-1]) + 1
        u2r, r2u = playback_maps(paths, U0, U1, R0, R1)
    else:
        tempo, user_span = float("nan"), 0.0
        U0 = U1 = R0 = R1 = 0
        u2r = r2u = np.zeros(0)
    user_t_for_ref = np.full(N, np.nan)
    user_t_for_ref[R0:R1] = np.interp(r2u, np.arange(len(t_user)), t_user) if len(r2u) else np.nan
    finite = np.isfinite(cost_for_ref)
    return {
        "tempo_ratio": tempo,
        "user_span": [seg_out[0]["user_start"], seg_out[-1]["user_end"]] if seg_out else [0.0, 0.0],
        "align_cost": float(np.nanmean(cost_for_ref[finite])) if finite.any() else float("nan"),
        "segments": seg_out,
        "user_map": user_map,
        "playback": {   # loop / jump policy, one value per analysis sample
            "user_t0": ti(U0, t_user), "dt": dt,
            "ref_t_for_user": [round(float(np.interp(x, np.arange(len(t_ref)), t_ref)), 2) for x in u2r],
            "ref_t0": ti(R0, t_ref),
            "user_t_for_ref": [round(float(np.interp(x, np.arange(len(t_user)), t_user)), 2) for x in r2u],
        },
        "series": {"user_aligned": U_al, "delta": U_al - F_ref, "user_t_for_ref": user_t_for_ref,
                   "align_cost": cost_for_ref, "user_idx_for_ref": user_idx_for_ref, "weight": weight},
    }
