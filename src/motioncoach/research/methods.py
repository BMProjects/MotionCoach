"""Correspondence methods under comparison. All return the same structure (``SectionOut`` per section).

M0  frame-level DTW inside each section (engine 0.3, the "first version"); no repetition notion.
M1  production 0.4: PCA cycle signal, peak cores, phase-linear pairing (``engine.reps.compare_by_reps``).
M2  scheme A (explicit): mirror-invariant cycle signal, period-aware repetition split at the valleys,
    phases (approach / hold / return) from a low-speed plateau, phase-wise slope-constrained DTW on
    position + velocity features, mirror-side pairing.
M3  gait-style: pose maps; reference repetitions from the distance-to-rest signal; practitioner
    repetitions found by searching the reference templates (both sides, several tempi) in the stream
    with slope-constrained subsequence DTW; the search path is the in-repetition correspondence.
M3f M3's matching with features instead of pose maps (separates representation from matching).
M3ca / M3fca  the same matching on same-body keypoint coordinates, alone or together with the features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from ..engine.align import align, mirror_invariant, smooth_nan, two_stage_align
from ..engine.compare import Segment
from ..engine.reps import compare_by_reps
from . import posemap
from .data import Pair, section_index, swapped
from ..engine.dtw import dtw_slope, hsmm_segment
from ..engine.templates import SCALES, _smooth, feat_cost, hold_plateau, search_reps, segment_reps  # noqa: F401  (shared with production)
from ..engine.templates import resample_rows as _resample_rows
from ..engine.templates import search as _search
from ..engine.templates import to_path as _to_path

DT = 0.1   # all sequences are sampled on the 10 fps grid


@dataclass
class Rep:
    start: int                       # sample index (inclusive) on its own side
    end: int                         # exclusive
    hold: tuple[int, int] | None = None
    ref: int | None = None           # user reps: index into the section's ref_reps
    mirror: bool = False
    cost: float = float("nan")
    paired: bool = False
    default: bool = False
    path: tuple[np.ndarray, np.ndarray] | None = None   # (user idx, ref idx), absolute, non-decreasing


@dataclass
class SectionOut:
    id: str
    r0: int
    r1: int
    u0: int
    u1: int
    ref_reps: list[Rep] = field(default_factory=list)
    user_reps: list[Rep] = field(default_factory=list)
    has_reps: bool = True
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- section bounds

def bounds_gt(pair: Pair) -> list[tuple[str, int, int, int, int]]:
    out = []
    for sid, a, b in pair.ref_sections:
        r0, r1 = section_index(pair.ref.t, a, b)
        if pair.gt is not None:
            g = next(s for s in pair.gt["sections"] if s["id"] == sid)
            u0, u1 = section_index(pair.user.t, *g["user"])
        else:
            u0, u1 = section_index(pair.user.t, a + pair.gt_offset, b + pair.gt_offset)
        out.append((sid, r0, r1, u0, u1))
    return out


def bounds_coarse(pair: Pair, filt: str) -> list[tuple[str, int, int, int, int]]:
    """Production stage 1 (mirror-invariant coarse DTW with skip cost)."""
    res = two_stage_align(pair.ref.F[filt], pair.user.F[filt], pair.ref.t, pair.names, pair.scales,
                          [(a, b) for _, a, b in pair.ref_sections], swap_mode="no",
                          coarse_window_s=1.0 if pair.name == "baduanjin" else 0.5, skip_cost=2.0)
    out = []
    for (sid, a, b), ev in zip(pair.ref_sections, res.segment_lr):
        out.append((sid, ev["ref"][0], ev["ref"][1], ev["user"][0], ev["user"][1]))
    return out


def bounds_hsmm(pair: Pair, filt: str, sub: int = 2) -> list[tuple[str, int, int, int, int]]:
    """Scheme C: ordered explicit-duration segmentation, nearest-neighbour emission (5 fps)."""
    G_r, g_sc = mirror_invariant(pair.ref.F[filt], pair.names, pair.scales)
    G_u, _ = mirror_invariant(pair.user.F[filt], pair.names, pair.scales)
    G_r, G_u = smooth_nan(G_r, 10)[::sub] / g_sc, smooth_nan(G_u, 10)[::sub] / g_sc
    secs = [(sid, *section_index(pair.ref.t[::sub], a, b)) for sid, a, b in pair.ref_sections]
    emis = []
    Uz, Uv = np.nan_to_num(G_u), np.isfinite(G_u)
    for _, r0, r1 in secs:
        R = G_r[r0:r1]
        Rz, Rv = np.nan_to_num(R), np.isfinite(R)
        both = Rv[:, None] & Uv[None]
        d = (np.abs(Rz[:, None] - Uz[None]) * both).sum(axis=2) / np.maximum(1, both.sum(axis=2))
        d[both.sum(axis=2) < 3] = 2.0
        k = min(3, len(R))
        emis.append(np.sort(np.minimum(d, 3.0), axis=0)[:k].mean(axis=0))
    emis = np.asarray(emis)
    dur = np.array([r1 - r0 for _, r0, r1 in secs], dtype=np.float64)
    tempo = len(G_u) / max(1, secs[-1][2] - secs[0][1])
    seg = hsmm_segment(emis, dur, tempo, skip=1.0)
    out = []
    for (sid, a, b), (s, e) in zip(pair.ref_sections, seg):
        r0, r1 = section_index(pair.ref.t, a, b)
        out.append((sid, r0, r1, min(len(pair.user.t) - 1, s * sub), min(len(pair.user.t), e * sub)))
    return out


# ----------------------------------------------------------------------------- shared helpers

def _std(F: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return F / scales


def _velocity(X: np.ndarray, sd: np.ndarray) -> np.ndarray:
    V = np.gradient(np.nan_to_num(X, nan=np.nanmean(X, axis=0)), axis=0) / DT
    V[~np.isfinite(X)] = np.nan
    return V / sd


def period_of(sig: np.ndarray, lo_s: float = 4.0, hi_s: float = 30.0) -> float | None:
    x = _smooth(np.nan_to_num(sig - np.nanmean(sig)), 10)
    if len(x) < 2 * int(lo_s / DT):
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / max(ac[0], 1e-9)
    lo, hi = int(lo_s / DT), min(len(ac) - 1, int(hi_s / DT))
    if hi <= lo:
        return None
    pk, _ = find_peaks(ac[lo:hi + 1])
    if not len(pk):
        return None
    pk = lo + pk
    top = ac[pk].max()
    if top <= 0.2:
        return None
    # shortest strong lag: for alternating left/right moves the full L+R cycle often correlates best,
    # but one repetition is the half cycle
    return float(pk[np.flatnonzero(ac[pk] >= 0.6 * top)[0]] * DT)


def _pick_defaults(sec: SectionOut) -> None:
    last = -1
    for j in range(len(sec.ref_reps)):
        cand = [u for u in sec.user_reps if u.paired and u.ref == j and u.start >= last]
        if cand:
            best = min(cand, key=lambda u: u.cost)
            best.default = True
            last = best.end


def _linear_path(ua: int, ub: int, ra: int, rb: int) -> tuple[np.ndarray, np.ndarray]:
    ui = np.arange(ua, ub)
    ri = np.round(ra + (ui - ua) / max(1, ub - ua - 1) * (rb - ra - 1)).astype(np.int64)
    return ui, ri


# ----------------------------------------------------------------------------- M0

def method_m0(pair: Pair, bounds, filt: str = "sg") -> list[SectionOut]:
    Fr, Fu = pair.ref.F[filt], pair.user.F[filt]
    out = []
    for sid, r0, r1, u0, u1 in bounds:
        sec = SectionOut(sid, r0, r1, u0, u1, has_reps=False)
        if r1 - r0 >= 3 and u1 - u0 >= 3:
            al = align(Fr[r0:r1], Fu[u0:u1], pair.scales, 0.02, skip_cost=1e4)
            sec.ref_reps = [Rep(r0, r1)]
            sec.user_reps = [Rep(u0, u1, ref=0, paired=True, default=True, cost=al.total_cost,
                                 path=(al.user_idx + u0, al.ref_idx + r0))]
        out.append(sec)
    return out


# ----------------------------------------------------------------------------- M1 (production)

def method_m1(pair: Pair, bounds, filt: str = "sg") -> list[SectionOut]:
    Fr, Fu = pair.ref.F[filt], pair.user.F[filt]
    neutral = None
    if pair.name == "baduanjin":
        with np.errstate(all="ignore"):
            neutral = np.nanmedian(Fr[pair.ref.t < 3.0], axis=0)
    segs = [Segment(sid, "", float(pair.ref.t[r0]), float(pair.ref.t[r1 - 1]) + 0.05) for sid, r0, r1, _, _ in bounds]
    cmp = compare_by_reps(pair.ref.t, Fr, pair.user.t, Fu, segs, [(u0, u1) for *_, u0, u1 in bounds],
                          pair.names, pair.scales, neutral=neutral)
    by_id = {s["id"]: s for s in cmp["segments"]}
    ti = lambda t, side: int(round(t / DT))   # noqa: E731  (10 fps grid starting at 0)
    out = []
    for sid, r0, r1, u0, u1 in bounds:
        sec = SectionOut(sid, r0, r1, u0, u1)
        s = by_id.get(sid)
        if s is not None:
            R = s["reps"]
            sec.ref_reps = [Rep(ti(q["start"], 0), ti(q["end"], 0) + 1, (ti(q["hold"][0], 0), ti(q["hold"][1], 0) + 1)) for q in R["ref"]]
            for q in R["user"]:
                u = Rep(ti(q["start"], 1), ti(q["end"], 1) + 1, (ti(q["hold"][0], 1), ti(q["hold"][1], 1) + 1),
                        ref=q["ref"], mirror=q["mirror"], cost=q["cost"], paired=q["paired"], default=q["default"])
                if u.paired:
                    rr = sec.ref_reps[u.ref]
                    u.path = _linear_path(u.start, u.end, rr.start, rr.end)
                sec.user_reps.append(u)
        out.append(sec)
    out_extra = {"user_t_for_ref": cmp["series"]["user_t_for_ref"], "user_map": cmp["user_map"]}
    for sec in out:
        sec.extra = out_extra
    return out


# ----------------------------------------------------------------------------- reference template

def ref_template(pair: Pair, sid: str, r0: int, r1: int, auto_reps, auto_holds, template: str):
    """Reference repetitions + holds, relative to r0. ``template``: "auto" (detected) or "annotated"
    (the reference asset's own annotation, as a coach would mark it once per reference video).
    Returns (reps [(a, b, peak)], holds [(h0, h1)], repetitive)."""
    if template != "annotated" or pair.gt is None:
        return auto_reps, auto_holds, True
    g = next(s for s in pair.gt["sections"] if s["id"] == sid)
    if not g.get("ref_reps"):
        return [(0, r1 - r0, (r1 - r0) // 2)], None, False
    n = r1 - r0
    idx = lambda t: int(np.clip(round(t / DT) - r0, 0, n))   # noqa: E731
    reps = [(idx(a), idx(b), (idx(ha) + idx(hb)) // 2) for a, b, ha, hb in g["ref_reps"]]
    holds = [(idx(ha), max(idx(ha) + 1, idx(hb))) for _, _, ha, hb in g["ref_reps"]]
    return reps, holds, True


# ----------------------------------------------------------------------------- M2 (scheme A)

def _cycle_mi(pair: Pair, filt: str, r0, r1, u0, u1):
    """Mirror-invariant first principal component of the reference section; start of section = low."""
    G_r, g_sc = mirror_invariant(pair.ref.F[filt][r0:r1], pair.names, pair.scales)
    G_u, _ = mirror_invariant(pair.user.F[filt][u0:u1], pair.names, pair.scales)
    Zr, Zu = G_r / g_sc, G_u / g_sc
    mu = np.nanmean(Zr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0)
    A, B = np.nan_to_num(Zr - mu), np.nan_to_num(Zu - mu)
    w = np.linalg.svd(A, full_matrices=False)[2][0]
    sr, su = A @ w, B @ w
    if np.median(sr[:10]) > np.median(sr):
        sr, su = -sr, -su
    vr = np.linalg.norm(np.nan_to_num(np.gradient(Zr, axis=0)), axis=1) / DT
    vu = np.linalg.norm(np.nan_to_num(np.gradient(Zu, axis=0)), axis=1) / DT
    return sr, su, vr, vu


def _phases(rep: tuple[int, int], hold: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = rep
    h0, h1 = hold
    parts = [(a, h0), (h0, h1), (h1, b)]
    merged = []
    for p in parts:
        if merged and (p[1] - p[0] < 3 or merged[-1][1] - merged[-1][0] < 3):
            merged[-1] = (merged[-1][0], p[1])
        else:
            merged.append(p)
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < 3:
        last = merged.pop()
        merged[-1] = (merged[-1][0], last[1])
    return merged


def _phase_dtw(Xr, Xu, ref_ph, user_ph):
    """Concatenate slope-constrained DTW over corresponding phases. Returns (u idx, r idx, mean cost)."""
    us, rs, costs, w = [], [], [], []
    whole = (ref_ph[-1][1] - ref_ph[0][0], user_ph[-1][1] - user_ph[0][0])
    R = whole[1] / max(1, whole[0])
    odd = len(ref_ph) != len(user_ph) or any(
        not (1 / 3 <= ((ub - ua) / max(1, rb - ra)) / R <= 3) for (ra, rb), (ua, ub) in zip(ref_ph, user_ph))
    if odd:                                  # phase structures disagree: align the repetition as one phase
        ref_ph, user_ph = [(ref_ph[0][0], ref_ph[-1][1])], [(user_ph[0][0], user_ph[-1][1])]
    for (ra, rb), (ua, ub) in zip(ref_ph, user_ph):
        na, nb = rb - ra, ub - ua
        L = max(4, int(round((na + nb) / 2)))
        A = _resample_rows(Xr[ra:rb], L)
        B = _resample_rows(Xu[ua:ub], L)
        ri, ui, c = dtw_slope(feat_cost(A, B))
        rs.append(ra + ri / (L - 1) * (na - 1))
        us.append(ua + ui / (L - 1) * (nb - 1))
        costs.append(c)
        w.append(L)
    return np.concatenate(us), np.concatenate(rs), float(np.average(costs, weights=w))


def method_m2(pair: Pair, bounds, filt: str = "oneeuro", alpha: float = 0.5, template: str = "auto") -> list[SectionOut]:
    Fr_all, Fu_all = pair.ref.F[filt], pair.user.F[filt]
    Fu_sw_all = swapped(Fu_all, pair.names)
    out = []
    for sid, r0, r1, u0, u1 in bounds:
        sec = SectionOut(sid, r0, r1, u0, u1)
        out.append(sec)
        if r1 - r0 < 6 or u1 - u0 < 6:
            continue
        sr, su, vr, vu = _cycle_mi(pair, filt, r0, r1, u0, u1)
        auto_r = segment_reps(sr)
        reps_r, hr, repetitive = ref_template(pair, sid, r0, r1, auto_r, [hold_plateau(sr, vr, r) for r in auto_r], template)
        if hr is None:
            hr = [hold_plateau(sr, vr, r) for r in reps_r]
        if repetitive:
            period = period_of(su)                # period-aware peak spacing
            reps_u = segment_reps(su, min_period_s=max(3.0, 0.6 * period) if period else 3.0)
        else:                                   # not a repeated move: the whole section is one interval
            reps_u = [(0, u1 - u0, int(np.argmax(su)))]
            sec.has_reps = False
        hu = [hold_plateau(su, vu, r) for r in reps_u]
        sec.ref_reps = [Rep(r0 + a, r0 + b, (r0 + h[0], r0 + h[1])) for (a, b, _), h in zip(reps_r, hr)]
        Zr = _std(Fr_all[r0:r1], pair.scales)
        sd = np.nanstd(np.gradient(np.nan_to_num(Zr), axis=0) / DT, axis=0) + 1e-3
        Xr = np.concatenate([Zr, alpha * _velocity(Zr, sd)], axis=1)
        Xu = {}
        for mirror, F in ((False, Fu_all), (True, Fu_sw_all)):
            Zu = _std(F[u0:u1], pair.scales)
            Xu[mirror] = np.concatenate([Zu, alpha * _velocity(Zu, sd)], axis=1)
        for k, ((ua, ub, _), h) in enumerate(zip(reps_u, hu)):
            best = None
            for j, ((ra, rb, _), hrr) in enumerate(zip(reps_r, hr)):
                ref_ph = _phases((ra, rb), hrr)
                user_ph = _phases((ua, ub), h)
                for mirror in ((False, True) if len(reps_r) == 1 else (False,)):   # both sides demonstrated: no mirroring
                    ui, ri, c = _phase_dtw(Xr, Xu[mirror], ref_ph, user_ph)
                    if best is None or c < best[0] - (0.05 if mirror else 0):
                        best = (c, j, mirror, ui, ri)
            c, j, mirror, ui, ri = best
            ra, rb, _ = reps_r[j]
            ratio = (ub - ua) / max(1, rb - ra)
            u = Rep(u0 + ua, u0 + ub, (u0 + h[0], u0 + h[1]), ref=j, mirror=mirror, cost=c,
                    paired=bool(0.4 <= ratio <= 2.5 and c <= 2.5))
            if u.paired:
                u.path = _to_path(u0 + ui, r0 + ri, u.start, u.end)
            sec.user_reps.append(u)
        _pick_defaults(sec)
    return out


# ----------------------------------------------------------------------------- M3 / M3f (template search)

class _MapRep:
    """Pose-map representation (cosine distance)."""
    name = "posemap"
    mirror_penalty, abs_cap, masked_cost = 0.03, 0.45, 1.0

    def __init__(self, pair: Pair, filt: str):
        self.R = pair.ref.M[filt]
        self.U = {False: pair.user.M[filt], True: posemap.flip(pair.user.M[filt])}

    def ref_rows(self, a, b, n):
        X = _resample_rows(self.R[a:b], n)
        nrm = np.linalg.norm(np.nan_to_num(X), axis=1, keepdims=True)
        return X / np.where(nrm > 0, nrm, 1)

    def cost(self, X, u0, u1, mirror, resample=None):
        U = self.U[mirror][u0:u1]
        if resample:
            U = _resample_rows(U, resample)
            nrm = np.linalg.norm(np.nan_to_num(U), axis=1, keepdims=True)
            U = U / np.where(nrm > 0, nrm, 1)
        return posemap.distance(X, U)

    def rest_signal(self, a, b):
        R = self.R[a:b]
        rest = np.nanmean(R[:10], axis=0, keepdims=True)
        d = posemap.distance(R, rest / max(1e-9, np.linalg.norm(rest)))[:, 0]
        v = np.r_[0, 1 - np.nansum(np.nan_to_num(R[1:]) * np.nan_to_num(R[:-1]), axis=1)] / DT
        return d * 10, v       # scaled so the generic amplitude threshold (0.5) means a visible move


class _FeatRep:
    """Feature representation (production features + velocity, standardised)."""
    name = "features"
    mirror_penalty, abs_cap, masked_cost = 0.05, 2.5, 3.0

    def __init__(self, pair: Pair, filt: str, alpha: float = 0.5):
        Zr = _std(pair.ref.F[filt], pair.scales)
        sd = np.nanstd(np.gradient(np.nan_to_num(Zr), axis=0) / DT, axis=0) + 1e-3
        self.R = np.concatenate([Zr, alpha * _velocity(Zr, sd)], axis=1)
        self.U = {}
        for mirror, F in ((False, pair.user.F[filt]), (True, swapped(pair.user.F[filt], pair.names))):
            Zu = _std(F, pair.scales)
            self.U[mirror] = np.concatenate([Zu, alpha * _velocity(Zu, sd)], axis=1)
        self.G, self.g_sc = mirror_invariant(pair.ref.F[filt], pair.names, pair.scales)

    def ref_rows(self, a, b, n):
        return _resample_rows(self.R[a:b], n)

    def cost(self, X, u0, u1, mirror, resample=None):
        U = self.U[mirror][u0:u1]
        return feat_cost(X, _resample_rows(U, resample) if resample else U)

    def rest_signal(self, a, b):
        Z = self.G[a:b] / self.g_sc
        rest = np.nanmedian(Z[:10], axis=0)
        d = np.sqrt(np.nanmean((Z - rest) ** 2, axis=1))
        v = np.linalg.norm(np.nan_to_num(np.gradient(Z, axis=0)), axis=1) / DT
        return d, v


from ..engine.normalize import CANON  # noqa: E402

_MIRROR_J = [CANON.index(j.replace("left_", "#").replace("right_", "left_").replace("#", "right_")) for j in CANON]
COORD_TOL = 0.15     # torso lengths: a coordinate difference of this size counts as one tolerance unit


def _coords(N: np.ndarray, mirror: bool) -> np.ndarray:
    """(T, 13, 2) same-body skeleton -> (T, 26); mirror = horizontal flip with left/right joints exchanged."""
    if mirror:
        N = N[:, _MIRROR_J].copy()
        N[..., 0] *= -1
    return N.reshape(len(N), -1) / COORD_TOL


class _CoordRep(_FeatRep):
    """Keypoint-coordinate representation (same-body normalised skeleton + velocity), optionally
    concatenated with the production features; both groups weigh the same in the mean-|d| cost."""

    def __init__(self, pair: Pair, filt: str, with_features: bool = False, alpha: float = 0.5):
        super().__init__(pair, filt, alpha)                      # rest signal and feature rows
        feat_r, feat_u = self.R, self.U
        Cr = _coords(pair.ref.N[filt], False)
        sd = np.nanstd(np.gradient(np.nan_to_num(Cr), axis=0) / DT, axis=0) + 1e-3
        self.R = np.concatenate([Cr, alpha * _velocity(Cr, sd)], axis=1)
        self.U = {m: np.concatenate([(Cu := _coords(pair.user.N[filt], m)), alpha * _velocity(Cu, sd)], axis=1)
                  for m in (False, True)}
        if with_features:
            w = feat_r.shape[1] / self.R.shape[1]
            self.R = np.concatenate([feat_r, w * self.R], axis=1)
            self.U = {m: np.concatenate([feat_u[m], w * self.U[m]], axis=1) for m in (False, True)}


def method_m3(pair: Pair, bounds, filt: str = "oneeuro", representation: str = "posemap",
              template: str = "auto") -> list[SectionOut]:
    rep = {"posemap": lambda: _MapRep(pair, filt), "features": lambda: _FeatRep(pair, filt),
           "coords": lambda: _CoordRep(pair, filt), "both": lambda: _CoordRep(pair, filt, True)}[representation]()
    out = []
    for sid, r0, r1, u0, u1 in bounds:
        sec = SectionOut(sid, r0, r1, u0, u1)
        out.append(sec)
        if r1 - r0 < 6 or u1 - u0 < 6:
            continue
        d, v = rep.rest_signal(r0, r1)
        auto_r = segment_reps(d)
        reps_r, hr, repetitive = ref_template(pair, sid, r0, r1, auto_r, [hold_plateau(d, v, r) for r in auto_r], template)
        if hr is None:
            hr = [hold_plateau(d, v, r) for r in reps_r]
        sec.ref_reps = [Rep(r0 + a, r0 + b, (r0 + h[0], r0 + h[1])) for (a, b, _), h in zip(reps_r, hr)]
        if not repetitive:                      # one interval: slope-constrained DTW over the whole section
            sec.has_reps = False
            L = max(4, int(round((r1 - r0 + u1 - u0) / 2)))
            ri, ui, c = dtw_slope(rep.cost(rep.ref_rows(r0, r1, L), u0, u1, False, resample=L))
            uf, rf = u0 + ui / (L - 1) * (u1 - u0 - 1), r0 + ri / (L - 1) * (r1 - r0 - 1)
            sec.user_reps = [Rep(u0, u1, None, ref=0, cost=c, paired=True, default=True, path=_to_path(uf, rf, u0, u1))]
            continue
        abs_reps = [(r0 + a, r0 + b, r0 + p) for a, b, p in reps_r]
        tempo = (u1 - u0) / max(1, r1 - r0)
        expected = max(1, int(round(tempo * len(reps_r))))
        occ, _ = search_reps(rep, abs_reps, u0, u1, expected, pair.user.F[filt], pair.names, pair.scales)
        for f in occ:
            ua, ub = u0 + f["start"], u0 + f["end"]
            ui, ri = _to_path(u0 + f["u"], f["r"], ua, ub)
            h0r, h1r = sec.ref_reps[f["ref"]].hold
            inh = np.flatnonzero((ri >= h0r) & (ri < h1r))
            hold = (int(ui[inh[0]]), int(ui[inh[-1]]) + 1) if len(inh) else None
            sec.user_reps.append(Rep(ua, ub, hold, ref=f["ref"], mirror=f["mirror"], cost=f["cost"],
                                     paired=True, path=(ui, ri)))
        _pick_defaults(sec)
    return out


METHODS = {
    "M0": ("逐帧 DTW（0.3 / 最初版本）", method_m0, "sg"),
    "M1": ("重复级·相位线性（0.4 / 现行）", method_m1, "sg"),
    "M2": ("方案 A：显式重复 + 阶段 + 约束 DTW", method_m2, "oneeuro"),
    "M3": ("步态式：姿态图 + 模板搜索", method_m3, "oneeuro"),
    "M3f": ("模板搜索（特征版，消融）", lambda p, b, f: method_m3(p, b, f, "features"), "oneeuro"),
    "M2a": ("方案 A + 标注模板", lambda p, b, f: method_m2(p, b, f, template="annotated"), "oneeuro"),
    "M3a": ("步态式 + 标注模板", lambda p, b, f: method_m3(p, b, f, template="annotated"), "oneeuro"),
    "M3fa": ("模板搜索（特征版）+ 标注模板", lambda p, b, f: method_m3(p, b, f, "features", template="annotated"), "oneeuro"),
    "M3ca": ("模板搜索（关键点坐标）+ 标注模板", lambda p, b, f: method_m3(p, b, f, "coords", template="annotated"), "oneeuro"),
    "M3fca": ("模板搜索（特征 + 坐标）+ 标注模板", lambda p, b, f: method_m3(p, b, f, "both", template="annotated"), "oneeuro"),
}
