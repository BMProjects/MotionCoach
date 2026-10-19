"""Temporal alignment of a reference and a user feature sequence.

Subsequence DTW: the whole reference must be matched, the user recording may contain extra
lead-in/lead-out. Features are divided by per-feature scales (their tolerances) so one unit
of cost means "one tolerance away"; missing features are skipped per cell.

Alignment answers "which user moment corresponds to which reference moment". Shape errors are
then measured at corresponding moments, and timing errors are read from the path itself;
the two are reported separately (a well-warped path is not evidence of correct rhythm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

MISSING_COST = 2.0   # cost of a cell where too few features are measurable on both sides


def cost_matrix(R: np.ndarray, U: np.ndarray, scales: np.ndarray, min_valid: int = 3,
                chunk: int = 256) -> np.ndarray:
    """Mean absolute standardised difference over mutually valid features. (N,K),(M,K) -> (N,M) float32."""
    min_valid = min(min_valid, R.shape[1])
    Rs = (R / scales).astype(np.float32)
    Us = (U / scales).astype(np.float32)
    Rv, Uv = np.isfinite(Rs), np.isfinite(Us)
    Rz, Uz = np.nan_to_num(Rs), np.nan_to_num(Us)
    N, M = len(R), len(U)
    C = np.empty((N, M), dtype=np.float32)
    for s in range(0, N, chunk):
        r, rv = Rz[s:s + chunk, None, :], Rv[s:s + chunk, None, :]
        both = rv & Uv[None]
        d = np.abs(r - Uz[None]) * both
        n = both.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.minimum(d.sum(axis=2) / n, 3.0)   # cap so a single wild frame cannot dominate
        c[n < min_valid] = MISSING_COST
        C[s:s + chunk] = c
    return C


@numba.njit(cache=True)
def _subseq_dtw(C, step_penalty, skip_cost):
    """Symmetric2 DTW (diagonal weight 2, vertical/horizontal weight 1).

    Unmatched user frames before the start / after the end cost ``skip_cost`` each, so every
    hypothesis accounts for the whole user recording. Without this, a free start/end favours
    squeezing the reference into a short user span (fewer cells = lower sum).
    """
    N, M = C.shape
    D = np.empty((N, M), dtype=np.float32)
    B = np.zeros((N, M), dtype=np.int8)   # 0 diag, 1 from (i-1, j), 2 from (i, j-1), 3 path start
    D[0, 0] = 2 * C[0, 0]
    B[0, 0] = 3
    for j in range(1, M):
        start = skip_cost * j + 2 * C[0, j]
        cont = D[0, j - 1] + C[0, j] + step_penalty
        if start <= cont:
            D[0, j] = start
            B[0, j] = 3
        else:
            D[0, j] = cont
            B[0, j] = 2
    for i in range(1, N):
        D[i, 0] = D[i - 1, 0] + C[i, 0] + step_penalty
        B[i, 0] = 1
        for j in range(1, M):
            d = D[i - 1, j - 1] + 2 * C[i, j]
            v = D[i - 1, j] + C[i, j] + step_penalty
            h = D[i, j - 1] + C[i, j] + step_penalty
            if d <= v and d <= h:
                D[i, j] = d
            elif v <= h:
                D[i, j] = v
                B[i, j] = 1
            else:
                D[i, j] = h
                B[i, j] = 2
    return D, B


@numba.njit(cache=True)
def _backtrack(B, j_end):
    N = B.shape[0]
    i, j = N - 1, j_end
    ri = np.empty(N + B.shape[1], dtype=np.int64)
    uj = np.empty(N + B.shape[1], dtype=np.int64)
    k = 0
    while True:
        ri[k] = i
        uj[k] = j
        k += 1
        if B[i, j] == 3:
            break
        b = B[i, j]
        if b == 0:
            i -= 1
            j -= 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return ri[:k][::-1], uj[:k][::-1]


@dataclass
class Alignment:
    ref_idx: np.ndarray        # path, reference sample indices (non-decreasing)
    user_idx: np.ndarray       # path, user sample indices (non-decreasing)
    local_cost: np.ndarray     # C along the path
    total_cost: float          # mean local cost along the path (standardised units)

    def user_for_ref(self, n_ref: int) -> list[np.ndarray]:
        """For each reference sample, the user samples matched to it."""
        out: list[list[int]] = [[] for _ in range(n_ref)]
        for i, j in zip(self.ref_idx, self.user_idx):
            out[i].append(j)
        return [np.asarray(v, dtype=np.int64) for v in out]


def align(R: np.ndarray, U: np.ndarray, scales: np.ndarray, step_penalty: float = 0.02,
          skip_cost: float = 1.0, min_valid: int = 3) -> Alignment:
    """``skip_cost`` (tolerance units per user frame) is the price of leaving user frames unmatched;
    frames whose best match costs more than this are treated as lead-in/lead-out."""
    if len(R) < 2 or len(U) < 2:
        raise ValueError("Sequences too short to align")
    C = cost_matrix(R, U, scales, min_valid=min_valid)
    D, B = _subseq_dtw(C, np.float32(step_penalty), np.float32(skip_cost))
    tail = skip_cost * (len(U) - 1 - np.arange(len(U)))
    j_end = int(np.argmin(D[-1] + tail))
    ri, uj = _backtrack(B, j_end)
    lc = C[ri, uj]
    return Alignment(ri, uj, lc, float(lc.mean()))


# ---------------------------------------------------------------------------------------
# Two-stage alignment
#
# Stage 1 (coarse): mirror-invariant, temporally smoothed features -> segment boundaries in the
#   user recording. Routines like Baduanjin alternate left/right repetitions, so a left/right
#   swap looks like a half-cycle phase shift; invariant features make the coarse pass immune to
#   mirroring and to which side the user starts on.
# Stage 2 (fine): inside each segment, fixed-end DTW on the full features, with and without a
#   left/right swap. The swap decision is made from the summed per-segment evidence.
# ---------------------------------------------------------------------------------------

def mirror_invariant(F: np.ndarray, names: list[str], scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .features import FEATURES

    pos = {n: i for i, n in enumerate(names)}
    cols, sc, done = [], [], set()
    for i, n in enumerate(names):
        if n in done:
            continue
        d = FEATURES[n]
        if d.lr_twin and d.lr_twin in pos:
            j = pos[d.lr_twin]
            done |= {n, d.lr_twin}
            cols += [(F[:, i] + F[:, j]) / 2, np.abs(F[:, i] - F[:, j])]
            sc += [scales[i], scales[i]]
        else:
            cols.append(np.abs(F[:, i]) if d.lateral else F[:, i])
            sc.append(scales[i])
    return np.stack(cols, axis=1), np.asarray(sc, dtype=np.float64)


def smooth_nan(F: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware centred moving average along time."""
    if win <= 1:
        return F
    k = np.ones(win)
    ok = np.isfinite(F)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, np.where(ok, F, 0.0))
    den = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < win * 0.3] = np.nan
    return out


@dataclass
class TwoStageResult:
    alignment: Alignment               # concatenated fine path
    swap: bool                         # chosen left/right interpretation
    swap_cost_ratio: float             # fine cost with swap / without (summed over segments)
    segment_lr: list[dict]             # per-segment evidence
    coarse_cost: float


def two_stage_align(F_ref: np.ndarray, F_user: np.ndarray, t_ref: np.ndarray, names: list[str],
                    scales: np.ndarray, seg_bounds: list[tuple[float, float]], swap_mode: str = "auto",
                    coarse_window_s: float = 2.0, skip_cost: float = 2.0, step_penalty: float = 0.02,
                    swap_threshold: float = 0.95) -> TwoStageResult:
    """``swap_mode``: auto | yes | no. ``seg_bounds`` are reference (start, end) seconds."""
    from .features import lr_swap_features

    dt = float(np.median(np.diff(t_ref)))
    win = max(1, int(round(coarse_window_s / dt)))
    G_r, g_sc = mirror_invariant(F_ref, names, scales)
    G_u, _ = mirror_invariant(F_user, names, scales)
    coarse = align(smooth_nan(G_r, win), smooth_nan(G_u, win), g_sc, step_penalty, skip_cost)

    first_user = np.full(len(F_ref), -1, dtype=np.int64)
    for i, j in zip(coarse.ref_idx[::-1], coarse.user_idx[::-1]):
        first_user[i] = j
    # reference index ranges per segment, and the matching user boundaries
    r_bounds, u_bounds = [], []
    for a, b in seg_bounds:
        idx = np.flatnonzero((t_ref >= a) & (t_ref < b))
        if len(idx) < 2:
            continue
        r_bounds.append((idx[0], idx[-1] + 1))
        u_bounds.append(first_user[idx[0]])
    u_end = int(coarse.user_idx[-1]) + 1
    u_bounds = np.maximum.accumulate(np.asarray(u_bounds + [u_end]))

    F_user_sw = lr_swap_features(F_user, names)
    seg_lr, parts = [], {False: [], True: []}
    totals = {False: 0.0, True: 0.0}
    for k, (r0, r1) in enumerate(r_bounds):
        u0, u1 = int(u_bounds[k]), int(max(u_bounds[k + 1], u_bounds[k] + 2))
        u1 = min(u1, len(F_user))
        ev = {"ref": [int(r0), int(r1)], "user": [u0, u1]}
        for sw, U in ((False, F_user), (True, F_user_sw)):
            al = align(F_ref[r0:r1], U[u0:u1], scales, step_penalty, skip_cost=1e4)   # fixed ends
            parts[sw].append((al.ref_idx + r0, al.user_idx + u0, al.local_cost))
            ev["cost_swap" if sw else "cost"] = al.total_cost
            totals[sw] += al.total_cost * (r1 - r0)
        seg_lr.append(ev)
    ratio = totals[True] / totals[False] if totals[False] > 0 else 1.0
    swap = {"yes": True, "no": False}.get(swap_mode, ratio < swap_threshold)
    ri = np.concatenate([p[0] for p in parts[swap]])
    uj = np.concatenate([p[1] for p in parts[swap]])
    lc = np.concatenate([p[2] for p in parts[swap]])
    return TwoStageResult(Alignment(ri, uj, lc, float(lc.mean())), swap, float(ratio), seg_lr, coarse.total_cost)
