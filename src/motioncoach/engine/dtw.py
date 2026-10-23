"""DTW kernels with a local slope constraint, template search, and explicit-duration segmentation.

Step pattern (Sakoe-Chiba, P = 1, symmetric weights): (1,1) w2, (1,2) and (2,1) w3 through the
intermediate cell. Any path therefore has local slope in [1/2, 2]: one side can never stand still
while the other moves, which is what let frame DTW pile a whole repetition onto one frame.
"""

from __future__ import annotations

import numba
import numpy as np

BIG = np.float32(1e9)


@numba.njit(cache=True)
def _dtw_slope(C):
    n, m = C.shape
    D = np.full((n, m), BIG, dtype=np.float32)
    B = np.full((n, m), -1, dtype=np.int8)
    D[0, 0] = 2 * C[0, 0]
    for i in range(1, n):
        for j in range(1, m):
            best = D[i - 1, j - 1] + 2 * C[i, j]
            b = 0
            if j >= 2:
                v = D[i - 1, j - 2] + 2 * C[i, j - 1] + C[i, j]
                if v < best:
                    best, b = v, 1
            if i >= 2:
                v = D[i - 2, j - 1] + 2 * C[i - 1, j] + C[i, j]
                if v < best:
                    best, b = v, 2
            D[i, j] = best
            B[i, j] = b
    return D, B


@numba.njit(cache=True)
def _backtrack(B, i, j):
    ri = np.empty(B.shape[0] + B.shape[1], dtype=np.int64)
    rj = np.empty(B.shape[0] + B.shape[1], dtype=np.int64)
    k = 0
    while True:
        ri[k] = i
        rj[k] = j
        k += 1
        b = B[i, j]
        if b == -1:
            break
        if b == 0:
            i -= 1
            j -= 1
        elif b == 1:
            ri[k] = i
            rj[k] = j - 1
            k += 1
            i -= 1
            j -= 2
        else:
            ri[k] = i - 1
            rj[k] = j
            k += 1
            i -= 2
            j -= 1
    return ri[:k][::-1], rj[:k][::-1]


def dtw_slope(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fixed-end, slope-constrained DTW. Returns (rows, cols, mean cost); needs 1/2 <= m/n <= 2."""
    C = np.ascontiguousarray(C, dtype=np.float32)
    n, m = C.shape
    D, B = _dtw_slope(C)
    if D[-1, -1] >= BIG / 2:
        raise ValueError(f"slope-infeasible lengths {n} x {m}")
    ri, rj = _backtrack(B, n - 1, m - 1)
    return ri, rj, float(D[-1, -1] / (n + m))


@numba.njit(cache=True)
def _subseq(C):
    """Template (rows) searched in a stream (cols): free start/end column, slope-constrained."""
    n, m = C.shape
    D = np.full((n, m), BIG, dtype=np.float32)
    B = np.full((n, m), -1, dtype=np.int8)
    S = np.zeros((n, m), dtype=np.int64)
    for j in range(m):
        D[0, j] = 2 * C[0, j]
        S[0, j] = j
    for i in range(1, n):
        for j in range(1, m):
            best = D[i - 1, j - 1] + 2 * C[i, j]
            b = 0
            s = S[i - 1, j - 1]
            if j >= 2:
                v = D[i - 1, j - 2] + 2 * C[i, j - 1] + C[i, j]
                if v < best:
                    best, b, s = v, 1, S[i - 1, j - 2]
            if i >= 2:
                v = D[i - 2, j - 1] + 2 * C[i - 1, j] + C[i, j]
                if v < best:
                    best, b, s = v, 2, S[i - 2, j - 1]
            D[i, j] = best
            B[i, j] = b
            S[i, j] = s
    return D, B, S


def subseq_profile(C: np.ndarray):
    """Normalised cost of the best occurrence ending at each stream column, its start, and the DP."""
    C = np.ascontiguousarray(C, dtype=np.float32)
    D, B, S = _subseq(C)
    n = C.shape[0]
    start = S[-1]
    length = np.arange(C.shape[1]) - start + 1
    cost = np.where(D[-1] < BIG / 2, D[-1] / (n + length), np.inf)
    return cost, start, B


def occurrence_path(B: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray]:
    return _backtrack(B, B.shape[0] - 1, end)


@numba.njit(cache=True)
def _hsmm(E, dmin, dmax, dpen_w, dmean, skip):
    """E: (K, T+1) cumulative emission cost per segment. Lead-in/out frames cost ``skip`` each."""
    K, T1 = E.shape
    T = T1 - 1
    V = np.full((K, T + 1), BIG, dtype=np.float64)     # V[k, t]: best cost with segment k ending at t (exclusive)
    P = np.zeros((K, T + 1), dtype=np.int64)            # start of segment k
    for k in range(K):
        for t in range(1, T + 1):
            lo = max(0, t - dmax[k])
            hi = t - dmin[k]
            for s in range(lo, hi + 1):
                prev = skip * s if k == 0 else V[k - 1, s]
                if prev >= BIG / 2:
                    continue
                d = t - s
                r = np.log(d / dmean[k])
                c = prev + (E[k, t] - E[k, s]) + dpen_w * r * r
                if c < V[k, t]:
                    V[k, t] = c
                    P[k, t] = s
    return V, P


def hsmm_segment(emis: np.ndarray, durations: np.ndarray, tempo: float, lo: float = 0.3, hi: float = 3.5,
                 dpen_w: float = 2.0, skip: float = 1.0) -> list[tuple[int, int]]:
    """Left-to-right explicit-duration segmentation of a stream into K ordered sections.

    ``emis`` (K, T): per-frame cost of the frame belonging to section k. ``durations``: reference
    section lengths in frames; the prior is centred on duration x ``tempo`` and limited to [lo, hi] x that.
    """
    K, T = emis.shape
    E = np.concatenate([np.zeros((K, 1)), np.cumsum(emis, axis=1)], axis=1)
    mean = np.maximum(2.0, durations * tempo)
    dmin = np.maximum(1, (mean * lo).astype(np.int64))
    dmax = np.maximum(dmin + 1, (mean * hi).astype(np.int64))
    V, P = _hsmm(E, dmin, dmax, float(dpen_w), mean.astype(np.float64), float(skip))
    tail = V[K - 1] + skip * (T - np.arange(T + 1))
    t = int(np.argmin(tail))
    out = []
    for k in range(K - 1, -1, -1):
        s = int(P[k, t])
        out.append((s, t))
        t = s
    return out[::-1]
