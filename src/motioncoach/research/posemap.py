"""Gait-recognition style representation: body-normalised skeletons rendered as pose maps.

Each frame becomes a small image (``H x W``) of Gaussian-blurred limb lines drawn from the
"same body" skeleton (``engine.normalize.normalized``: hip-midpoint origin, reference torso unit,
practitioner bones rescaled to the reference). Frames are compared by cosine distance of the maps,
so the representation is holistic (no hand-picked angles) and joint-label free (a left/right
mirror is a horizontal flip of the image). A repetition can be summarised by its pose energy
image (mean map) or by phase-binned maps.
"""

from __future__ import annotations

import numpy as np

from ..engine.normalize import CANON, EDGES

BOX = (-1.9, 1.9, -2.7, 2.4)        # x0, x1, y0, y1 in torso units (same box as the workbench skeleton view)
_IDX = {j: i for i, j in enumerate(CANON)}
_SEG = [(_IDX[a], _IDX[b]) for a, b in EDGES]


def render(N: np.ndarray, W: int = 32, H: int = 40, sigma: float = 0.10, min_edges: int = 8) -> np.ndarray:
    """(T, len(CANON), 2) normalised skeletons -> (T, H*W) float32 L2-normalised maps (NaN rows if too few limbs)."""
    x0, x1, y0, y1 = BOX
    gx, gy = np.meshgrid(np.linspace(x0, x1, W), np.linspace(y0, y1, H))
    G = np.stack([gx.ravel(), gy.ravel()], axis=1)                     # (P, 2)
    sh = 0.5 * (N[:, _IDX["left_shoulder"]] + N[:, _IDX["right_shoulder"]])
    segs = [(N[:, a], N[:, b]) for a, b in _SEG] + [(sh, N[:, _IDX["nose"]])]
    T = len(N)
    out = np.zeros((T, len(G)), dtype=np.float32)
    count = np.zeros(T, dtype=np.int64)
    for A, B in segs:
        ok = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
        count += ok
        if not ok.any():
            continue
        a, b = A[ok], B[ok]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
        # distance from every grid point to segment a-b, in chunks of frames
        for s in range(0, len(a), 512):
            aa, abv, l2 = a[s:s + 512, None], ab[s:s + 512, None], L2[s:s + 512, None]
            u = np.clip(((G[None] - aa) * abv).sum(axis=2) / l2, 0, 1)
            d2 = ((G[None] - (aa + u[..., None] * abv)) ** 2).sum(axis=2)
            v = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
            rows = np.flatnonzero(ok)[s:s + 512]
            out[rows] = np.maximum(out[rows], v)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.where(nrm > 0, nrm, 1)
    out[count < min_edges] = np.nan
    return out


def flip(M: np.ndarray, W: int = 32, H: int = 40) -> np.ndarray:
    """Horizontal mirror of flattened maps."""
    return M.reshape(len(M), H, W)[:, :, ::-1].reshape(len(M), H * W)


def distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos) between rows of A and B; 1.0 where either frame is invalid."""
    a, b = np.nan_to_num(A), np.nan_to_num(B)
    D = 1.0 - a @ b.T
    D[~np.isfinite(A).all(axis=1)] = 1.0
    D[:, ~np.isfinite(B).all(axis=1)] = 1.0
    return np.clip(D, 0, 1).astype(np.float32)


def phase_bins(M: np.ndarray, a: int, b: int, k: int = 8) -> np.ndarray:
    """Mean map in ``k`` equal phase bins of rows a..b-1 (a chrono pose-energy descriptor), (k, P)."""
    edges = np.linspace(a, b, k + 1).round().astype(int)
    out = []
    for s, e in zip(edges[:-1], edges[1:]):
        blk = M[s:max(e, s + 1)]
        blk = blk[np.isfinite(blk).all(axis=1)]
        out.append(blk.mean(axis=0) if len(blk) else np.full(M.shape[1], np.nan))
    return np.asarray(out)
