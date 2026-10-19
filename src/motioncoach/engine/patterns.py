"""Clustering of joint-angle difference vectors into recurring deviation patterns.

Each aligned moment is a vector of signed differences (user - reference) divided by the
per-feature tolerance. Moments within tolerance on every feature form a fixed class 0
("close to the reference"); only the remaining moments are clustered (k-means, k by
silhouette) into classes 1..k, so that a large "fine" majority cannot swamp the deviation
patterns. Missing features are imputed as 0 (no deviation); moments with <70% measurable
features stay unclassified (-1). Standardised differences are clipped to ±4 so detection glitches
cannot dominate, and clusters holding < 2% of moments are reported as outliers (-2), not patterns.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2


def _silhouette(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n: int = 1200) -> float:
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    D = np.linalg.norm(Xs[:, None] - Xs[None], axis=2)
    ks = np.unique(ls)
    if len(ks) < 2:
        return -1.0
    s = np.zeros(len(Xs))
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, ls == k].mean() for k in ks if k != ls[i])
        s[i] = (b - a) / max(a, b, 1e-9)
    return float(s.mean())


def cluster_deviations(delta: np.ndarray, tol: np.ndarray, mask: np.ndarray, k_range=range(2, 7),
                       min_valid: float = 0.7, near_thr: float = 1.0, min_share: float = 0.02, seed: int = 0) -> dict:
    """delta (N,K) signed differences; tol (K,) tolerance; mask (N,) moments to consider."""
    Z = delta / tol
    valid = np.isfinite(Z).mean(axis=1) >= min_valid
    used = np.flatnonzero(mask & valid)
    labels = np.full(len(delta), -1, dtype=np.int64)
    Zu = np.nan_to_num(Z[used])   # missing dims (<30% by construction of `used`) cannot make a moment "near"
    near = np.nanmax(np.abs(Z[used]), axis=1) < near_thr if len(used) else np.zeros(0, bool)
    labels[used[near]] = 0
    near_center = Zu[near].mean(axis=0) if near.any() else np.zeros(delta.shape[1])
    use = used[~near]
    if len(use) < 20:
        return {"labels": labels, "centers_z": near_center[None], "k": 0, "silhouette": None, "used": used}
    X = np.clip(np.nan_to_num(Z[use]), -4.0, 4.0)
    rng = np.random.default_rng(seed)
    best = None
    for k in k_range:
        if k >= len(X):
            break
        cent, lab = kmeans2(X, k, seed=seed, minit="++", iter=50)
        if len(np.unique(lab)) < k:
            continue
        sil = _silhouette(X, lab, rng)
        if best is None or sil > best[0]:
            best = (sil, k, cent, lab)
    if best is None:
        return {"labels": labels, "centers_z": near_center[None], "k": 0, "silhouette": None, "used": used}
    sil, k, cent, lab = best
    counts = np.bincount(lab, minlength=k)
    keep = [c for c in np.argsort(-counts) if counts[c] >= max(5, min_share * len(used))]
    remap = np.full(k, -2, dtype=np.int64)          # -2 = outlier moments (tiny clusters)
    for new_id, c in enumerate(keep, start=1):      # order deviation clusters by share; class 0 stays "within tolerance"
        remap[c] = new_id
    labels[use] = remap[lab]
    return {"labels": labels, "centers_z": np.vstack([near_center[None], cent[keep]]) if keep else near_center[None],
            "k": len(keep), "silhouette": sil, "used": used}
