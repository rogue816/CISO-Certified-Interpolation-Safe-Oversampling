"""fidelity.py -- RQ6 metrics (Protocol Section 13). Frozen specification.

Primary:   unbiased MMD^2, RBF kernel, bandwidth = median pairwise
           distance among TRAINING minority (shared across methods).
Secondary: largest principal angle between top-2 PCs of synthetic vs
           held-out minority.
Diagnostic: classifier two-sample test AUC (duplication-favoring; caveat
           in protocol).
"""
import numpy as np


def median_bandwidth(X_train_min):
    """Median pairwise distance among training minority; per-fold, shared."""
    n = len(X_train_min)
    if n < 2:
        return 1.0
    d2 = (np.sum(X_train_min**2, 1)[:, None]
          + np.sum(X_train_min**2, 1)[None, :]
          - 2 * X_train_min @ X_train_min.T)
    d = np.sqrt(np.maximum(d2[np.triu_indices(n, 1)], 0))
    med = np.median(d)
    return float(med) if med > 0 else 1.0


def _rbf(A, B, sigma):
    d2 = (np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :]
          - 2 * A @ B.T)
    return np.exp(-np.maximum(d2, 0) / (2 * sigma**2))


def mmd2_unbiased(X_syn, X_held, sigma):
    """Unbiased MMD^2 estimator (Gretton et al. 2012, Eq. 4)."""
    m, n = len(X_syn), len(X_held)
    if m < 2 or n < 2:
        return np.nan
    Kxx = _rbf(X_syn, X_syn, sigma)
    Kyy = _rbf(X_held, X_held, sigma)
    Kxy = _rbf(X_syn, X_held, sigma)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sxx + syy - 2 * Kxy.mean())


def principal_angle_top2(X_syn, X_held):
    """Largest principal angle (degrees) between top-2 PC subspaces."""
    if len(X_syn) < 3 or len(X_held) < 3 or X_syn.shape[1] < 2:
        return np.nan

    def top2(X):
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt[:2].T                              # d x 2

    U, V = top2(X_syn), top2(X_held)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.degrees(np.arccos(s.min())))     # largest angle


def c2st_auc(X_syn, X_held, seed=0):
    """Classifier two-sample test AUC. DIAGNOSTIC ONLY (duplication-
    favoring; see Protocol 13). AUC near 0.5 = indistinguishable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    m, n = len(X_syn), len(X_held)
    if m < 5 or n < 5:
        return np.nan
    X = np.vstack([X_syn, X_held])
    y = np.hstack([np.zeros(m), np.ones(n)])
    k = int(min(5, m, n))
    try:
        p = cross_val_predict(
            LogisticRegression(max_iter=1000),
            X, y, method="predict_proba",
            cv=StratifiedKFold(k, shuffle=True, random_state=seed))[:, 1]
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan
