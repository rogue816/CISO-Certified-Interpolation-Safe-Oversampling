"""metrics.py -- Protocol Section 6: primary/secondary metrics + diagnostics."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix


def compute_scores(y_true, y_score, y_pred=None, threshold=0.5):
    """PR-AUC (primary), ROC-AUC from the ranking score; G-mean from the
    hard prediction (probability >= 0.5, or decision_function >= 0 for
    SVC per Protocol 5). Single-class test folds -> NaN AUCs."""
    y_true = np.asarray(y_true)
    out = {}
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["pr_auc"] = np.nan
        out["roc_auc"] = np.nan
    if y_pred is None:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    out["gmean"] = float(np.sqrt(sens * spec))
    return out


def synth_diagnostics(X_syn, X_min_orig):
    """Novelty and spread (Protocol 6). ROS scores ~0 on both.
    Diagnostics of mechanism, never of merit."""
    if X_syn is None or len(X_syn) == 0:
        return {"novelty": np.nan, "spread": np.nan}
    d_to_orig = _min_cross_dist(X_syn, X_min_orig)
    novelty = float(d_to_orig.mean())
    if len(X_syn) >= 2:
        d_pair = _min_cross_dist(X_syn, X_syn, exclude_self=True)
        spread = float(d_pair.mean())
    else:
        spread = np.nan
    return {"novelty": novelty, "spread": spread}


def _min_cross_dist(A, B, exclude_self=False):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    D = np.sqrt(np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0))
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return D.min(axis=1)
