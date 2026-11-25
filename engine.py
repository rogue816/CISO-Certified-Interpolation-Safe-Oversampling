"""engine.py -- Protocol Section 3/7: paired-fold benchmark engine.

Guarantees encoded here:
  * Fold indices generated once per (dataset, repeat) with seed 42+repeat
    and shared by ALL methods and classifiers (paired comparisons).
  * Preprocessing (mean imputation + standardization) fit on the training
    fold only.
  * Sampler seed = 1000*repeat + fold, identical across methods.
  * Failure handling (Section 7): a failing sampler is recorded as a
    failure row and falls back to no-resampling FOR THAT FOLD ONLY,
    flagged in the output; never silent.
  * Output: one long-format CSV row per
    (dataset, method, classifier, repeat, fold) with all metrics,
    diagnostics, timings, and flags.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from metrics import compute_scores, synth_diagnostics

N_SPLITS = 5
N_REPEATS = 3
FOLD_SEED_BASE = 42
SAMPLER_SEED = lambda rep, fold: 1000 * rep + fold


def make_classifiers(include_svc=True):
    clfs = {
        "LR": lambda cw: LogisticRegression(max_iter=2000, random_state=42,
                                            class_weight=cw),
        "RF": lambda cw: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=42,
                                                class_weight=cw),
        "HGB": lambda cw: HistGradientBoostingClassifier(random_state=42,
                                                         class_weight=cw),
    }
    if include_svc:
        # Protocol 5: SVC scores via decision_function (no probability=True;
        # that would add an internal 5-fold CV and deviate from the freeze).
        clfs["SVC"] = lambda cw: SVC(kernel="rbf", random_state=42,
                                     class_weight=cw)
    return clfs


def scores_for(clf, X_te):
    """Ranking scores per Protocol 5/6: predict_proba where available,
    decision_function otherwise. Returns (ranking_score, hard_pred)."""
    if hasattr(clf, "predict_proba"):
        s = clf.predict_proba(X_te)[:, 1]
        return s, (s >= 0.5).astype(int)
    s = clf.decision_function(X_te)
    return s, (s >= 0.0).astype(int)


def paired_folds(y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """Yield (repeat, fold, train_idx, test_idx); n_splits reduced to the
    minority count when minority < n_splits (Protocol Section 3)."""
    n_min = int(np.sum(y == 1))
    k = min(n_splits, n_min)
    if k < 2:
        return
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=FOLD_SEED_BASE + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield rep, fold, tr, te


def preprocess(X_tr, X_te):
    """Train-fold mean imputation + standardization (Protocol Section 2)."""
    col_mean = np.nanmean(X_tr, axis=0)
    X_tr = np.where(np.isnan(X_tr), col_mean, X_tr)
    X_te = np.where(np.isnan(X_te), col_mean, X_te)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd


def run_benchmark(datasets, methods, classifiers, verbose=True):
    """datasets: iterable of (X, y, meta). Returns (results_df, failures_df)."""
    rows, failures = [], []
    for X, y, meta in datasets:
        ds = meta["name"]
        if verbose:
            print(f"=== {ds} (n={meta['n']}, IR={meta['ir']}) ===", flush=True)
        for rep, fold, tr, te in paired_folds(y):
            X_tr, X_te = preprocess(X[tr].astype(float), X[te].astype(float))
            y_tr, y_te = y[tr], y[te]
            seed = SAMPLER_SEED(rep, fold)

            for m in methods:
                if not m.available:
                    continue
                fallback = False
                try:
                    X_res, y_res, info = m.resample(X_tr, y_tr, seed)
                except Exception as e:
                    failures.append({"dataset": ds, "method": m.name,
                                     "repeat": rep, "fold": fold,
                                     "error": f"{type(e).__name__}: {e}"})
                    X_res, y_res = X_tr, y_tr
                    info = {"X_syn": None, "resample_time": np.nan}
                    fallback = True

                diag = synth_diagnostics(info.get("X_syn"),
                                         X_tr[y_tr == 1])
                cw = "balanced" if m.kind == "weights" else None

                for clf_name, clf_factory in classifiers.items():
                    clf = clf_factory(cw)
                    t0 = time.time()
                    clf.fit(X_res, y_res)
                    fit_time = time.time() - t0
                    rank_s, hard = scores_for(clf, X_te)
                    scores = compute_scores(y_te, rank_s, hard)
                    rows.append({
                        "dataset": ds, "method": m.name, "tier": m.tier,
                        "classifier": clf_name, "repeat": rep, "fold": fold,
                        **scores, **diag,
                        "certified_fraction": info.get("certified_fraction",
                                                       np.nan),
                        "resample_time": info.get("resample_time", np.nan),
                        "clf_fit_time": fit_time,
                        "fallback": fallback,
                        "ir": meta["ir"], "n": meta["n"],
                    })
    results = pd.DataFrame(rows)
    fail_df = pd.DataFrame(failures)
    return results, fail_df


def aggregate(results):
    """Dataset-level means (over 15 fold-evaluations), per Protocol 3."""
    return (results.groupby(["dataset", "method", "tier", "classifier"])
            [["pr_auc", "roc_auc", "gmean", "novelty", "spread",
              "certified_fraction", "resample_time"]]
            .mean().reset_index())


def rank_table(agg, metric="pr_auc", classifier="LR"):
    sub = agg[agg["classifier"] == classifier]
    pivot = sub.pivot(index="dataset", columns="method", values=metric)
    ranks = pivot.rank(axis=1, ascending=False)
    return ranks.mean().sort_values()
