"""run_alpha_sweep.py -- LABELED EXPLORATORY (Protocol Section 9 / note 8).

Maps the fidelity-safety frontier: CISO swept over alpha, measured on the
same paired folds against held-out minority (fidelity, MMD^2) and against
the training majority (safety, mean distance of synthetic to nearest
majority). The four fixed baselines are measured on both axes in the same
folds, so the frontier figure is self-contained from this one CSV.

No classifiers involved -- expect ~30-60 min for all 37 datasets.
"""
import os
import time
import numpy as np
import pandas as pd
KEEL_DIR = os.environ.get("CISO_KEEL", "data/keel")
RESULTS  = os.environ.get("CISO_RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from loaders import parse_keel
from registry import build_registry
from fidelity import median_bandwidth, mmd2_unbiased, principal_angle_top2
from run_evaluation import EVAL, LOCKED
from ciso import CISO

ALPHAS = [-1.0, 0.0, 0.5, 1.0, 2.0]
BASELINES = ["ROS", "SMOTE", "Borderline-SMOTE", "ADASYN"]
OUT = os.path.join(RESULTS, "alpha_sweep_results.csv")


def safety_of(X_syn, X_maj, chunk=2000):
    """Mean distance from synthetic points to their nearest majority."""
    mins = []
    for i in range(0, len(X_syn), chunk):
        d = CISO._cdist(X_syn[i:i + chunk], X_maj)
        mins.append(d.min(axis=1))
    return float(np.concatenate(mins).mean())


def done():
    if not os.path.exists(OUT):
        return set()
    return set(pd.read_csv(OUT, usecols=["dataset"]).dataset.unique())


if __name__ == "__main__":
    reg = {m.name: m for m in build_registry(tuned=LOCKED)}
    base = [reg[n] for n in BASELINES if n in reg and reg[n].available]
    skip = done()
    for name in EVAL:
        if name in skip:
            print(f"[skip] {name}", flush=True)
            continue
        X, y, meta = parse_keel(os.path.join(KEEL_DIR, f"{name}.dat"))
        t0, rows = time.time(), []
        for rep in range(3):
            skf = StratifiedKFold(n_splits=min(5, int(y.sum())),
                                  shuffle=True, random_state=42 + rep)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                X_held = X[te][y[te] == 1]
                if len(X_held) < 3:
                    continue
                sc = StandardScaler().fit(X[tr])
                X_tr, X_hd = sc.transform(X[tr]), sc.transform(X_held)
                y_tr = y[tr]
                X_maj = X_tr[y_tr == 0]
                sigma = median_bandwidth(X_tr[y_tr == 1])
                seed = 1000 * rep + fold
                # CISO alphas
                for a in ALPHAS:
                    m = CISO(random_state=seed, **{**LOCKED, "alpha": a})
                    Xr, _ = m.fit_resample(X_tr, y_tr)
                    syn = Xr[len(X_tr):]
                    if len(syn) < 2:
                        continue
                    rows.append(dict(
                        dataset=name, method=f"CISO[a={a:+.1f}]",
                        repeat=rep, fold=fold,
                        mmd2=mmd2_unbiased(syn, X_hd, sigma),
                        pangle=principal_angle_top2(syn, X_hd),
                        safety=safety_of(syn, X_maj),
                        certfrac=float((m.gamma_ > 0).mean())))
                # fixed baselines
                for spec in base:
                    try:
                        Xr, yr, info = spec.resample(X_tr, y_tr, seed)
                        if info.get("failed"):
                            continue
                        syn = Xr[len(X_tr):]
                        if len(syn) < 2:
                            continue
                        rows.append(dict(
                            dataset=name, method=spec.name,
                            repeat=rep, fold=fold,
                            mmd2=mmd2_unbiased(syn, X_hd, sigma),
                            pangle=principal_angle_top2(syn, X_hd),
                            safety=safety_of(syn, X_maj),
                            certfrac=np.nan))
                    except Exception:
                        continue
        pd.DataFrame(rows).to_csv(OUT, mode="a",
                                  header=not os.path.exists(OUT),
                                  index=False)
        print(f"[done] {name}: {len(rows)} rows, "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    print("ALPHA SWEEP COMPLETE", flush=True)
