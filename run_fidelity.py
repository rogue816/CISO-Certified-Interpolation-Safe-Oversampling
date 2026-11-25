"""run_fidelity.py -- RQ6 (Protocol Section 13). Same paired 5x3 folds as
the evaluation run; per fold, each sampler generates from the training
fold and is scored against the HELD-OUT minority. Checkpointed per
dataset; rerun resumes. No classifiers involved -- this run is fast.

Usage:  python run_fidelity.py            # evaluation suite (37)
        python run_fidelity.py --tuning   # machinery validation only
"""
import os
import sys
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
from fidelity import median_bandwidth, mmd2_unbiased, principal_angle_top2, c2st_auc
from run_evaluation import EVAL, LOCKED

TUNING = ["glass0", "haberman", "ecoli3", "yeast3", "abalone19"]
RQ6_METHODS = ["CISO", "SMOTE", "Borderline-SMOTE", "ADASYN", "ROS"]
OUT  = os.path.join(RESULTS, "fidelity_results.csv")
FAIL = os.path.join(RESULTS, "fidelity_failures.csv")
N_SPLITS, N_REPEATS = 5, 3


def done_datasets():
    if not os.path.exists(OUT):
        return set()
    return set(pd.read_csv(OUT, usecols=["dataset"]).dataset.unique())


if __name__ == "__main__":
    tuning_mode = "--tuning" in sys.argv
    names = TUNING if tuning_mode else EVAL
    out = os.path.join(RESULTS, "tuning_fidelity_results.csv") if tuning_mode else OUT
    reg = {m.name: m for m in build_registry(tuned=LOCKED)}
    methods = [reg[n] for n in RQ6_METHODS if n in reg and reg[n].available]
    missing = [n for n in RQ6_METHODS if n not in [m.name for m in methods]]
    if missing:
        print(f"WARNING unavailable (recorded): {missing}", flush=True)

    done = done_datasets() if not tuning_mode else set()
    rows, fails = [], []
    for name in names:
        if name in done:
            print(f"[skip] {name}", flush=True)
            continue
        X, y, meta = parse_keel(os.path.join(KEEL_DIR, f"{name}.dat"))
        t0 = time.time()
        usable = 0
        for rep in range(N_REPEATS):
            skf = StratifiedKFold(n_splits=min(N_SPLITS, int(y.sum())),
                                  shuffle=True, random_state=42 + rep)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                y_te_min = y[te] == 1
                X_held = X[te][y_te_min]
                if len(X_held) < 3:                       # exclusion rule
                    continue
                usable += 1
                scaler = StandardScaler().fit(X[tr])
                X_tr = scaler.transform(X[tr])
                X_hd = scaler.transform(X[te][y_te_min])
                sigma = median_bandwidth(X_tr[y[tr] == 1])
                seed = 1000 * rep + fold
                for spec in methods:
                    info_row = dict(dataset=name, method=spec.name,
                                    repeat=rep, fold=fold)
                    try:
                        Xr, yr, info = spec.resample(X_tr, y[tr], seed)
                        if info.get("failed"):
                            raise RuntimeError(info["error"])
                        X_syn = Xr[len(X_tr):]
                        if len(X_syn) < 2:
                            raise RuntimeError("no synthetic block")
                        info_row.update(
                            mmd2=mmd2_unbiased(X_syn, X_hd, sigma),
                            pangle=principal_angle_top2(X_syn, X_hd),
                            c2st=c2st_auc(X_syn, X_hd, seed=seed),
                            n_syn=len(X_syn), n_held=len(X_hd),
                            sigma=sigma)
                        rows.append(info_row)
                    except Exception as e:
                        fails.append(dict(dataset=name, method=spec.name,
                                          repeat=rep, fold=fold,
                                          error=f"{type(e).__name__}: {e}"))
        df = pd.DataFrame([r for r in rows if r["dataset"] == name])
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"[done] {name}: {usable} usable folds, "
              f"{len(df)} rows, {(time.time()-t0)/60:.1f} min", flush=True)
    if fails:
        pd.DataFrame(fails).to_csv(
            ("tuning_" if tuning_mode else "") + FAIL, mode="a",
            header=not os.path.exists(
                ("tuning_" if tuning_mode else "") + FAIL), index=False)
    print("FIDELITY RUN COMPLETE", flush=True)
