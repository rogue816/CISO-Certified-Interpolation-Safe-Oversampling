"""run_scale.py -- Protocol Section 2: the UCI scale suite (8 datasets).
SVC excluded here per the preregistered restriction; LR, RF, HGB run.
Requires network (ucimlrepo) + imbalanced-learn + smote_variants.
Checkpointed per dataset; rerun resumes. Smallest datasets first; letter
and htru2 last (expect the long tail there -- hours are normal).
"""
import os
import time
import pandas as pd
KEEL_DIR = os.environ.get("CISO_KEEL", "data/keel")
RESULTS  = os.environ.get("CISO_RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)

from loaders import load_uci
from registry import build_registry
from engine import run_benchmark, make_classifiers
from run_evaluation import LOCKED

SCALE = ["spambase", "waveform", "musk_v2", "statlog_landsat",
         "page_blocks", "online_shoppers", "htru2", "letter"]
OUT  = os.path.join(RESULTS, "scale_results.csv")
FAIL = os.path.join(RESULTS, "scale_failures.csv")


def done():
    if not os.path.exists(OUT):
        return set()
    return set(pd.read_csv(OUT, usecols=["dataset"]).dataset.unique())


if __name__ == "__main__":
    methods = [m for m in build_registry(tuned=LOCKED) if m.available]
    clfs = make_classifiers(include_svc=False)      # preregistered exclusion
    skip = done()
    print(f"{len(SCALE)} datasets x {len(methods)} methods x {list(clfs)}",
          flush=True)
    for name in SCALE:
        if name in skip:
            print(f"[skip] {name}", flush=True)
            continue
        X, y, meta = load_uci(name)                  # self-checks n/d/IR
        t0 = time.time()
        res, fail = run_benchmark([(X, y, meta)], methods, clfs,
                                  verbose=False)
        res.to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                   index=False)
        if len(fail):
            fail.to_csv(FAIL, mode="a", header=not os.path.exists(FAIL),
                        index=False)
        print(f"[done] {name}: {len(res)} rows, {len(fail)} failures, "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    print("SCALE RUN COMPLETE", flush=True)
