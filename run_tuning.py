"""run_tuning.py -- Protocol Section 8: the preregistered tuning run.

Grid: k {5,7} x alpha {-1,0,0.5,1,2} x q {1.2,1.5,2.0} x beta {0.1,0.2,0.4}
      = 90 configurations, eps0 = 0.1 fixed.
Datasets: the 5 tuning datasets ONLY. Full CV protocol (5x3 paired folds,
4 classifiers). Selection criterion applied afterward by lock_section10.py.

Checkpointing: results are appended to tuning_results.csv one dataset at a
time (smallest first); on restart, completed datasets are skipped.
"""
import os
import time
import itertools
import pandas as pd
KEEL_DIR = os.environ.get("CISO_KEEL", "data/keel")
RESULTS  = os.environ.get("CISO_RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)

from loaders import parse_keel
from registry import Method
from engine import run_benchmark, make_classifiers
from ciso import CISO

OUT = os.path.join(RESULTS, "tuning_results.csv")
TUNING = ["glass0", "haberman", "ecoli3", "yeast3", "abalone19"]  # small first

GRID = [dict(k=k, alpha=a, q=q, beta=b)
        for k, a, q, b in itertools.product(
            (5, 7), (-1.0, 0.0, 0.5, 1.0, 2.0), (1.2, 1.5, 2.0),
            (0.1, 0.2, 0.4))]
assert len(GRID) == 90


def cfg_name(c):
    return f"k{c['k']}_a{c['alpha']}_q{c['q']}_b{c['beta']}"


def make_methods():
    ms = []
    for c in GRID:
        ms.append(Method(cfg_name(c), 0,
                         factory=(lambda s, cc=dict(c): CISO(
                             random_state=s, eps0=0.1, **cc))))
    return ms


def done_pairs():
    """Set of (dataset, config-name) pairs already checkpointed."""
    if not os.path.exists(OUT):
        return set()
    try:
        df = pd.read_csv(OUT, usecols=["dataset", "method"])
        return set(map(tuple, df.drop_duplicates().values))
    except Exception:
        return set()


BATCH = 15  # configs per checkpoint write


if __name__ == "__main__":
    clfs = make_classifiers(include_svc=True)
    all_methods = make_methods()
    done = done_pairs()
    print(f"grid={len(GRID)} configs; {len(done)} (dataset,cfg) pairs done",
          flush=True)
    for name in TUNING:
        X, y, meta = parse_keel(os.path.join(KEEL_DIR, f"{name}.dat"))
        todo = [m for m in all_methods if (name, m.name) not in done]
        if not todo:
            print(f"[skip] {name}: complete", flush=True)
            continue
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            t0 = time.time()
            res, fail = run_benchmark([(X, y, meta)], batch, clfs,
                                      verbose=False)
            res.to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                       index=False)
            if len(fail):
                fail.to_csv(os.path.join(RESULTS, "tuning_failures.csv"), mode="a",
                            header=not os.path.exists(os.path.join(RESULTS, "tuning_failures.csv")),
                            index=False)
            print(f"[ckpt] {name}: configs {i+1}-{i+len(batch)}/{len(todo)} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        print(f"[done] {name}", flush=True)
    print("TUNING RUN COMPLETE", flush=True)
