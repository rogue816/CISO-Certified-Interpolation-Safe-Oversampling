"""run_evaluation.py -- Protocol Section 3-5: the single evaluation run.

Locked config (Section 10): k=7, alpha=2.0, q=1.2, beta=0.1, eps0=0.1.
37 KEEL evaluation datasets x all available methods x 4 classifiers x
5x3 paired folds. Checkpointed per (dataset, method) so it resumes.

RUN EXACTLY ONCE. Re-running only fills missing (dataset, method) pairs;
it never overwrites committed rows.
"""
import os
import time
import pandas as pd
KEEL_DIR = os.environ.get("CISO_KEEL", "data/keel")
RESULTS  = os.environ.get("CISO_RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)

from loaders import parse_keel
from registry import build_registry
from engine import run_benchmark, make_classifiers

LOCKED = dict(k=7, alpha=2.0, q=1.2, beta=0.1, eps0=0.1)
OUT = os.path.join(RESULTS, "eval_results.csv")
FAIL = os.path.join(RESULTS, "eval_failures.csv")

EVAL = [  # 37 datasets (Protocol Section 2)
    "abalone9-18", "dermatology-6", "ecoli-0_vs_1", "ecoli1", "ecoli4",
    "flare-F", "glass-0-1-2-3_vs_4-5-6", "glass-0-1-6_vs_2",
    "glass-0-1-6_vs_5", "glass1", "glass2", "glass4", "glass5", "glass6",
    "new-thyroid1", "newthyroid2", "pima", "segment0", "shuttle-c0-vs-c4",
    "vehicle0", "vehicle1", "vehicle2", "vehicle3", "vowel0", "car-good",
    "winequality-red-4", "wisconsin", "yeast-0-5-6-7-9_vs_4",
    "yeast-1-2-8-9_vs_7", "yeast-1-4-5-8_vs_7", "yeast-1_vs_7",
    "yeast-2_vs_4", "yeast-2_vs_8", "yeast1", "yeast4", "yeast5", "yeast6",
]


def done_pairs():
    if not os.path.exists(OUT):
        return set()
    df = pd.read_csv(OUT, usecols=["dataset", "method"])
    return set(map(tuple, df.drop_duplicates().values))


if __name__ == "__main__":
    reg = build_registry(tuned=LOCKED)
    methods = [m for m in reg if m.available]
    unavailable = [m.name for m in reg if not m.available]
    if unavailable:
        print(f"UNAVAILABLE (install libs to include): {unavailable}",
              flush=True)
    clfs = make_classifiers(include_svc=True)
    print(f"methods: {[m.name for m in methods]}", flush=True)
    print(f"classifiers: {list(clfs)}", flush=True)

    done = done_pairs()
    for name in EVAL:
        X, y, meta = parse_keel(os.path.join(KEEL_DIR, f"{name}.dat"))
        todo = [m for m in methods if (name, m.name) not in done]
        if not todo:
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        res, fail = run_benchmark([(X, y, meta)], todo, clfs, verbose=False)
        res.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        if len(fail):
            fail.to_csv(FAIL, mode="a", header=not os.path.exists(FAIL),
                        index=False)
        print(f"[done] {name} (IR={meta['ir']}): {len(res)} rows, "
              f"{len(fail)} failures, {(time.time()-t0)/60:.1f} min",
              flush=True)
    print("EVALUATION RUN COMPLETE", flush=True)
