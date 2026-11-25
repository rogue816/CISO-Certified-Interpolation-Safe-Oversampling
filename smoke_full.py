"""Step-3 smoke test (Protocol): full pipeline on 3 small real datasets,
all sandbox-available methods, all 4 classifiers, 5x3 paired folds.
Doubles as the compute estimator for the tuning and evaluation runs.
"""
import time
import numpy as np
import pandas as pd
from loaders import parse_keel
from registry import build_registry
from engine import run_benchmark, make_classifiers, aggregate, rank_table

SMOKE = ["ecoli3", "glass0", "haberman"]

datasets = []
for name in SMOKE:
    X, y, meta = parse_keel(f"keel_data/{name}.dat")
    datasets.append((X, y, meta))

methods = [m for m in build_registry() if m.available]
clfs = make_classifiers(include_svc=True)
print("methods:", [m.name for m in methods])
print("classifiers:", list(clfs))

t0 = time.time()
results, failures = run_benchmark(datasets, methods, clfs)
wall = time.time() - t0

results.to_csv("smoke_results.csv", index=False)
failures.to_csv("smoke_failures.csv", index=False)

print(f"\nrows={len(results)}  failures={len(failures)}  wall={wall:.1f}s")
assert not results[["pr_auc", "gmean"]].isna().all().any()

agg = aggregate(results)
print("\n=== mean PR-AUC rank per method (HGB) ===")
print(rank_table(agg, "pr_auc", "HGB").round(2))
print("\n=== sanity: novelty/spread by method ===")
print(agg.groupby("method")[["novelty", "spread"]].mean().round(3))

# ---------------- compute projection --------------------------------
# cost model: per-(dataset,fold) cost ~ sum over methods of
# (resample + classifier fits). Use measured per-row times.
per_cell = results.groupby(["dataset"])[["resample_time", "clf_fit_time"]].sum()
smoke_total = per_cell.sum().sum()
print(f"\nmeasured compute in smoke (resample+fit): {smoke_total:.1f}s "
      f"of {wall:.1f}s wall")

# scale factors (crude, linear in folds x methods x clfs; refined later):
n_eval_ds, n_methods_full, n_clfs = 37, 17, 4      # KEEL evaluation run
mean_ds_cost = wall / len(SMOKE)                   # per dataset, 8 methods x 4 clfs
eval_est = n_eval_ds * mean_ds_cost * (n_methods_full / len(methods))
tune_est = 5 * mean_ds_cost * (90 / len(methods)) * (1 / 1)  # 90 configs, CISO only-ish
print(f"projection: KEEL evaluation run ~ {eval_est/60:.0f} min "
      f"(small-data extrapolation; scale suite estimated separately)")
print(f"projection: tuning run (90 cfgs x 5 ds x 15 folds x 4 clfs) "
      f"~ {tune_est/60:.0f} min")
