"""lock_section10.py -- apply the PREREGISTERED selection criterion
(Protocol Section 8) to a completed tuning_results.csv and print the
configuration to be written into Protocol Section 10.

Criterion (frozen): maximize mean over {5 tuning datasets} of the mean
over {4 classifiers} of dataset-level PR-AUC (mean over 15 folds).
Ties: higher G-mean, then smaller |alpha| (parsimony).

Refuses to run on an incomplete grid.
"""
import re
import sys
import pandas as pd

EXPECTED_DATASETS = {"glass0", "haberman", "ecoli3", "yeast3", "abalone19"}
EXPECTED_CONFIGS = 90
EXPECTED_CLFS = {"LR", "RF", "HGB", "SVC"}

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/tuning_results.csv")

# ---- completeness gate ----------------------------------------------
missing_ds = EXPECTED_DATASETS - set(df.dataset.unique())
n_cfg = df.method.nunique()
missing_clf = EXPECTED_CLFS - set(df.classifier.unique())
pairs = df[["dataset", "method"]].drop_duplicates().shape[0]
want = len(EXPECTED_DATASETS) * EXPECTED_CONFIGS
if missing_ds or n_cfg != EXPECTED_CONFIGS or missing_clf or pairs != want:
    sys.exit(f"INCOMPLETE GRID: missing datasets={missing_ds}, "
             f"configs={n_cfg}/{EXPECTED_CONFIGS}, missing clfs={missing_clf}, "
             f"pairs={pairs}/{want}. Refusing to lock.")

# ---- criterion -------------------------------------------------------
# fold-mean per (dataset, config, classifier), then mean over classifiers,
# then mean over datasets
lvl = (df.groupby(["dataset", "method", "classifier"])
         [["pr_auc", "gmean"]].mean()
         .groupby(["dataset", "method"]).mean()      # over classifiers
         .groupby("method").mean())                   # over datasets

lvl["abs_alpha"] = [abs(float(re.search(r"_a(-?[\d.]+)_", m).group(1)))
                    for m in lvl.index]
lvl = lvl.sort_values(["pr_auc", "gmean", "abs_alpha"],
                      ascending=[False, False, True])

best = lvl.index[0]
parts = dict(re.findall(r"([kaqb])([-\d.]+)", best))
print("=== Tuning selection (preregistered criterion) ===")
print(lvl.head(10).round(4).to_string())
print(f"\nLOCKED CONFIGURATION -> Protocol Section 10:")
print(f"  k = {parts['k']}, alpha = {parts['a']}, "
      f"q = {parts['q']}, beta = {parts['b']}, eps0 = 0.1")
print("\nCopy this into PROTOCOL.md Section 10 with the run date, then the")
print("evaluation suite may be run EXACTLY ONCE.")
