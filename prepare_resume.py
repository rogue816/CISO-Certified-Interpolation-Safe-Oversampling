"""prepare_resume.py -- one-time fix before completing the evaluation run.
Backs up eval_results.csv, removes ONLY MWMOTE fallback placeholder rows
(so the resume retries MWMOTE after `pip install -U smote-variants`), and
leaves every committed result row untouched. Logged in Protocol S12.
"""
import shutil
import pandas as pd

shutil.copy("results/eval_results.csv", "results/eval_results.backup_pass1.csv")
df = pd.read_csv("results/eval_results.csv")
before = len(df)
df = df[~((df.method == "MWMOTE") & (df.fallback == True))]
df.to_csv("results/eval_results.csv", index=False)
print(f"removed {before - len(df)} MWMOTE fallback rows "
      f"({len(df)} committed rows preserved; backup written)")
