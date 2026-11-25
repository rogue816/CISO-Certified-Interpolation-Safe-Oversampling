"""Canonical method labels, and the legacy map for the frozen record.

The method was called PO-qG while the experiments were run. It was renamed
to CISO (Certified Interpolation Safe Oversampling) before submission. The
CSVs under results/ are the frozen run artefacts and still carry the old
labels. They are shipped unmodified, because rewriting a frozen record is
worse than carrying a map.

Fresh runs emit canonical labels directly, so `canonicalize` is the identity
on anything produced by the current registry.

Usage
-----
    import pandas as pd
    from labels import canonicalize
    df = canonicalize(pd.read_csv("results/eval_results.csv"))
"""
import os

PROPOSED = "CISO"

ABLATIONS = [
    "CISO[alpha=0]",
    "CISO[uniform-lambda]",
    "CISO[s=gamma]",
    "CISO[no-clearance-II]",
]

#: legacy label -> canonical label
LEGACY = {
    "PO-qG": "CISO",
    "PO-qG[alpha=0]": "CISO[alpha=0]",
    "PO-qG[uniform-lambda]": "CISO[uniform-lambda]",
    "PO-qG[s=gamma]": "CISO[s=gamma]",
    "PO-qG[no-clearance-II]": "CISO[no-clearance-II]",
    # alpha sweep labels
    "PO-qG[a=-1.0]": "CISO[a=-1.0]",
    "PO-qG[a=+0.0]": "CISO[a=+0.0]",
    "PO-qG[a=+0.5]": "CISO[a=+0.5]",
    "PO-qG[a=+1.0]": "CISO[a=+1.0]",
    "PO-qG[a=+2.0]": "CISO[a=+2.0]",
}


def canonical(name):
    """Map one label to its canonical form."""
    return LEGACY.get(name, name)


def canonicalize(df, column="method"):
    """Return `df` with `column` mapped to canonical labels (copy)."""
    if column not in df.columns:
        return df
    out = df.copy()
    out[column] = out[column].map(lambda v: LEGACY.get(v, v))
    return out


def results_dir():
    """Where the run CSVs live. Override with CISO_RESULTS."""
    return os.environ.get("CISO_RESULTS", "results")


def results_path(fname):
    return os.path.join(results_dir(), fname)
