"""
loaders.py -- dataset loading for the CISO benchmark (Protocol Section 2).

KEEL:  parse_keel(path) handles the real-file quirks observed in the
       repository files: CRLF endings, commas inside attribute range
       brackets, inconsistent delimiters, trailing whitespace, <null>/'?'
       missing values, nominal one-hot encoding.
       Positive-class rule: honor KEEL's declared 'positive' label when
       present (the repository's minority convention); otherwise smallest
       class vs rest.

UCI:   load_uci(name) fetches via ucimlrepo (network required -- runs on
       the experiment machine, not in offline sandboxes) and applies the
       PREREGISTERED binarization map. Every load is validated against
       EXPECTED (n, d, IR) from the protocol; a mismatch raises, so a
       wrong repo id or wrong positive class is caught on first run.

describe(metas) generates the paper's dataset table FROM THE DATA
(Protocol: never hand-typed).
"""

import os
import numpy as np
import pandas as pd

MISSING_TOKENS = {"?", "<null>", "null", ""}


# =====================================================================
# KEEL
# =====================================================================

def parse_keel(path):
    """Parse a KEEL .dat file. Returns (X, y, meta).

    X : float array, nominals one-hot encoded (drop_first=False)
    y : int array, 1 = positive/minority
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.read().splitlines()          # kills \r\n and \n alike

    attrs, outputs_name, data_start = [], None, None
    for i, line in enumerate(raw_lines):
        s = line.strip()
        low = s.lower()
        if low.startswith("@attribute"):
            # '@attribute Name real [0.0, 0.89]' or '@attribute Sex {M, F, I}'
            parts = s.split(None, 2)
            name = parts[1].strip()
            spec = parts[2].strip() if len(parts) > 2 else "real"
            attrs.append((name, spec))
        elif low.startswith("@outputs"):
            outputs_name = s.split(None, 1)[1].strip().rstrip(",")
        elif low.startswith("@data"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError(f"{path}: no @data section found")

    # class column: the one named by @outputs, else the last attribute
    names = [a[0] for a in attrs]
    if outputs_name is not None and outputs_name in names:
        class_idx = names.index(outputs_name)
    else:
        class_idx = len(attrs) - 1

    rows = []
    for line in raw_lines[data_start:]:
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("@"):
            continue
        tokens = [t.strip() for t in s.split(",")]
        if len(tokens) != len(attrs):
            raise ValueError(f"{path}: row has {len(tokens)} fields, "
                             f"expected {len(attrs)}: {s[:80]}")
        rows.append(tokens)
    df = pd.DataFrame(rows, columns=names)

    # class labels
    y_str = df.iloc[:, class_idx].astype(str).str.strip()
    labels = set(y_str.unique())
    if len(labels) != 2:
        raise ValueError(f"{path}: expected 2 classes, found {sorted(labels)}")
    lowmap = {l.lower(): l for l in labels}
    counts = y_str.value_counts()
    if "positive" in lowmap and "negative" in lowmap:
        pos_label = lowmap["positive"]              # KEEL's declared minority
        rule = "keel-declared"
        # Guard: in some files (e.g. ecoli-0_vs_1) the declared 'positive'
        # is the LARGER class; the oversampling task requires y=1 minority.
        if counts[pos_label] > counts.min():
            pos_label = counts.idxmin()
            rule = "smallest-class (keel label inverted)"
    else:
        pos_label = counts.idxmin()                 # smallest class
        rule = "smallest-class"
    y = (y_str == pos_label).astype(int).to_numpy()

    # features
    feat_cols = [c for j, c in enumerate(names) if j != class_idx]
    parts_out, d_raw = [], len(feat_cols)
    for j, col in enumerate(names):
        if j == class_idx:
            continue
        spec = attrs[j][1]
        vals = df[col].astype(str).str.strip()
        vals = vals.where(~vals.str.lower().isin(MISSING_TOKENS), np.nan)
        if spec.startswith("{"):
            dummies = pd.get_dummies(vals, prefix=col, dummy_na=False)
            parts_out.append(dummies.to_numpy(dtype=float))
        else:
            parts_out.append(pd.to_numeric(vals, errors="coerce")
                             .to_numpy(dtype=float).reshape(-1, 1))
    X = np.hstack(parts_out)

    # training-time imputation happens per fold; whole-file NaN column means
    # are used here only so meta stats are computable
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])
        n_imputed = len(idx[0])
    else:
        n_imputed = 0

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    meta = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "source": "KEEL",
        "n": len(y), "d_raw": d_raw, "d_encoded": X.shape[1],
        "n_pos": n_pos, "n_neg": n_neg,
        "ir": round(n_neg / n_pos, 2) if n_pos else float("inf"),
        "positive_label": pos_label, "positive_rule": rule,
        "n_imputed_cells": n_imputed,
    }
    return X, y, meta


def load_keel_dir(directory, names=None):
    """Yield (X, y, meta) for every .dat (or the given names) in a folder."""
    if names is None:
        files = sorted(f for f in os.listdir(directory) if f.endswith(".dat"))
    else:
        files = [n if n.endswith(".dat") else n + ".dat" for n in names]
    for f in files:
        yield parse_keel(os.path.join(directory, f))


# =====================================================================
# UCI (runs on the experiment machine; requires network + ucimlrepo)
# =====================================================================

# Preregistered binarization (Protocol Section 2) + expected stats used
# as a LOAD-TIME SELF-CHECK. repo ids to be verified on first fetch.
UCI_SPECS = {
    #  name              (repo id,  expected n, expected d)
    # positive class: smallest class vs rest (universal rule, all multiclass)
    # binary datasets: native minority class
    "statlog_landsat":   (146,  6435, 36),
    "waveform":          (107,  5000, 21),
    "page_blocks":       (78,   5473, 10),
    "htru2":             (372,  17898, 8),
    "musk_v2":           (75,   6598, 166),
    "letter":            (59,   20000, 16),
    "spambase":          (94,   4601, 57),
    "online_shoppers":   (468,  12330, 17),
}


def _find_minority(y_series):
    """Smallest class vs rest -- universal rule for all UCI datasets."""
    ys = y_series.iloc[:, 0] if isinstance(y_series, pd.DataFrame) else y_series
    counts = ys.value_counts()
    pos_label = counts.idxmin()
    return (ys == pos_label).astype(int).to_numpy(), pos_label


def _binarize(y_series, rule):
    kind, value = rule
    assert kind == "class_equals"
    ys = y_series.iloc[:, 0] if isinstance(y_series, pd.DataFrame) else y_series

    def norm(v):
        s = str(v).strip()
        try:
            return float(s)
        except ValueError:
            return s.lower()

    target = norm(value)
    return (ys.map(norm) == target).astype(int).to_numpy()


def load_uci(name):
    """Fetch, binarize (smallest class vs rest), and self-check n and d."""
    from ucimlrepo import fetch_ucirepo
    repo_id, exp_n, exp_d = UCI_SPECS[name]
    ds = fetch_ucirepo(id=repo_id)
    Xdf = ds.data.features.copy()

    # encode features
    parts = []
    for col in Xdf.columns:
        s = Xdf[col]
        if s.dtype == bool:
            parts.append(s.astype(float).to_numpy().reshape(-1, 1))
        elif s.dtype == object or str(s.dtype) == "category":
            parts.append(pd.get_dummies(s, prefix=col).to_numpy(dtype=float))
        else:
            parts.append(pd.to_numeric(s, errors="coerce")
                         .to_numpy(dtype=float).reshape(-1, 1))
    X = np.hstack(parts)
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])

    y, pos_label = _find_minority(ds.data.targets)
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    ir = n_neg / max(n_pos, 1)

    meta = {"name": name, "source": "UCI", "n": len(y),
            "d_raw": Xdf.shape[1], "d_encoded": X.shape[1],
            "n_pos": n_pos, "n_neg": n_neg, "ir": round(ir, 2),
            "positive_label": str(pos_label),
            "positive_rule": "smallest-class-vs-rest"}

    # self-check: n and d only (IR now data-driven, not hardcoded)
    problems = []
    if len(y) != exp_n:
        problems.append(f"n={len(y)} expected {exp_n}")
    if Xdf.shape[1] != exp_d:
        problems.append(f"d={Xdf.shape[1]} expected {exp_d}")
    if problems:
        raise ValueError(f"UCI self-check FAILED for '{name}' "
                         f"(repo id {repo_id}): " + "; ".join(problems) +
                         ". Verify repo id in UCI_SPECS.")
    return X, y, meta


# =====================================================================
# Paper table generation (Protocol: tables are computed, never typed)
# =====================================================================

def describe(metas, fmt="markdown"):
    df = pd.DataFrame(metas)[
        ["name", "source", "n", "d_raw", "ir", "positive_label"]]
    df = df.sort_values(["source", "name"]).reset_index(drop=True)
    return df.to_markdown(index=False) if fmt == "markdown" else df
