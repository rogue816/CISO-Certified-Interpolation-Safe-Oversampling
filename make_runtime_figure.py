"""make_runtime_figure.py -- regenerate the runtime figure of the paper.

Panel (a): median resampling time per fold on the 37-dataset evaluation suite.
Panel (b): median time per fold against dataset size on the 8-dataset scale suite.

Reads the frozen CSVs in results/ by default. Override with CISO_RESULTS.

Usage:
    python make_runtime_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from labels import canonicalize, results_path

matplotlib.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.8, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

OURS = "CISO"
OUT = "figures/fig_runtime.pdf"

# panel (a): samplers that actually resample, in the paper's order
BARS = ["ROS", "SMOTE", "ADASYN", "SMOTE-Tomek", "Safe-Level-SMOTE",
        "G-SMOTE", OURS]
DISP = {"SMOTE-Tomek": "SMOTE-TL", "Safe-Level-SMOTE": "SafeLvl"}

# panel (b): three curves, enough to show the gap without clutter
CURVES = [(OURS, "#1f6fb4", "o"), ("G-SMOTE", "#8a8a8a", "^"),
          ("SMOTE", "#c0392b", "s")]


def _median_times(fname):
    df = canonicalize(pd.read_csv(results_path(fname)))
    return df


def main(out=OUT):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # ---------------- panel (a): evaluation suite ----------------
    ev = _median_times("eval_results.csv")
    med = ev.groupby("method").resample_time.median()
    bars = [m for m in BARS if m in med.index]
    vals = [med[m] for m in bars]

    ypos = range(len(bars))
    colors = ["#1f6fb4" if m == OURS else "#b8b8b8" for m in bars]
    axa.barh(list(ypos), vals, color=colors, edgecolor="#4a4a4a", height=0.62,
             linewidth=0.6)
    axa.set_yticks(list(ypos))
    axa.set_yticklabels([DISP.get(m, m) for m in bars], fontsize=8)
    axa.invert_yaxis()
    axa.set_xscale("log")
    axa.set_xlabel("median resampling time per fold (s), log scale", fontsize=8.5)
    axa.set_title("(a) evaluation suite", fontsize=9)
    axa.tick_params(labelsize=8)
    axa.grid(axis="x", alpha=0.25, lw=0.5)
    axa.set_axisbelow(True)
    for y, v in zip(ypos, vals):
        axa.annotate(f"{v * 1000:.1f} ms", (v, y), textcoords="offset points",
                     xytext=(5, -3), fontsize=7.5)
    axa.set_xlim(right=max(vals) * 4)

    # ---------------- panel (b): scale suite ----------------
    sc = _median_times("scale_results.csv")
    size = sc.groupby("dataset").n.max()
    g = sc.groupby(["dataset", "method"]).resample_time.median().unstack()
    for m, col, mk in CURVES:
        if m not in g.columns:
            continue
        s = pd.DataFrame({"n": size, "t": g[m]}).dropna().sort_values("n")
        axb.plot(s.n, s.t, marker=mk, color=col, lw=1.2, ms=4.5, label=m)
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlabel("dataset size $N$", fontsize=8.5)
    axb.set_ylabel("median time per fold (s)", fontsize=8.5)
    axb.set_title("(b) scale suite", fontsize=9)
    axb.tick_params(labelsize=8)
    axb.grid(alpha=0.25, lw=0.5)
    axb.set_axisbelow(True)
    axb.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

    # ---------------- the numbers quoted in the paper ----------------
    for fname, lab in [("eval_results.csv", "evaluation"),
                       ("scale_results.csv", "scale")]:
        df = _median_times(fname)
        p = df.groupby(["dataset", "method"]).resample_time.median().unstack()
        r = (p[OURS] / p["SMOTE"]).dropna()
        print(f"{lab:10s} median per-dataset ratio to SMOTE: {r.median():.0f}x"
              f"   range {r.min():.0f}-{r.max():.0f}x")


if __name__ == "__main__":
    main()
