"""analyze.py -- Protocol Section 9: all statistics and figures from the
raw eval_results.csv. Every number/figure in the paper comes from here;
nothing is hand-edited.

Produces:
  tables/  mean-rank tables, win-tie-loss, Holm-vs-control per (clf,metric)
  figs/    CD diagrams (real, with connecting bars), RQ3 correlation,
           RQ2 ablation summary
  RQ2_ablation.csv, RQ3_correlation.txt
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from labels import PROPOSED, ABLATIONS, canonicalize, results_path

CONTROL = PROPOSED

MAIN_TIERS = [1, 2, 3, 0]     # comparison set excludes ablation-only variants
CLFS = ["LR", "SVC", "RF", "HGB"]
METRICS = ["pr_auc", "roc_auc", "gmean"]


def load():
    df = canonicalize(pd.read_csv(results_path("eval_results.csv")))
    agg = (df.groupby(["dataset", "method", "tier", "classifier"])
             [["pr_auc", "roc_auc", "gmean", "novelty", "spread",
               "certified_fraction"]].mean().reset_index())
    return df, agg


def holm(pvals, names):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return dict(zip(names, adj))


def main_comparison(agg):
    """RQ1: per (classifier, metric) Friedman + Holm vs control."""
    main_methods = sorted(agg[agg.tier.isin(MAIN_TIERS)].method.unique())
    main_methods = [m for m in main_methods if m not in ABLATIONS]
    summary = []
    for clf in CLFS:
        for metric in METRICS:
            sub = agg[(agg.classifier == clf)]
            piv = sub.pivot_table(index="dataset", columns="method",
                                  values=metric)[main_methods].dropna()
            if piv.shape[0] < 5 or CONTROL not in piv.columns:
                continue
            # Friedman
            stat, p = friedmanchisquare(*[piv[m].values for m in main_methods])
            # ranks (higher metric = rank 1)
            ranks = piv.rank(axis=1, ascending=False).mean()
            # Holm vs control (Wilcoxon signed-rank, control vs each)
            others = [m for m in main_methods if m != CONTROL]
            pv = []
            for m in others:
                try:
                    _, pw = wilcoxon(piv[CONTROL], piv[m])
                except ValueError:
                    pw = 1.0
                pv.append(pw)
            adj = holm(pv, others)
            for m in main_methods:
                wtl = ""
                if m != CONTROL:
                    d = piv[CONTROL] - piv[m]
                    wtl = f"{(d>1e-3).sum()}/{(d.abs()<=1e-3).sum()}/{(d<-1e-3).sum()}"
                summary.append(dict(
                    classifier=clf, metric=metric, method=m,
                    mean_rank=round(ranks[m], 3),
                    mean_score=round(piv[m].mean(), 4),
                    holm_p_vs_control=(round(adj[m], 4) if m in adj else np.nan),
                    win_tie_loss_ctrl=wtl,
                    friedman_p=round(p, 6)))
    out = pd.DataFrame(summary)
    out.to_csv("tables/rq1_main_comparison.csv", index=False)
    return out, main_methods


def cd_diagram(agg, metric="pr_auc", clf="HGB", methods=None):
    """Real CD diagram with connecting bars (Demsar) via Nemenyi CD."""
    sub = agg[agg.classifier == clf]
    piv = sub.pivot_table(index="dataset", columns="method",
                          values=metric)[methods].dropna()
    N, k = piv.shape
    if N < 5:
        return
    ranks = piv.rank(axis=1, ascending=False)
    avg = ranks.mean().sort_values()
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.948,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
               13: 3.313}.get(k, 3.35)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

    fig, ax = plt.subplots(figsize=(8, 0.5 * k + 1.5))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 2)
    names = list(avg.index)
    rs = avg.values
    for i, (nm, r) in enumerate(zip(names, rs)):
        yv = k + 1 - i
        ax.plot([r, r], [yv, k + 1.5], color="0.6", lw=0.8)
        ax.plot([r, lo - 0.4 if r < (lo + hi) / 2 else hi + 0.4],
                [yv, yv], color="0.6", lw=0.8)
        side = lo - 0.45 if r < (lo + hi) / 2 else hi + 0.45
        ha = "right" if r < (lo + hi) / 2 else "left"
        star = " *" if nm == CONTROL else ""
        ax.text(side, yv, f"{nm}{star} ({r:.2f})", va="center", ha=ha,
                fontsize=9, fontweight="bold" if nm == CONTROL else "normal")
    ax.plot([lo, hi], [k + 1.5, k + 1.5], "k-", lw=1)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [k + 1.45, k + 1.5], "k-", lw=1)
        ax.text(x, k + 1.7, str(x), ha="center", fontsize=8)
    # CD bar
    ax.plot([lo, lo + cd], [k + 0.3, k + 0.3], "k-", lw=2.5)
    ax.text(lo + cd / 2, k + 0.5, f"CD={cd:.2f}", ha="center", fontsize=8)
    # connecting bars for non-significant cliques
    grouped = []
    for i in range(len(names)):
        for j in range(len(names) - 1, i, -1):
            if rs[j] - rs[i] <= cd:
                grouped.append((rs[i], rs[j], i))
                break
    for lvl, (a, b, i) in enumerate(grouped):
        yv = k + 1 - i + 0.15
        ax.plot([a - 0.05, b + 0.05], [yv, yv], "r-", lw=2.5)
    ax.set_title(f"CD diagram: {metric}, {clf}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"figures/cd_{metric}_{clf}.png", dpi=160, bbox_inches="tight")
    plt.close()


def rq2_ablations(agg):
    """Wilcoxon full-vs-each-ablation, per classifier, Holm across the 4."""
    rows = []
    for clf in CLFS:
        sub = agg[agg.classifier == clf]
        piv = sub.pivot_table(index="dataset", columns="method",
                              values="pr_auc")
        if CONTROL not in piv:
            continue
        present = [a for a in ABLATIONS if a in piv.columns]
        pv = []
        for a in present:
            pair = piv[[CONTROL, a]].dropna()
            try:
                _, p = wilcoxon(pair[CONTROL], pair[a])
            except ValueError:
                p = 1.0
            pv.append(p)
        adj = holm(pv, present)
        for a in present:
            pair = piv[[CONTROL, a]].dropna()
            d = pair[CONTROL] - pair[a]
            rows.append(dict(classifier=clf, ablation=a,
                             delta_pr_auc=round(d.mean(), 4),
                             holm_p=round(adj[a], 4),
                             confirmed=bool(adj[a] < 0.05 and d.mean() > 0)))
    out = pd.DataFrame(rows)
    out.to_csv("tables/rq2_ablation.csv", index=False)
    return out


def rq3_correlation(df, agg):
    """CISO advantage over SMOTE vs certified fraction and IR (the
    preregistered Section-10 hypothesis)."""
    ir_by_ds = df.groupby("dataset")["ir"].first()
    lines = []
    ref = "SMOTE" if "SMOTE" in agg.method.unique() else "ROS"
    for clf in CLFS:
        sub = agg[agg.classifier == clf]
        piv = sub.pivot_table(index="dataset", columns="method",
                              values="pr_auc")
        if not {CONTROL, ref} <= set(piv.columns):
            continue
        adv = (piv[CONTROL] - piv[ref]).dropna()
        cf = (agg[(agg.classifier == clf) & (agg.method == CONTROL)]
              .set_index("dataset")["certified_fraction"].reindex(adv.index))
        m = adv.index.intersection(cf.dropna().index)
        if len(m) >= 5:
            rho_c, p_c = spearmanr(cf.loc[m], adv.loc[m])
            rho_i, p_i = spearmanr(ir_by_ds.reindex(m), adv.loc[m])
            lines.append(f"{clf}: rho(certfrac, adv)={rho_c:.3f} (p={p_c:.3g}); "
                         f"rho(IR, adv)={rho_i:.3f} (p={p_i:.3g}); n={len(m)}")
    with open("RQ3_correlation.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    # scatter for HGB
    sub = agg[(agg.classifier == "HGB")]
    piv = sub.pivot_table(index="dataset", columns="method", values="pr_auc")
    if {CONTROL, ref} <= set(piv.columns):
        adv = (piv[CONTROL] - piv[ref]).dropna()
        cf = (agg[(agg.classifier == "HGB") & (agg.method == CONTROL)]
              .set_index("dataset")["certified_fraction"].reindex(adv.index))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(cf, adv, s=30)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("certified fraction")
        ax.set_ylabel(f"PR-AUC advantage over {ref} (HGB)")
        ax.set_title("RQ3: where CISO helps")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("figures/rq3_certfrac_vs_advantage.png", dpi=160,
                    bbox_inches="tight")
        plt.close()
    return lines


if __name__ == "__main__":
    df, agg = load()
    rq1, main_methods = main_comparison(agg)
    print("RQ1 written to tables/rq1_main_comparison.csv")
    for metric in METRICS:
        for clf in CLFS:
            cd_diagram(agg, metric, clf, main_methods)
    print("CD diagrams written to figs/")
    print("\nRQ2 ablations:\n", rq2_ablations(agg).to_string(index=False))
    print("\nRQ3:\n", "\n".join(rq3_correlation(df, agg)))
    print("\nAnalysis complete.")
