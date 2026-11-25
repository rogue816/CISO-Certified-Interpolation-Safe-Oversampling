"""
make_paper_figures.py — regenerate the CISO paper figures at publication quality.

Produces, into ./figures/:
    fig_rank_boxplots.pdf   fig_sensitivity.pdf   fig_fidelity.pdf
    fig_frontier.pdf        fig_reliability.pdf

fig_reliability.pdf is generated but is not used in the paper.
fig_runtime.pdf comes from make_runtime_figure.py.
fig_certificate_panel.pdf is a toy-geometry panel and is not regenerated here;
it is shipped under figures/ as a run artefact.

Usage:
    python make_paper_figures.py             # all of the above
    python make_paper_figures.py --only box  # one of: box,sens,rel,fid,front

Reads results/eval_results.csv, results/tuning_results.csv,
results/fidelity_results.csv and results/alpha_sweep_results.csv.
Override the directory with CISO_RESULTS.

Edit STYLE below to tune fonts/sizes globally.
"""
import argparse, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- style
STYLE = dict(font_size=9, family="serif", axes_lw=0.8,
             text_width=7.0)     # Springer sn-jnl text width in inches
matplotlib.rcParams.update({
    "font.size": STYLE["font_size"], "font.family": STYLE["family"],
    "mathtext.fontset": "cm", "axes.linewidth": STYLE["axes_lw"],
    "savefig.bbox": "tight", "pdf.fonttype": 42,   # editable text in PDF
})
from labels import canonicalize, results_path
os.makedirs("figures", exist_ok=True)
OURS = "CISO"

MAIN = ["NoResampling","ClassWeights","ROS","SMOTE","Borderline-SMOTE","ADASYN",
        "SMOTE-ENN","SMOTE-Tomek","Safe-Level-SMOTE","G-SMOTE","CISO"]
DISP = {"NoResampling":"No resampling","ClassWeights":"Class weighting",
        "ROS":"Random oversampl.","Safe-Level-SMOTE":"Safe-Level-SMOTE"}
CLFS = ["LR","SVC","RF","HGB"]


def _load_eval():
    df = pd.read_csv(results_path("eval_results.csv")).rename(columns={"method":"m"})
    df = canonicalize(df, "m")
    return df


# ------------------------------------------------- Fig: rank boxplots
def fig_rank_boxplots(fname="figures/fig_rank_boxplots.pdf"):
    df = _load_eval()
    d = df[df.m.isin(MAIN)]
    lvl = d.groupby(["dataset","classifier","m"]).pr_auc.mean().reset_index()
    R = {c: lvl[lvl.classifier==c].pivot(index="dataset",columns="m",
         values="pr_auc")[MAIN].rank(axis=1,ascending=False) for c in CLFS}
    order = R["HGB"].mean().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(STYLE["text_width"], 6.4), sharex=True)
    for ax, c in zip(axes.ravel(), CLFS):
        bp = ax.boxplot([R[c][m].values for m in order], vert=False, widths=0.65,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.2))
        for i, p in enumerate(bp["boxes"]):
            ours = order[i] == OURS
            p.set(facecolor="#bcd4ea" if ours else "#efefef",
                  edgecolor="#1f4e79" if ours else "#909090",
                  lw=1.3 if ours else 0.8)
        ax.plot([R[c][m].mean() for m in order], range(1, len(order)+1), "D",
                ms=3.8, color="#b23b3b", zorder=6)
        ax.set_yticks(range(1, len(order)+1))
        ax.set_yticklabels([DISP.get(m, m) for m in order], fontsize=8)
        for lab, m in zip(ax.get_yticklabels(), order):
            if m == OURS: lab.set_fontweight("bold")
        ax.set_title(c, fontsize=9.5, pad=6)
        ax.grid(axis="x", alpha=0.2, lw=0.5); ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("rank (lower is better)", fontsize=9)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    fig.savefig(fname); plt.close(fig); print("wrote", fname)


# ------------------------------------------------- Fig: sensitivity
def fig_sensitivity(fname="figures/fig_sensitivity.pdf"):
    df = pd.read_csv(results_path("tuning_results.csv"))
    for p, ch in [("k","k"),("alpha","a"),("q","q"),("beta","b")]:
        df[p] = [float(re.search(rf"{ch}(-?[\d.]+)", m).group(1)) for m in df.method]

    fig, axes = plt.subplots(1, 2, figsize=(STYLE["text_width"], 3.0))
    ax = axes[0]
    styles = {"alpha":("#1f4e79","o","-",r"$\alpha$",1.8,4.5),
              "k":("#6b6b6b","s","--","$k$",1.0,3.5),
              "q":("#9a9a9a","^",":","$q$",1.0,3.5),
              "beta":("#bcbcbc","v",":",r"$\beta$",1.0,3.5)}
    for p,(c,mk,ls,lab,lw,ms) in styles.items():
        s = df.groupby(p).pr_auc.mean()
        ax.plot(np.linspace(0,1,len(s)), s.values, ls, marker=mk, color=c,
                ms=ms, lw=lw, label=lab, zorder=4 if p=="alpha" else 2)
    ax.set_xlabel("normalized parameter range"); ax.set_ylabel("mean PR-AUC")
    ax.set_title("(a) marginal parameter effects", fontsize=9.5, pad=8)
    ax.legend(fontsize=8.5, ncol=4, handlelength=1.8, columnspacing=1.0,
              loc="upper left", frameon=False)
    ax.grid(alpha=0.2, lw=0.5); ax.tick_params(labelsize=8)

    ax = axes[1]
    cols = {"haberman":"#c0392b","ecoli3":"#e67e22","yeast3":"#7f8c8d",
            "glass0":"#2980b9","abalone19":"#16a085"}
    for ds, g in df.groupby("dataset"):
        s = g.groupby("alpha").pr_auc.mean(); s = s - s.mean()
        ax.plot(s.index, s.values, marker="o", ms=4, lw=1.6,
                color=cols.get(ds,"k"), solid_capstyle="round")
        ax.annotate(ds, (s.index[-1], s.values[-1]), xytext=(5,0),
                    textcoords="offset points", fontsize=8,
                    color=cols.get(ds,"k"), va="center")
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(r"anchor temperature $\alpha$"); ax.set_ylabel("centred PR-AUC")
    ax.set_title(r"(b) $\alpha$ effect per dataset", fontsize=9.5, pad=8)
    ax.set_xticks(sorted(df.alpha.unique())); ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, lw=0.5); ax.set_xlim(-1.3, 2.9)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(fname); plt.close(fig); print("wrote", fname)


# ------------------------------------------------- Fig: reliability
def fig_reliability(fname="figures/fig_reliability.pdf"):
    df = _load_eval()
    tot = df.groupby("m").size()
    fail = df[df.fallback == True].groupby("m").size()
    rate = (fail.reindex(tot.index).fillna(0) / tot * 100).sort_values()
    fig, ax = plt.subplots(figsize=(STYLE["text_width"]*0.62, 3.4))
    colors = ["#1f4e79" if m.startswith(OURS) else
              ("#b23b3b" if v > 0 else "#8a8a8a") for m, v in rate.items()]
    ax.barh(range(len(rate)), rate.values, color=colors, height=0.7)
    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels([DISP.get(m, m) for m in rate.index], fontsize=8)
    for i, v in enumerate(rate.values):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=7.5)
    ax.set_xlabel("fold-level failure rate (%)", fontsize=9)
    ax.grid(axis="x", alpha=0.2, lw=0.5); ax.tick_params(labelsize=8)
    ax.set_xlim(0, max(rate.values)*1.18)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig); print("wrote", fname)


# ------------------------------------------------- Fig: fidelity gradient
def fig_fidelity(fname="figures/fig_fidelity.pdf"):
    df = pd.read_csv(results_path("fidelity_results.csv"))
    df = canonicalize(df)
    lvl = df.groupby(["dataset","method"])[["mmd2","c2st"]].mean()
    order = lvl["mmd2"].groupby("method").mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(STYLE["text_width"], 3.0))
    for ax, col, lab in [(axes[0],"mmd2",r"held-out MMD$^2$"),
                         (axes[1],"c2st","C2ST AUC")]:
        data = [lvl[col].xs(m, level="method").dropna().values for m in order]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.1))
        for i, p in enumerate(bp["boxes"]):
            ours = order[i] == OURS
            p.set(facecolor="#bcd4ea" if ours else "#efefef",
                  edgecolor="#1f4e79" if ours else "#909090",
                  lw=1.2 if ours else 0.8)
        ax.plot(range(1,len(order)+1), [np.mean(d) for d in data], "D",
                ms=3.8, color="#b23b3b", zorder=6)
        ax.set_xticks(range(1,len(order)+1))
        ax.set_xticklabels([DISP.get(m,m) for m in order], rotation=30,
                           ha="right", fontsize=8)
        ax.set_ylabel(lab, fontsize=9); ax.grid(axis="y", alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[1].axhline(0.5, ls="--", color="k", lw=0.7, alpha=0.6)
    fig.tight_layout(w_pad=2.0); fig.savefig(fname); plt.close(fig)
    print("wrote", fname)


# ------------------------------------------------- Fig: frontier
def fig_frontier(fname="figures/fig_frontier.pdf"):
    df = pd.read_csv(results_path("alpha_sweep_results.csv"))
    df = canonicalize(df)
    lvl = df.groupby("method")[["mmd2","safety"]].mean()
    sweep = sorted([m for m in lvl.index if m.startswith(OURS) or "a=" in m],
                   key=lambda m: float(re.search(r"a=([+-][\d.]+)", m).group(1))
                   if re.search(r"a=([+-][\d.]+)", m) else 0)
    fig, ax = plt.subplots(figsize=(STYLE["text_width"]*0.62, 3.6))
    ax.plot(lvl.loc[sweep,"mmd2"], lvl.loc[sweep,"safety"], "o-",
            color="#1f4e79", lw=1.8, ms=6, zorder=3, label=rf"{OURS} ($\alpha$ sweep)")
    for m in sweep:
        mt = re.search(r"a=([+-][\d.]+)", m)
        if mt:
            ax.annotate(rf"$\alpha={mt.group(1)}$",
                        (lvl.loc[m,"mmd2"], lvl.loc[m,"safety"]),
                        textcoords="offset points", xytext=(7,4), fontsize=8)
    for m, mk in [("ROS","s"),("SMOTE","^"),("Borderline-SMOTE","v"),("ADASYN","D")]:
        if m in lvl.index:
            ax.scatter(lvl.loc[m,"mmd2"], lvl.loc[m,"safety"], marker=mk, s=55,
                       color="#b23b3b", zorder=4)
            ax.annotate(m, (lvl.loc[m,"mmd2"], lvl.loc[m,"safety"]),
                        textcoords="offset points", xytext=(7,-9),
                        fontsize=8, color="#b23b3b")
    ax.set_xlabel(r"held-out MMD$^2$ (fidelity cost $\rightarrow$)", fontsize=9)
    ax.set_ylabel(r"synthetic$\rightarrow$majority clearance", fontsize=9)
    ax.grid(alpha=0.2, lw=0.5); ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig); print("wrote", fname)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="one of: box,sens,rel,fid,front")
    a = ap.parse_args()
    jobs = {"box":fig_rank_boxplots, "sens":fig_sensitivity,
            "rel":fig_reliability, "fid":fig_fidelity, "front":fig_frontier}
    for k, fn in jobs.items():
        if a.only and a.only != k: continue
        try: fn()
        except FileNotFoundError as e: print("skip", k, "-", e)
        except Exception as e: print("FAIL", k, "-", type(e).__name__, e)
