"""Gate C (v2): visual inspection on a fixture WITH class overlap.

Design principle: the mechanism is only visible where safety is scarce.
This fixture embeds part of the minority class inside the majority blob,
so the safety field has genuine contrast and the alpha regimes must differ.

Panels:
  (1) safety field + certified anchors  -- should show a gradient now
  (2) ECDF of synthetic distance-to-nearest-majority for three alphas
      -- the quantitative signature of the alpha dial (curves must order)
  (3-5) synthesis scatter at alpha = +2, 0, -2
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ciso import CISO


def make_overlap_data(seed=1):
    rng = np.random.default_rng(seed)
    maj = rng.normal([0.0, 0.0], [1.3, 1.3], (260, 2))
    mino = np.vstack([
        rng.normal([1.4, 0.3], 0.55, (16, 2)),    # embedded: heavy overlap
        rng.normal([3.2, -0.4], 0.45, (14, 2)),   # frontier: partial overlap
        rng.normal([5.0, 1.2], 0.35, (12, 2)),    # safe: clearly separated
    ])
    X = np.vstack([maj, mino])
    y = np.hstack([np.zeros(len(maj), int), np.ones(len(mino), int)])
    X = (X - X.mean(0)) / X.std(0)
    return X, y


X, Y = make_overlap_data()

fig = plt.figure(figsize=(13.5, 8.5))
gs = fig.add_gridspec(2, 6)
ax_field = fig.add_subplot(gs[0, 0:3])
ax_ecdf = fig.add_subplot(gs[0, 3:6])
ax_a = [fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6])]

# ---- Panel 1: safety field ---------------------------------------------
m0 = CISO(k=5, alpha=1.0).fit(X, Y)
ax_field.scatter(*m0.X_maj_.T, c="0.8", s=12)
sc = ax_field.scatter(*m0.X_min_.T, c=m0.s_, cmap="viridis", s=60,
                      edgecolors="k", linewidths=0.5, vmin=0, vmax=m0.s_.max())
cert = m0.gamma_ > 0
ax_field.scatter(*m0.X_min_[cert].T, facecolors="none", edgecolors="crimson",
                 s=170, linewidths=1.4)
plt.colorbar(sc, ax=ax_field, shrink=0.85, label="safety score $s_i$")
ax_field.set_title(f"Safety field | certified anchors circled "
                   f"({cert.mean():.0%})")

# ---- Panels 2-5: three alpha regimes -----------------------------------
colors = {2.0: "tab:blue", 0.0: "tab:orange", -2.0: "tab:red"}
stats_lines = []
for ax, alpha in zip(ax_a, [2.0, 0.0, -2.0]):
    m = CISO(k=5, alpha=alpha, q=1.5, beta=0.2, random_state=0).fit(X, Y)
    syn = m.generate(300)
    d_syn = CISO._cdist(syn, m.X_maj_).min(axis=1)

    # scatter panel
    ax.scatter(*m.X_maj_.T, c="0.8", s=10)
    ax.scatter(*m.X_min_.T, c="k", s=26, zorder=3)
    ax.scatter(*syn.T, c=colors[alpha], marker="x", s=20, alpha=0.7, zorder=2)
    # honest intrusion count: syn points whose nearest ORIGINAL point is majority
    d_min = CISO._cdist(syn, m.X_min_).min(axis=1)
    frac_in = np.mean(d_syn < d_min)
    ax.set_title(rf"$\alpha={alpha:+.0f}$ | nearest-original-is-majority: "
                 rf"{frac_in:.0%}")
    ax.set_xticks([]); ax.set_yticks([])

    # ECDF curve
    xs = np.sort(d_syn)
    ax_ecdf.plot(xs, np.arange(1, len(xs) + 1) / len(xs),
                 color=colors[alpha], lw=2, label=rf"$\alpha={alpha:+.0f}$")
    stats_lines.append((alpha, d_syn.mean(), frac_in))

# reference: ORIGINAL minority distances to majority
d_orig = CISO._cdist(m0.X_min_, m0.X_maj_).min(axis=1)
xs = np.sort(d_orig)
ax_ecdf.plot(xs, np.arange(1, len(xs) + 1) / len(xs),
             color="k", lw=1.5, ls="--", label="original minority")
ax_ecdf.set_xlabel("distance to nearest majority instance")
ax_ecdf.set_ylabel("ECDF")
ax_ecdf.set_title("The $\\alpha$ dial, quantitatively:\n"
                  "curves must order right-to-left as $\\alpha$ decreases")
ax_ecdf.legend(fontsize=9)
ax_ecdf.grid(alpha=0.3)

fig.suptitle("CISO Gate C v2: overlap fixture -- where the mechanism has real work",
             fontsize=13)
fig.tight_layout()
fig.savefig("figures/toy_overlap.png", dpi=160, bbox_inches="tight")
print("saved gate_c_overlap.png")
print(f"certified fraction: {cert.mean():.3f}")
for a, dm, fi in stats_lines:
    print(f"alpha={a:+.0f}: mean dist-to-majority={dm:.3f}, "
          f"nearest-orig-is-majority={fi:.1%}")
