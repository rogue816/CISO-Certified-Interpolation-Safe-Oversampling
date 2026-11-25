"""Gate C visual inspection: CISO on the adversarial toy fixture."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ciso import CISO
from test_ciso import make_data

X, Y = make_data()
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# ---- Panel 1: safety field + certificates -----------------------------
m0 = CISO(k=5, alpha=1.0).fit(X, Y)
ax = axes[0, 0]
ax.scatter(*m0.X_maj_.T, c="0.75", s=14, label="majority")
sc = ax.scatter(*m0.X_min_.T, c=m0.s_, cmap="viridis", s=55,
                edgecolors="k", linewidths=0.5, label="minority (color = s)")
cert = m0.gamma_ > 0
ax.scatter(*m0.X_min_[cert].T, facecolors="none", edgecolors="crimson",
           s=160, linewidths=1.4, label=f"certified ({cert.mean():.0%})")
plt.colorbar(sc, ax=ax, shrink=0.8, label="safety score $s_i$")
ax.set_title("Safety field and certified anchors")
ax.legend(fontsize=8, loc="lower left")

# ---- Panels 2-4: synthesis under three alpha regimes ------------------
for ax, alpha, title in [
    (axes[0, 1],  2.0, r"$\alpha=+2$  (interior-seeking)"),
    (axes[1, 0],  0.0, r"$\alpha=0$  (uniform, SMOTE-like seeding)"),
    (axes[1, 1], -2.0, r"$\alpha=-2$  (borderline-seeking)"),
]:
    m = CISO(k=5, alpha=alpha, q=1.5, beta=0.2, random_state=0).fit(X, Y)
    syn = m.generate(180)
    ax.scatter(*m.X_maj_.T, c="0.75", s=14)
    ax.scatter(*m.X_min_.T, c="tab:red", s=40, edgecolors="k",
               linewidths=0.4, zorder=3)
    ax.scatter(*syn.T, c="tab:green", marker="x", s=26, alpha=0.8, zorder=2)
    ax.set_title(title)

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("CISO Gate C: toy fixture with deliberate small-cluster bridging trap",
             fontsize=12)
fig.tight_layout()
fig.savefig("figures/toy_certificate.png", dpi=160, bbox_inches="tight")
print("saved gate_c_toy.png")

d = m0.diagnostics()
print("\nDiagnostics:", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in d.items()})
