"""registry.py -- Protocol Section 4: the 18 methods, tiered.

Rules encoded here:
  * Library-backed baselines are NEVER replaced by homebrew stand-ins.
    If the library is missing, the method is marked unavailable and the
    engine records that fact instead of silently substituting.
  * Every method exposes the same interface:
        resample(X, y, seed) -> (X_res, y_res, info)
    where info may contain 'X_syn' (the synthetic block), 'fit_time',
    'gen_time', and CISO diagnostics.
  * kind='weights' methods do no resampling; the engine passes
    class_weight='balanced' (or sample_weight) to the classifier.
"""
import time
import numpy as np
from ciso import CISO
from labels import PROPOSED

try:
    import imblearn.over_sampling as ios
    import imblearn.combine as icomb
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import smote_variants as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False


class Method:
    def __init__(self, name, tier, factory=None, kind="sampler",
                 available=True, note=""):
        self.name, self.tier, self.kind = name, tier, kind
        self.factory, self.available, self.note = factory, available, note

    def resample(self, X, y, seed):
        if self.kind in ("none", "weights"):
            return X, y, {"X_syn": None, "resample_time": 0.0}
        t0 = time.time()
        sampler = self.factory(seed)
        Xr, yr = sampler.fit_resample(X, y)
        info = {"resample_time": time.time() - t0}
        n_orig = len(X)
        info["X_syn"] = Xr[n_orig:] if len(Xr) > n_orig else None
        if isinstance(sampler, CISO) and hasattr(sampler, "gamma_"):
            info["certified_fraction"] = float(np.mean(sampler.gamma_ > 0))
        return Xr, yr, info


def _ros_factory(seed):
    """Random oversampling -- dependency-free by design (Tier-1 null)."""
    class _ROS:
        def fit_resample(self, X, y):
            rng = np.random.default_rng(seed)
            n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
            deficit = n_neg - n_pos
            if deficit <= 0:
                return X, y
            idx_pos = np.where(y == 1)[0]
            pick = rng.choice(idx_pos, size=deficit, replace=True)
            return (np.vstack([X, X[pick]]),
                    np.hstack([y, np.ones(deficit, dtype=y.dtype)]))
    return _ROS()


def _ciso_factory(seed, **overrides):
    params = dict(k=5, alpha=1.0, q=1.5, beta=0.2, eps0=0.1)  # placeholder;
    params.update(overrides)              # tuned values injected after lock
    return CISO(random_state=seed, **params)


def build_registry(tuned=None):
    """tuned: dict of locked CISO hyperparameters (Protocol Section 10).
    Until the lock, defaults above are used and the engine tags results
    as pre-lock."""
    t = tuned or {}
    R = []

    # ---- Tier 1: nulls ------------------------------------------------
    R.append(Method("NoResampling", 1, kind="none"))
    R.append(Method("ClassWeights", 1, kind="weights"))
    R.append(Method("ROS", 1, factory=_ros_factory))

    # ---- Tier 2: traditional family (imblearn only, never homebrew) --
    def add_lib(name, tier, make, has, lib):
        R.append(Method(name, tier, factory=make if has else None,
                        available=has,
                        note="" if has else f"requires {lib}"))

    if HAS_IMBLEARN:
        add_lib("SMOTE", 2, lambda s: ios.SMOTE(k_neighbors=5, random_state=s),
                True, "imblearn")
        add_lib("Borderline-SMOTE", 2,
                lambda s: ios.BorderlineSMOTE(kind="borderline-2",
                                              k_neighbors=5, random_state=s),
                True, "imblearn")
        add_lib("ADASYN", 2, lambda s: ios.ADASYN(random_state=s),
                True, "imblearn")
        add_lib("SMOTE-ENN", 2, lambda s: icomb.SMOTEENN(random_state=s),
                True, "imblearn")
        add_lib("SMOTE-Tomek", 2, lambda s: icomb.SMOTETomek(random_state=s),
                True, "imblearn")
        add_lib("KMeans-SMOTE", 2,
                lambda s: ios.KMeansSMOTE(random_state=s), True, "imblearn")
    else:
        for n in ["SMOTE", "Borderline-SMOTE", "ADASYN", "SMOTE-ENN",
                  "SMOTE-Tomek", "KMeans-SMOTE"]:
            add_lib(n, 2, None, False, "imblearn")

    # ---- Tier 3: density/weight-aware relatives -----------------------
    if HAS_SV:
        add_lib("Safe-Level-SMOTE", 3,
                lambda s: _SVWrap(sv.Safe_Level_SMOTE, s), True, "smote_variants")
        add_lib("MWMOTE", 3, lambda s: _SVWrap(sv.MWMOTE, s), True,
                "smote_variants")
        add_lib("G-SMOTE", 3, lambda s: _SVWrap(sv.G_SMOTE, s), True,
                "smote_variants")
    else:
        for n in ["Safe-Level-SMOTE", "MWMOTE", "G-SMOTE"]:
            add_lib(n, 3, None, False, "smote_variants")

    # ---- Proposed + ablation ladder (Protocol Section 4, items 14-18) -
    R.append(Method(PROPOSED, 5, factory=lambda s: _ciso_factory(s, **t)))
    R.append(Method("CISO[alpha=0]", 5,
                    factory=lambda s: _ciso_factory(s, **{**t, "alpha": 0.0})))
    R.append(Method("CISO[uniform-lambda]", 5,
                    factory=lambda s: _ciso_factory(s, **{**t, "beta": 1e6})))
    R.append(Method("CISO[s=gamma]", 5,
                    factory=lambda s: _ciso_factory(s, **{**t,
                                                          "s_mode": "gamma"})))
    R.append(Method("CISO[no-clearance-II]", 5,
                    factory=lambda s: _ciso_factory(s, **{**t,
                                                          "phase2_clearance": False})))
    return R


class _SVWrap:
    """Adapter: smote_variants uses .sample(X, y)."""
    def __init__(self, cls, seed):
        self.obj = cls(random_state=seed)

    def fit_resample(self, X, y):
        return self.obj.sample(X, y)
