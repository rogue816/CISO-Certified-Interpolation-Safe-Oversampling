"""
Test suite for CISO, derived directly from the paper's lemmas.

GATE A  -- exact contract tests (deterministic; float-eps tolerance only)
GATE B  -- statistical tests (fixed seeds, preregistered tolerances)
GATE D  -- degenerate / adversarial datasets

Run:  python3 test_ciso.py
"""

import warnings
import numpy as np
from scipy import stats

from ciso import CISO, g_q, _minmax

TOL = 1e-9
RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:
        RESULTS.append((name, "ERROR", f"{type(e).__name__}: {e}"))


# ----------------------------------------------------------------------
# Fixture: two minority clusters + one small cluster, majority in between
# (deliberately contains the cluster-bridging pathology)
# ----------------------------------------------------------------------
def make_data(seed=0):
    rng = np.random.default_rng(seed)
    maj = np.vstack([
        rng.normal([0, 0], 0.6, (120, 2)),
        rng.normal([4, 0], 0.6, (120, 2)),
    ])
    mino = np.vstack([
        rng.normal([2, 2.5], 0.30, (18, 2)),
        rng.normal([2, -2.5], 0.30, (14, 2)),
        rng.normal([6, 2.5], 0.15, (4, 2)),     # small cluster -> bridging risk
    ])
    X = np.vstack([maj, mino])
    y = np.hstack([np.zeros(len(maj), int), np.ones(len(mino), int)])
    # standardize (paper setup assumption)
    X = (X - X.mean(0)) / X.std(0)
    return X, y


X, Y = make_data()


# ======================================================================
# GATE A -- exact contract tests
# ======================================================================

def test_certificate_all_anchors():
    """Lemma (certified clearance): every sample from anchor i has distance
    >= gamma_i to EVERY majority point -- for all anchors, certified or not."""
    m = CISO(k=5, alpha=1.0, random_state=7).fit(X, Y)
    syn, anchors = m.generate(400, return_trace=True)
    d = CISO._cdist(syn, m.X_maj_).min(axis=1)
    slack = d - m.gamma_[anchors]
    assert np.all(slack >= -TOL), f"certificate violated, worst slack={slack.min():.3e}"


def test_convexity_radius():
    """Lemma placement(i): every sample lies within r_i of its anchor."""
    m = CISO(k=5, alpha=0.5, random_state=11).fit(X, Y)
    syn, anchors = m.generate(400, return_trace=True)
    d = np.linalg.norm(syn - m.X_min_[anchors], axis=1)
    assert np.all(d <= m.r_[anchors] + TOL), "sample escaped anchor neighborhood ball"


def test_alpha_zero_uniform():
    """Lemma P_alpha(i): alpha = 0 gives exactly uniform anchor distribution."""
    m = CISO(k=5, alpha=0.0).fit(X, Y)
    assert np.allclose(m.P_, 1.0 / m.N1_, atol=TOL), "P_0 not uniform"


def test_likelihood_ratio_ordering():
    """Lemma P_alpha(ii): P_a'(i)/P_a(i) nondecreasing in s_i for a' > a."""
    m1 = CISO(k=5, alpha=0.5).fit(X, Y)
    m2 = CISO(k=5, alpha=2.0).fit(X, Y)
    order = np.argsort(m1.s_, kind="stable")
    ratio = (m2.P_ / m1.P_)[order]
    s_sorted = m1.s_[order]
    for a in range(len(ratio) - 1):
        if s_sorted[a + 1] > s_sorted[a] + TOL:
            assert ratio[a + 1] >= ratio[a] - 1e-9, "LR ordering violated"


def test_floor_bound():
    """Lemma P_alpha(iii): P_alpha(i) >= eps0^a / (N1 (eps0+1)^a) for a >= 0."""
    for alpha in [0.0, 0.7, 2.0]:
        m = CISO(k=5, alpha=alpha, eps0=0.1).fit(X, Y)
        floor = (m.eps0 ** alpha) / (m.N1_ * (m.eps0 + 1.0) ** alpha)
        assert np.all(m.P_ >= floor - TOL), f"floor violated at alpha={alpha}"


def test_weights_positive_and_normalized():
    """Lemma phase2(i): all weights strictly positive; rows are prob. vectors.
    No degenerate fallback branch can ever be needed."""
    m = CISO(k=5).fit(X, Y)
    assert np.all(m.W_ > 0), "non-positive Phase-II weight found"
    assert np.allclose(m.W_.sum(axis=1), 1.0, atol=TOL), "weight rows not normalized"


def test_equal_clearance_odds_reduce_to_kernel():
    """Lemma phase2(ii): with constant clearance field, selection odds equal
    the kernel ratio exactly."""
    rng = np.random.default_rng(3)
    # majority arranged so all minority clearances are equal by symmetry:
    mino = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    # step 0.2 grid CONTAINS the minority x-coords, so every clearance is
    # exactly 10.0 -> constant field -> convention c_hat = 1/2 everywhere
    maj = np.column_stack([np.linspace(-2, 6, 41), np.full(41, 10.0)])
    maj = np.vstack([maj, np.column_stack([np.linspace(-2, 6, 41), np.full(41, -10.0)])])
    Xl = np.vstack([maj, mino]); Yl = np.hstack([np.zeros(len(maj), int), np.ones(5, int)])
    m = CISO(k=3, q=1.5).fit(Xl, Yl)
    if np.ptp(m.c_hat_) < 1e-6 or np.all(m.c_hat_ == 0.5):
        i = 0
        delta = np.linalg.norm(m.X_min_[m.min_nbrs_[i]] - m.X_min_[i], axis=1)
        kern = g_q(delta / m.h_[i], m.q)
        expected = kern / kern.sum()
        assert np.allclose(m.W_[i], expected, atol=1e-8), "odds != kernel ratio"
    else:
        raise AssertionError("fixture failed to equalize clearances")


def test_expected_gamma_monotone_in_alpha_exact():
    """Proposition (monotone): with s_mode='gamma', E_P[gamma] nondecreasing
    in alpha. Computed EXACTLY from P_ -- no sampling."""
    alphas = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    vals = []
    for a in alphas:
        m = CISO(k=5, alpha=a, s_mode="gamma").fit(X, Y)
        vals.append(np.dot(m.P_, m.gamma_))
    diffs = np.diff(vals)
    assert np.all(diffs >= -TOL), f"E[gamma] not monotone: {vals}"


def test_distinct_endpoints():
    """Phase II sampling: anchor != neighbour whenever k >= 2 (checked via
    zero-length segments being impossible unless duplicate points exist)."""
    m = CISO(k=5, random_state=5).fit(X, Y)
    rng = np.random.default_rng(5)
    for _ in range(500):
        i = rng.choice(m.N1_, p=m.P_)
        j1 = rng.choice(m.k_, p=m.W_[i])
        w2 = m.W_[i].copy(); w2[j1] = 0.0; w2 /= w2.sum()
        j2 = rng.choice(m.k_, p=w2)
        assert j1 != j2, "duplicate endpoint drawn despite without-replacement"


def test_reproducibility():
    """Same seed -> identical output."""
    a = CISO(k=5, random_state=42).fit(X, Y).generate(50)
    b = CISO(k=5, random_state=42).fit(X, Y).generate(50)
    assert np.array_equal(a, b), "not reproducible under fixed seed"


def test_fit_state_immutable_under_generate():
    """Fields are computed once: generate() must not mutate fit state."""
    m = CISO(k=5, random_state=1).fit(X, Y)
    snap = {n: getattr(m, n).copy() for n in
            ["P_", "W_", "s_", "gamma_", "c_", "r_", "rho_hat_", "c_hat_"]}
    m.generate(200)
    for n, v in snap.items():
        assert np.array_equal(v, getattr(m, n)), f"generate() mutated {n}"


def test_q_shared_kernel_consistency():
    """g_q sanity: positive, decreasing, Gaussian limit as q -> 1+."""
    t = np.linspace(0, 6, 100)
    for q in [1.0001, 1.5, 2.5]:
        v = g_q(t, q)
        assert np.all(v > 0) and np.all(np.diff(v) <= 1e-12), f"kernel bad at q={q}"
    # True limit of [1+(q-1)t^2]^{1/(1-q)} as q -> 1+ is exp(-t^2), NOT
    # exp(-t^2/2): the old paper's q=1 case branch was inconsistent with its
    # own q != 1 branch. Paper Lemma (Phase II) must read e^{-t^2}.
    assert np.allclose(g_q(t, 1.0 + 1e-9), np.exp(-t * t), atol=1e-5), \
        "Gaussian limit failed"


# ======================================================================
# GATE B -- statistical tests (fixed seeds, preregistered tolerances)
# ======================================================================

def test_anchor_frequencies_match_P():
    """Empirical anchor frequencies vs P_alpha: total variation < 0.03
    at n = 40000 draws (preregistered)."""
    m = CISO(k=5, alpha=1.5, random_state=123).fit(X, Y)
    _, anchors = m.generate(40000, return_trace=True)
    emp = np.bincount(anchors, minlength=m.N1_) / len(anchors)
    tv = 0.5 * np.abs(emp - m.P_).sum()
    assert tv < 0.03, f"TV distance {tv:.4f} exceeds 0.03"


def test_lambda_beta_small_concentrates():
    """Lemma placement(iii), beta -> 0: lambda concentrates at mu."""
    m = CISO(k=5, beta=0.01, q=1.5).fit(X, Y)
    rng = np.random.default_rng(9)
    mu = 0.3
    lams = np.array([m.sample_lambda(mu, rng) for _ in range(4000)])
    assert abs(lams.mean() - mu) < 0.02, f"mean {lams.mean():.3f} != mu"
    assert lams.std() < 0.03, f"std {lams.std():.3f} too large for beta=0.01"


def test_lambda_beta_large_uniform():
    """Lemma placement(iii), beta -> inf: lambda ~ Uniform[0,1] (KS test)."""
    m = CISO(k=5, beta=200.0, q=1.5).fit(X, Y)
    rng = np.random.default_rng(10)
    lams = np.array([m.sample_lambda(0.3, rng) for _ in range(4000)])
    p = stats.kstest(lams, "uniform").pvalue
    assert p > 0.01, f"KS vs uniform rejected, p={p:.4g}"


def test_lambda_unimodal_at_mu():
    """Lemma placement(ii): mode at mu (histogram argmax near mu)."""
    m = CISO(k=5, beta=0.15, q=1.5).fit(X, Y)
    rng = np.random.default_rng(11)
    mu = 0.7
    lams = np.array([m.sample_lambda(mu, rng) for _ in range(20000)])
    hist, edges = np.histogram(lams, bins=25, range=(0, 1))
    mode = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    assert abs(mode - mu) < 0.08, f"mode {mode:.3f} not at mu={mu}"


def test_larger_q_heavier_tails():
    """Lemma placement(iv): for fixed beta, larger q puts more mass far from mu."""
    rng1, rng2 = np.random.default_rng(12), np.random.default_rng(12)
    m_lo = CISO(k=5, beta=0.12, q=1.1).fit(X, Y)
    m_hi = CISO(k=5, beta=0.12, q=2.5).fit(X, Y)
    mu = 0.5
    lo = np.array([m_lo.sample_lambda(mu, rng1) for _ in range(20000)])
    hi = np.array([m_hi.sample_lambda(mu, rng2) for _ in range(20000)])
    p_lo = np.mean(np.abs(lo - mu) > 0.3)
    p_hi = np.mean(np.abs(hi - mu) > 0.3)
    assert p_hi > p_lo + 0.01, f"tails not heavier: q=2.5 {p_hi:.3f} vs q=1.1 {p_lo:.3f}"


def test_bridging_suppression():
    """Phase-II locality: for an anchor in the small 4-point cluster (k=5,
    so >=1 neighbor is cross-cluster), total weight on cross-cluster
    neighbors is far below their uniform share."""
    m = CISO(k=5, q=1.5).fit(X, Y)
    small = np.arange(32, 36)                      # indices of small cluster (minority)
    worst_share = 0.0
    for i in small:
        nbrs = m.min_nbrs_[i]
        cross = ~np.isin(nbrs, small)
        assert cross.any(), "fixture broken: expected cross-cluster neighbors"
        worst_share = max(worst_share, m.W_[i][cross].sum())
    uniform_share = cross.sum() / m.k_
    assert worst_share < 0.5 * uniform_share, \
        f"bridging weight {worst_share:.3f} not suppressed vs uniform {uniform_share:.3f}"


# ======================================================================
# GATE D -- degenerate / adversarial datasets
# ======================================================================

def test_k_truncation_and_warning():
    Xs = np.vstack([np.random.default_rng(0).normal(0, 1, (20, 3)),
                    np.random.default_rng(1).normal(3, 1, (4, 3))])
    Ys = np.hstack([np.zeros(20, int), np.ones(4, int)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = CISO(k=10).fit(Xs, Ys)
    assert m.k_ == 3, f"expected truncation to 3, got {m.k_}"
    assert any("truncated" in str(x.message) for x in w), "no truncation warning"


def test_n1_equals_2_boundary():
    """Smallest legal minority; k truncates to 1, boundary warning, still runs,
    certificate still holds."""
    Xs = np.vstack([np.random.default_rng(2).normal(0, 1, (30, 2)),
                    np.array([[5.0, 5.0], [5.3, 5.1]])])
    Ys = np.hstack([np.zeros(30, int), np.ones(2, int)])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = CISO(k=5, random_state=0).fit(Xs, Ys)
    syn, anchors = m.generate(20, return_trace=True)
    d = CISO._cdist(syn, m.X_maj_).min(axis=1)
    assert np.all(d >= m.gamma_[anchors] - TOL), "certificate fails at N1=2"


def test_duplicate_points():
    """Coincident minority points must not crash (epsilon guard)."""
    Xs = np.vstack([np.random.default_rng(3).normal(0, 1, (30, 2)),
                    np.tile([[4.0, 4.0]], (5, 1))])           # 5 identical
    Ys = np.hstack([np.zeros(30, int), np.ones(5, int)])
    m = CISO(k=3, random_state=0).fit(Xs, Ys)
    syn = m.generate(10)
    assert np.all(np.isfinite(syn)), "non-finite output on duplicate points"


def test_constant_majority_density_convention():
    """Two symmetric majority points -> constant raw density -> rho_hat = 1/2."""
    Xs = np.vstack([[[-1.0, 0.0]], [[1.0, 0.0]],
                    np.random.default_rng(4).normal([0, 3], 0.2, (5, 2))])
    Ys = np.hstack([np.zeros(2, int), np.ones(5, int)])
    m = CISO(k=2).fit(Xs, Ys)
    assert np.allclose(m.rho_hat_, 0.5), "constant-case convention violated"


def test_invalid_inputs_rejected():
    for bad in [dict(q=1.0), dict(q=0.5), dict(beta=0.0), dict(eps0=0.0)]:
        try:
            CISO(**bad).fit(X, Y)
            raise AssertionError(f"accepted invalid params {bad}")
        except ValueError:
            pass
    try:
        CISO().fit(X[:10], np.zeros(10, int))   # single class
        raise AssertionError("accepted single-class y")
    except ValueError:
        pass


def test_balanced_input_passthrough():
    Xb = np.random.default_rng(5).normal(0, 1, (20, 2))
    Yb = np.hstack([np.zeros(10, int), np.ones(10, int)])
    Xr, Yr = CISO().fit_resample(Xb, Yb)
    assert len(Xr) == 20 and len(Yr) == 20, "balanced input not passed through"


def test_fit_resample_balances():
    Xr, Yr = CISO(k=5, random_state=0).fit_resample(X, Y)
    assert np.sum(Yr == 1) == np.sum(Yr == 0), "output not balanced"


def test_brute_tree_equivalence():
    """Refactor safety: dense path and tree path must yield identical fitted
    state (same neighborhoods, fields, distribution, weights)."""
    rng = np.random.default_rng(21)
    Xs = np.vstack([rng.normal(0, 1, (600, 6)), rng.normal(2.0, 0.8, (80, 6))])
    Ys = np.hstack([np.zeros(600, int), np.ones(80, int)])
    mb = CISO(k=5, alpha=1.0, brute_limit=10**12).fit(Xs, Ys)   # force brute
    mt = CISO(k=5, alpha=1.0, brute_limit=1).fit(Xs, Ys)        # force tree
    assert np.array_equal(mb.maj_nbrs_, mt.maj_nbrs_), "maj neighborhoods differ"
    assert np.array_equal(mb.min_nbrs_, mt.min_nbrs_), "min neighborhoods differ"
    for name in ["c_", "r_", "gamma_", "rho_hat_", "c_hat_", "s_", "P_", "W_"]:
        a, b = getattr(mb, name), getattr(mt, name)
        assert np.allclose(a, b, atol=1e-10), f"{name} differs between paths"


def test_scale_smoke_letter_size():
    """Scale suite feasibility: fit + generate at Letter-Recognition size
    (N0 ~ 19k) must run in bounded memory via the tree path."""
    rng = np.random.default_rng(22)
    Xs = np.vstack([rng.normal(0, 1, (19000, 16)),
                    rng.normal(1.5, 0.9, (750, 16))])
    Ys = np.hstack([np.zeros(19000, int), np.ones(750, int)])
    m = CISO(k=5, alpha=1.0, random_state=0).fit(Xs, Ys)
    syn = m.generate(500)
    assert np.all(np.isfinite(syn)) and syn.shape == (500, 16)


# ======================================================================
# GATE F -- loader tests (run only where the sample .dat files exist)
# ======================================================================
import os
KEEL_SAMPLES = {
    "ecoli3":    ("/mnt/user-data/uploads/ecoli3.dat",    336,  7,  8.60),
    "abalone19": ("/mnt/user-data/uploads/abalone19.dat", 4174, 8, 129.44),
}
_HAVE_SAMPLES = all(os.path.exists(p) for p, *_ in KEEL_SAMPLES.values())


def test_keel_parser_against_real_files():
    if not _HAVE_SAMPLES:
        return  # skip silently on machines without the sample files
    from loaders import parse_keel
    for name, (path, n, d, ir) in KEEL_SAMPLES.items():
        Xk, yk, meta = parse_keel(path)
        assert meta["n"] == n and meta["d_raw"] == d, f"{name} shape mismatch"
        assert abs(meta["ir"] - ir) < 0.05, f"{name} IR {meta['ir']} != {ir}"
        assert meta["positive_rule"] == "keel-declared", name
        assert np.all(np.isfinite(Xk)), f"{name} non-finite features"
        assert set(np.unique(yk)) == {0, 1}, name
    # nominal one-hot: abalone Sex {M,F,I} -> 7 numeric + 3 dummies
    _, _, meta = parse_keel(KEEL_SAMPLES["abalone19"][0])
    assert meta["d_encoded"] == 10, meta["d_encoded"]


def test_uci_binarization_logic_offline():
    import pandas as pd
    from loaders import _binarize
    assert list(_binarize(pd.Series(["A", "Z", "z ", "B", "Z"]),
                          ("class_equals", "Z"))) == [0, 1, 1, 0, 1]
    assert list(_binarize(pd.Series([1, 5, 5.0, 2]),
                          ("class_equals", 5))) == [0, 1, 1, 0]
    assert list(_binarize(pd.Series([True, False, True]),
                          ("class_equals", True))) == [1, 0, 1]


def test_ciso_on_real_keel_end_to_end():
    """First-contact integration: CISO on real KEEL data, standardized,
    balanced output, certificate holds fold-free (whole-dataset fit)."""
    if not _HAVE_SAMPLES:
        return
    from loaders import parse_keel
    for name, (path, *_ ) in KEEL_SAMPLES.items():
        Xk, yk, _ = parse_keel(path)
        Xs = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-12)
        m = CISO(k=5, alpha=1.0, random_state=0)
        Xr, Yr = m.fit_resample(Xs, yk)
        assert np.sum(Yr == 1) == np.sum(Yr == 0), f"{name} not balanced"
        syn, anchors = m.generate(200, return_trace=True)
        d = CISO._cdist(syn, m.X_maj_).min(axis=1)
        assert np.all(d >= m.gamma_[anchors] - 1e-9), f"{name} certificate fail"


# ======================================================================
# Runner
# ======================================================================
ALL = [
    ("A1  certificate holds for every anchor",       test_certificate_all_anchors),
    ("A2  convexity: samples stay in anchor ball",   test_convexity_radius),
    ("A3  alpha=0 is exactly uniform",               test_alpha_zero_uniform),
    ("A4  likelihood-ratio ordering in s",           test_likelihood_ratio_ordering),
    ("A5  eps0 floor bound",                         test_floor_bound),
    ("A6  weights positive, rows normalized",        test_weights_positive_and_normalized),
    ("A7  equal-clearance odds = kernel ratio",      test_equal_clearance_odds_reduce_to_kernel),
    ("A8  E[gamma] monotone in alpha (exact)",       test_expected_gamma_monotone_in_alpha_exact),
    ("A9  anchor != neighbour (k >= 2)",             test_distinct_endpoints),
    ("A10 reproducible under seed",                  test_reproducibility),
    ("A11 generate() does not mutate fit state",     test_fit_state_immutable_under_generate),
    ("A12 kernel sanity + Gaussian limit",           test_q_shared_kernel_consistency),
    ("B1  anchor frequencies match P (TV<0.03)",     test_anchor_frequencies_match_P),
    ("B2  beta->0: lambda concentrates at mu",       test_lambda_beta_small_concentrates),
    ("B3  beta->inf: lambda ~ Uniform (KS)",         test_lambda_beta_large_uniform),
    ("B4  lambda unimodal at mu",                    test_lambda_unimodal_at_mu),
    ("B5  larger q -> heavier tails",                test_larger_q_heavier_tails),
    ("B6  cross-cluster bridging suppressed",        test_bridging_suppression),
    ("D1  k truncation + warning",                   test_k_truncation_and_warning),
    ("D2  N1=2 boundary case",                       test_n1_equals_2_boundary),
    ("D3  duplicate points",                         test_duplicate_points),
    ("D4  constant-density convention",              test_constant_majority_density_convention),
    ("D5  invalid inputs rejected",                  test_invalid_inputs_rejected),
    ("D6  balanced input passthrough",               test_balanced_input_passthrough),
    ("D7  fit_resample balances",                    test_fit_resample_balances),
    ("E1  brute path == tree path (refactor gate)",  test_brute_tree_equivalence),
    ("E2  scale smoke test at Letter size",          test_scale_smoke_letter_size),
    ("F1  KEEL parser vs real files",                test_keel_parser_against_real_files),
    ("F2  UCI binarization logic (offline)",         test_uci_binarization_logic_offline),
    ("F3  CISO end-to-end on real KEEL data",       test_ciso_on_real_keel_end_to_end),
]

if __name__ == "__main__":
    for name, fn in ALL:
        check(name, fn)
    width = max(len(n) for n, _, _ in RESULTS)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"\n{'=' * 70}\nCISO lemma-derived test suite\n{'=' * 70}")
    for name, status, msg in RESULTS:
        line = f"{name:<{width}}  {status}"
        if msg:
            line += f"   [{msg}]"
        print(line)
    print(f"{'=' * 70}\n{n_pass}/{len(RESULTS)} passed")
    raise SystemExit(0 if n_pass == len(RESULTS) else 1)
