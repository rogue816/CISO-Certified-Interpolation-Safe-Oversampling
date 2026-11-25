"""
CISO: Certified Interpolation Safe Oversampling.

Direct implementation of the finalized mathematical blueprint. Every public
attribute corresponds to a numbered equation; every guarantee corresponds to
a test in test_ciso.py.

Structure mirrors the algorithm box:
    fit(X, y)    -> the "Precompute (once)" stage (Stages 0-1, Phase I & II tables)
    generate(n)  -> the per-sample loop (three draws + one affine combination)

Notation map (paper -> code):
    M_k(i), N_k(i)          -> self.maj_nbrs_, self.min_nbrs_   (index arrays)
    c_i, r_i                -> self.c_, self.r_                 (Eq. extremal)
    rho_hat (density field) -> self.rho_hat_                    (Eq. density_field)
    rho_Reg(i)              -> self.rho_reg_                    (Eq. rhoreg)
    c_hat (clearance field) -> self.c_hat_
    s_i                     -> self.s_                          (Eq. safety)
    P_alpha                 -> self.P_                          (Eq. anchor_dist)
    h_i, w~_ij              -> self.h_, self.W_                 (Eqs. medscale, phase2_weight)
    gamma_i                 -> self.gamma_                      (Lemma: certificate)

The caller is expected to standardize features beforehand (paper Sec. setup);
fit_resample does not scale.
"""

import warnings
import numpy as np


def _minmax(v, eps=0.0):
    """Min-max normalize to [0,1]; constant-case convention -> 1/2."""
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo <= eps:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def _stable_knn_indices(dist_row_matrix, k):
    """k smallest indices per row, ties broken by dataset index (stable sort)."""
    return np.argsort(dist_row_matrix, axis=1, kind="stable")[:, :k]


def _knn_query(A, B, k, self_query=False, brute_limit=4_000_000):
    """k nearest rows of B for each row of A. Returns (dist, idx), each (len(A), k).

    self_query=True means A is B and each point must exclude itself.
    Small problems (len(A)*len(B) <= brute_limit) use the exact dense path with
    stable index tie-breaking (protocol convention). Large problems use
    sklearn's tree/brute kNN with k + 5 over-query and per-row lexsort on
    (distance, index) to restore the stable tie-break among candidates.
    """
    nA, nB = len(A), len(B)
    kk = k + (1 if self_query else 0)
    if nA * nB <= brute_limit:
        aa = np.sum(A * A, axis=1)[:, None]
        bb = np.sum(B * B, axis=1)[None, :]
        D = np.sqrt(np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0))
        if self_query:
            np.fill_diagonal(D, np.inf)
        idx = _stable_knn_indices(D, k)
        dist = np.take_along_axis(D, idx, axis=1)
        return dist, idx

    from sklearn.neighbors import NearestNeighbors
    n_query = min(kk + 5, nB)                      # over-query for tie safety
    nn = NearestNeighbors(n_neighbors=n_query).fit(B)
    dist, idx = nn.kneighbors(A)
    out_d = np.empty((nA, k))
    out_i = np.empty((nA, k), dtype=int)
    for r in range(nA):
        d_r, i_r = dist[r], idx[r]
        if self_query:                              # drop one self occurrence
            hits = np.where(i_r == r)[0]
            keep = np.ones(len(i_r), bool)
            keep[hits[0] if len(hits) else len(i_r) - 1] = False
            d_r, i_r = d_r[keep], i_r[keep]
        order = np.lexsort((i_r, d_r))[:k]          # stable (dist, index) order
        out_d[r], out_i[r] = d_r[order], i_r[order]
    return out_d, out_i


def g_q(t, q):
    """q-Gaussian kernel G_q(t) = [1 + (q-1) t^2]^{1/(1-q)}, for q > 1.

    Strictly positive, strictly decreasing in |t|, Gaussian limit as q -> 1+.
    (Eq. qkernel)
    """
    t = np.asarray(t, dtype=float)
    return (1.0 + (q - 1.0) * t * t) ** (1.0 / (1.0 - q))


class CISO:
    """Certified Interpolation Safe Oversampling (CISO).

    Parameters
    ----------
    k : int            neighborhood size (both classes), truncated to feasibility
    alpha : float      anchor temperature; >0 safe-seeking, 0 uniform, <0 borderline
    eps0 : float       strictly positive floor in P_alpha and Phase-II clearance factor
    q : float          tail index, must be > 1 (shared by Phase II and III kernels)
    beta : float       Phase-III placement scale, fraction of segment length
    s_mode : str       'product' (Eq. safety, practical) or 'gamma' (exact variant
                       for Proposition monotone; used in ablation/tests)
    random_state : int or None
    """

    def __init__(self, k=5, alpha=1.0, eps0=0.1, q=1.5, beta=0.2,
                 s_mode="product", random_state=None,
                 brute_limit=4_000_000, phase2_clearance=True):
        self.k = k
        self.alpha = alpha
        self.eps0 = eps0
        self.q = q
        self.beta = beta
        self.s_mode = s_mode
        self.random_state = random_state
        self.brute_limit = brute_limit   # dense-path threshold (pair count)
        self.phase2_clearance = phase2_clearance  # ablation: neighbor-safety factor
        self._eps = 1e-12  # coincident-point guard (paper's epsilon)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self, X, y):
        if self.q <= 1:
            raise ValueError("q must be > 1 (heavy-tailed regime).")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.eps0 <= 0:
            raise ValueError("eps0 must be strictly positive.")
        if self.s_mode not in ("product", "gamma"):
            raise ValueError("s_mode must be 'product' or 'gamma'.")
        classes = np.unique(y)
        if not np.array_equal(np.sort(classes), np.array([0, 1])):
            raise ValueError("y must contain exactly the labels {0, 1}.")
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        if n1 < 2:
            raise ValueError(f"Need N1 >= 2 minority instances, got {n1}.")
        if n0 < 1:
            raise ValueError("Need at least one majority instance.")
        return n0, n1

    # ------------------------------------------------------------------
    # Precompute-once stage
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n0, n1 = self._validate(X, y)

        self.X_min_ = X[y == 1]
        self.X_maj_ = X[y == 0]
        self.N1_, self.N0_ = n1, n0
        self.delta_n_ = n0 - n1

        # Feasibility truncation: 1 <= k <= min(N1 - 1, N0)   (setup block)
        k = int(min(self.k, n1 - 1, n0))
        if k < 1:
            raise ValueError("Feasible k < 1; dataset too small.")
        if k < self.k:
            warnings.warn(f"k truncated from {self.k} to {k} for feasibility.")
        self.k_ = k
        if k == 1:
            warnings.warn("k == 1: anchor and neighbour coincide; Phase III "
                          "degenerates to that single neighbor (boundary case).")

        # --- Stage 0: neighborhoods and extremal statistics -----------
        d_mm, self.maj_nbrs_ = _knn_query(self.X_min_, self.X_maj_, k,
                                          brute_limit=self.brute_limit)
        d_nn, self.min_nbrs_ = _knn_query(self.X_min_, self.X_min_, k,
                                          self_query=True,
                                          brute_limit=self.brute_limit)
        self.c_ = d_mm[:, 0]                                     # clearance
        self.r_ = d_nn[:, -1]                                    # nbhd radius
        self.gamma_ = self.c_ - self.r_                          # certificate

        # --- Stage 1a: majority density field (Eq. density_field) -----
        if n0 >= 2:
            k_rho = min(k, n0 - 1)
            d_jj, _ = _knn_query(self.X_maj_, self.X_maj_, k_rho,
                                 self_query=True,
                                 brute_limit=self.brute_limit)
            dbar = d_jj.mean(axis=1)
            rho_raw = 1.0 / (dbar + self._eps)
            self.rho_hat_ = _minmax(rho_raw)
        else:
            self.rho_hat_ = np.array([0.5])                      # convention

        # --- Stage 1b: minority clearance field ------------------------
        self.c_hat_ = _minmax(self.c_)

        # --- Phase I: regional density, safety, anchor distribution ---
        self.rho_reg_ = self.rho_hat_[self.maj_nbrs_].mean(axis=1)  # Eq. rhoreg
        if self.s_mode == "product":
            self.s_ = self.c_hat_ * (1.0 - self.rho_reg_)            # Eq. safety
        else:  # 'gamma' -- exact variant for the monotonicity proposition
            self.s_ = _minmax(self.gamma_)
        base = (self.eps0 + self.s_) ** self.alpha                   # Eq. anchor_dist
        self.P_ = base / base.sum()

        # --- Phase II: per-anchor scale and weight matrix --------------
        delta = d_nn                                                 # N1 x k
        self.h_ = np.median(delta, axis=1) + self._eps               # Eq. medscale
        d_hat = delta / self.h_[:, None]
        kern = g_q(d_hat, self.q)
        if self.phase2_clearance:
            clearance_of_nbr = self.c_hat_[self.min_nbrs_]           # field read
            W = (self.eps0 + clearance_of_nbr) * kern                # Eq. phase2_weight
        else:                                    # ablation: locality kernel only
            W = kern.copy()
        self.W_ = W / W.sum(axis=1, keepdims=True)

        return self

    @staticmethod
    def _cdist(A, B):
        """Euclidean distance matrix (metric ell of the paper)."""
        aa = np.sum(A * A, axis=1)[:, None]
        bb = np.sum(B * B, axis=1)[None, :]
        sq = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
        return np.sqrt(sq)

    # ------------------------------------------------------------------
    # Phase III sampler (unit-testable in isolation)
    # ------------------------------------------------------------------
    def sample_lambda(self, mu, rng):
        """Draw lambda ~ f_i on [0,1]: truncated q-Gaussian centered at mu.

        Exact rejection sampling (Remark: sampling): since G_q <= 1, propose
        U[0,1], accept w.p. G_q((lambda - mu)/beta). Acceptance rate is
        bounded below by G_q(1/beta) > 0.
        """
        while True:
            lam = rng.random()
            if rng.random() <= g_q((lam - mu) / self.beta, self.q):
                return lam

    # ------------------------------------------------------------------
    # Per-sample stage
    # ------------------------------------------------------------------
    def generate(self, n=None, return_trace=False):
        """Generate n synthetic minority instances (default: Delta N)."""
        if n is None:
            n = self.delta_n_
        if n <= 0:
            empty = np.empty((0, self.X_min_.shape[1]))
            return (empty, np.empty(0, int)) if return_trace else empty

        rng = np.random.default_rng(self.random_state)
        k, N1 = self.k_, self.N1_
        out = np.empty((n, self.X_min_.shape[1]))
        anchors = np.empty(n, dtype=int)

        for t in range(n):
            i = rng.choice(N1, p=self.P_)                    # Phase I: where
            if k >= 2:                                       # Phase II: with whom
                j1 = rng.choice(k, p=self.W_[i])
                w2 = self.W_[i].copy()
                w2[j1] = 0.0
                w2 /= w2.sum()
                j2 = rng.choice(k, p=w2)
            else:
                j1 = j2 = 0                                  # flagged boundary case
            px = self.X_min_[self.min_nbrs_[i, j1]]
            orx = self.X_min_[self.min_nbrs_[i, j2]]

            a = self.X_min_[i]                               # Phase III: how far
            d_px = np.linalg.norm(a - px)
            d_or = np.linalg.norm(a - orx)
            tot = d_px + d_or
            mu = 0.5 if tot <= self._eps else d_px / tot     # Eq. center
            lam = self.sample_lambda(mu, rng)
            out[t] = (1.0 - lam) * px + lam * orx            # Eq. synthesis
            anchors[t] = i

        return (out, anchors) if return_trace else out

    def fit_resample(self, X, y):
        """imblearn-style interface. Returns (X_res, y_res)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if np.sum(y == 0) <= np.sum(y == 1):
            return X, y                                       # Delta N <= 0
        self.fit(X, y)
        X_syn = self.generate()
        X_res = np.vstack([X, X_syn])
        y_res = np.hstack([y, np.ones(len(X_syn), dtype=y.dtype)])
        return X_res, y_res

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self):
        """Reportable statistics from the Guarantees subsection."""
        return {
            "k_effective": self.k_,
            "certified_fraction": float(np.mean(self.gamma_ > 0)),
            "gamma_min": float(self.gamma_.min()),
            "gamma_mean": float(self.gamma_.mean()),
            "expected_gamma_under_P": float(np.dot(self.P_, self.gamma_)),
            "safety_score_range": (float(self.s_.min()), float(self.s_.max())),
        }


# ----------------------------------------------------------------------
# Backward-compatible alias.
# The method was named PO-qG in earlier drafts and in the frozen result
# CSVs under results/. The class was renamed to CISO; this alias keeps
# the archived scripts importable.
# ----------------------------------------------------------------------
POQG = CISO
