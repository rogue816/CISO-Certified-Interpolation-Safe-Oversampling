"""

This code implements the full experimental pipeline for Q-Regulated Oversampling (QRO).
Note: This code requires the KEEL datasets. 

"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from itertools import product

# =============================================================================
# plotting style
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 1,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

# =============================================================================
# QRO Class Definition (with per‑neighbor density)
# =============================================================================

class QRO(BaseEstimator, TransformerMixin):
    """
    Q-Regulated Oversampling (QRO) – formerly PO‑qG.
    """
    def __init__(self, k=5, q=1.5, beta=0.1, alpha=0.5, epsilon=1e-6, random_state=None):
        self.k = k
        self.q = q
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self._check_params(X, y)
        X_min = X[y == 1]
        X_maj = X[y == 0]
        n_min, n_maj = len(X_min), len(X_maj)
        n_synthetic = n_maj - n_min
        if n_synthetic <= 0:
            return X, y
        self.random_state_ = check_random_state(self.random_state)
        self.synthetic_samples_ = self._generate_samples(X_min, X_maj, n_synthetic)
        X_res = np.vstack([X, self.synthetic_samples_])
        y_res = np.hstack([y, np.ones(n_synthetic, dtype=y.dtype)])
        return X_res, y_res

    def _check_params(self, X, y):
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.q <= 1:
            raise ValueError("q must be > 1 for heavy-tailed behaviour")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")
        unique = np.unique(y)
        if len(unique) != 2:
            raise ValueError("Only binary classification is supported")
        if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
            raise ValueError("Both classes must be present")

    def _generate_samples(self, X_min, X_maj, n_synthetic):
        n_min = len(X_min)
        k = min(self.k, n_min - 1, len(X_maj))
        maj_density = self._majority_density(X_maj, k)
        nn_min = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        _, idx_min = nn_min.kneighbors(X_min, n_neighbors=k + 1)
        idx_min = idx_min[:, 1:]
        nn_maj = NearestNeighbors(n_neighbors=k).fit(X_maj)
        dist_maj, idx_maj = nn_maj.kneighbors(X_min, n_neighbors=k)
        synthetic = []
        for _ in range(n_synthetic):
            anchor_idx = self.random_state_.randint(n_min)
            minority_neighbors = idx_min[anchor_idx]
            majority_neighbors = idx_maj[anchor_idx]
            weights = np.zeros(k)
            for j in range(k):
                d_anchor = dist_maj[anchor_idx, j]
                neighbor_idx = minority_neighbors[j]
                d_neighbor = np.linalg.norm(X_min[neighbor_idx] - X_maj[majority_neighbors], axis=1)
                dbar_neighbor = np.mean(d_neighbor)
                density_j = maj_density[majority_neighbors[j]]
                density_term = density_j ** self.alpha
                weights[j] = d_anchor / (density_term * dbar_neighbor + self.epsilon)
            weights = np.maximum(weights, 0)
            if weights.sum() == 0:
                weights = np.ones(k) / k
            else:
                weights /= weights.sum()
            idx1 = self._roulette_select(weights)
            idx2 = self._roulette_select(weights)
            prox = X_min[minority_neighbors[idx1]]
            orion = X_min[minority_neighbors[idx2]]
            d_prox = np.linalg.norm(prox - X_min[anchor_idx])
            d_orion = np.linalg.norm(orion - X_min[anchor_idx])
            q_prox = self._q_gaussian(d_prox)
            q_orion = self._q_gaussian(d_orion)
            total = q_prox + q_orion
            coeff = 0.5 if total == 0 else q_prox / total
            x_syn = coeff * prox + (1 - coeff) * orion
            synthetic.append(x_syn)
        return np.array(synthetic)

    def _majority_density(self, X_maj, k):
        nn = NearestNeighbors(n_neighbors=k).fit(X_maj)
        distances, _ = nn.kneighbors(X_maj, n_neighbors=k)
        mean_dist = np.mean(distances, axis=1)
        raw_density = 1.0 / (mean_dist + self.epsilon)
        d_min, d_max = raw_density.min(), raw_density.max()
        if d_max - d_min == 0:
            return np.ones_like(raw_density)
        return (raw_density - d_min) / (d_max - d_min)

    def _q_gaussian(self, d):
        x = d / self.beta
        if self.q == 1:
            return np.exp(-0.5 * x ** 2)
        else:
            arg = 1.0 - (1.0 - self.q) * x ** 2
            if arg <= 0:
                return 0.0
            return arg ** (1.0 / (1.0 - self.q))

    def _roulette_select(self, weights):
        cumsum = np.cumsum(weights)
        r = self.random_state_.rand()
        return np.searchsorted(cumsum, r)


# =============================================================================
# Ablation variant without density (for q-sensitivity and ablation)
# =============================================================================

class QRO_NoDensity(QRO):
    def _generate_samples(self, X_min, X_maj, n_synthetic):
        n_min = len(X_min)
        k = min(self.k, n_min - 1, len(X_maj))
        nn_min = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        _, idx_min = nn_min.kneighbors(X_min, n_neighbors=k + 1)
        idx_min = idx_min[:, 1:]
        nn_maj = NearestNeighbors(n_neighbors=k).fit(X_maj)
        dist_maj, idx_maj = nn_maj.kneighbors(X_min, n_neighbors=k)
        synthetic = []
        for _ in range(n_synthetic):
            anchor_idx = self.random_state_.randint(n_min)
            minority_neighbors = idx_min[anchor_idx]
            majority_neighbors = idx_maj[anchor_idx]
            rho_reg = 1.0
            weights = np.zeros(k)
            for j in range(k):
                d_anchor = dist_maj[anchor_idx, j]
                neighbor_idx = minority_neighbors[j]
                d_neighbor = np.linalg.norm(X_min[neighbor_idx] - X_maj[majority_neighbors], axis=1)
                dbar_neighbor = np.mean(d_neighbor)
                weights[j] = d_anchor / (rho_reg * dbar_neighbor + self.epsilon)
            weights = np.maximum(weights, 0)
            if weights.sum() == 0:
                weights = np.ones(k) / k
            else:
                weights /= weights.sum()
            idx1 = self._roulette_select(weights)
            idx2 = self._roulette_select(weights)
            prox = X_min[minority_neighbors[idx1]]
            orion = X_min[minority_neighbors[idx2]]
            d_prox = np.linalg.norm(prox - X_min[anchor_idx])
            d_orion = np.linalg.norm(orion - X_min[anchor_idx])
            q_prox = self._q_gaussian(d_prox)
            q_orion = self._q_gaussian(d_orion)
            total = q_prox + q_orion
            coeff = 0.5 if total == 0 else q_prox / total
            x_syn = coeff * prox + (1 - coeff) * orion
            synthetic.append(x_syn)
        return np.array(synthetic)


# =============================================================================
# KMeans-SMOTE Implementation
# =============================================================================

class KMeansSMOTE(BaseEstimator, TransformerMixin):
    """
    KMeans-SMOTE: cluster minority class with k-means,
    then generate synthetic samples by interpolating within each cluster.
    """
    def __init__(self, kmeans_args=None, random_state=42):
        self.kmeans_args = kmeans_args if kmeans_args is not None else {'n_clusters': 5}
        self.random_state = random_state
        np.random.seed(random_state)

    def fit_resample(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        X_min = X[y == 1]
        X_maj = X[y == 0]
        n_min, n_maj = len(X_min), len(X_maj)
        n_synthetic = n_maj - n_min

        if n_synthetic <= 0:
            return X, y

        # If minority is too small, fallback to simple interpolation
        if len(X_min) < self.kmeans_args.get('n_clusters', 5) or len(X_min) < 2:
            synthetic = self._interpolate_random_pairs(X_min, n_synthetic)
            X_res = np.vstack([X, synthetic])
            y_res = np.hstack([y, np.ones(len(synthetic), dtype=y.dtype)])
            return X_res, y_res

        # Cluster the minority class
        kmeans = KMeans(**self.kmeans_args, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_min)
        unique_clusters = np.unique(cluster_labels)

        # Determine how many synthetic samples to generate per cluster (proportional to cluster size)
        cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
        total_cluster_size = sum(cluster_sizes)
        samples_per_cluster = [int(round(n_synthetic * sz / total_cluster_size)) for sz in cluster_sizes]
        diff = n_synthetic - sum(samples_per_cluster)
        samples_per_cluster[-1] += diff

        synthetic = []
        for c, n_gen in zip(unique_clusters, samples_per_cluster):
            if n_gen <= 0:
                continue
            cluster_points = X_min[cluster_labels == c]
            if len(cluster_points) < 2:
                # Single-point cluster: add small noise
                for _ in range(n_gen):
                    noise = np.random.normal(0, 0.01, size=cluster_points.shape[1])
                    synth_pt = cluster_points[0] + noise
                    synthetic.append(synth_pt)
                continue
            # Generate n_gen synthetic samples by interpolating between random pairs
            indices = np.random.choice(len(cluster_points), size=n_gen*2, replace=True)
            pairs = indices.reshape(n_gen, 2)
            for i, j in pairs:
                if i == j:
                    j = (j + 1) % len(cluster_points)
                alpha = np.random.rand()
                new_point = cluster_points[i] + alpha * (cluster_points[j] - cluster_points[i])
                synthetic.append(new_point)

        synthetic = np.array(synthetic) if synthetic else np.array([])
        X_res = np.vstack([X, synthetic])
        y_res = np.hstack([y, np.ones(len(synthetic), dtype=y.dtype)])
        return X_res, y_res

    def _interpolate_random_pairs(self, X_min, n_synthetic):
        """Fallback: generate synthetic by interpolating random pairs from whole minority set."""
        if len(X_min) < 2:
            return np.array([])
        synthetic = []
        for _ in range(n_synthetic):
            i, j = np.random.choice(len(X_min), size=2, replace=False)
            alpha = np.random.rand()
            new_point = X_min[i] + alpha * (X_min[j] - X_min[i])
            synthetic.append(new_point)
        return np.array(synthetic)


# =============================================================================
# Configuration & Constants – 40 datasets
# =============================================================================

DATASETS = [
    'abalone19', 'abalone9-18', 'ecoli3', 'dermatology-6', 'ecoli-0_vs_1',
    'ecoli1', 'ecoli4', 'flare-F', 'glass-0-1-2-3_vs_4-5-6', 'glass-0-1-6_vs_2',
    'glass-0-1-6_vs_5', 'glass0', 'glass1', 'glass2', 'glass4', 'glass5', 'glass6',
    'haberman', 'new-thyroid1', 'newthyroid2', 'pima', 'segment0', 'shuttle-c0-vs-c4',
    'vehicle0', 'vehicle1', 'vehicle2', 'vehicle3', 'vowel0', 'car-good',
    'winequality-red-4', 'wisconsin', 'yeast-0-5-6-7-9_vs_4', 'yeast-1-2-8-9_vs_7',
    'yeast-1-4-5-8_vs_7', 'yeast-1_vs_7', 'yeast-2_vs_4', 'yeast-2_vs_8',
    'yeast3', 'yeast5', 'yeast6'
]  # 40 datasets

TUNE_DATASETS = ['ecoli3', 'glass0', 'haberman', 'yeast3', 'abalone19']
SENSITIVITY_DATASETS = TUNE_DATASETS

# Set this to the folder containing the KEEL .dat files
DATA_DIR = "/path/to/keel_datasets"  # <-- CHANGE THIS
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Three classifiers
CLASSIFIERS = {
    'SVM': SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42),
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42)
}

N_FOLDS = 10
RANDOM_STATE = 42


# =============================================================================
# Data Loader 
# =============================================================================

def load_keel_dataset(dataset_name):
    filepath = os.path.join(DATA_DIR, dataset_name + '.dat')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    attrs = []
    data_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith('@attribute'):
            parts = line.split(None, 2)
            name = parts[1]
            dtype = parts[2]
            attrs.append((name, dtype))
        elif line.lower().startswith('@data'):
            data_start = i + 1
            break

    data_lines = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        data_lines.append(line.split(','))

    n_features = len(attrs) - 1
    feature_types = []
    for i in range(n_features):
        dtype_str = attrs[i][1]
        if dtype_str.startswith('{'):
            categories = [cat.strip() for cat in dtype_str.strip('{}').split(',')]
            feature_types.append(('nominal', categories))
        else:
            feature_types.append(('numeric', None))

    class_dtype = attrs[-1][1]
    if class_dtype.startswith('{'):
        class_categories = [cat.strip() for cat in class_dtype.strip('{}').split(',')]
    else:
        class_categories = None

    df = pd.DataFrame(data_lines)
    for i, (ftype, _) in enumerate(feature_types):
        if ftype == 'numeric':
            df[i] = pd.to_numeric(df[i], errors='coerce')
    class_col = n_features
    df[class_col] = df[class_col].astype(str).str.strip()

    X_parts = []
    for i, (ftype, categories) in enumerate(feature_types):
        if ftype == 'numeric':
            X_parts.append(df[i].values.reshape(-1, 1))
        else:
            dummies = pd.get_dummies(df[i], prefix=f"f{i}", drop_first=False)
            X_parts.append(dummies.values)
    X = np.hstack(X_parts).astype(np.float64)

    y_str = df[class_col].values
    unique, counts = np.unique(y_str, return_counts=True)
    if len(unique) != 2:
        minority_class = unique[np.argmin(counts)]
        y = (y_str == minority_class).astype(int)
    else:
        minority_class = unique[np.argmin(counts)]
        y = (y_str == minority_class).astype(int)

    if np.any(np.isnan(X)):
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    return X, y


# =============================================================================
# Toy Dataset Generation (for visualisation)
# =============================================================================

def generate_toy_dataset(seed=42):
    np.random.seed(seed)
    # Negative class clusters
    negative_x1 = np.random.normal(35, 3, 100)
    negative_y1 = np.random.normal(60, 5, 100)
    negative_x2 = np.random.normal(45, 3, 100)
    negative_y2 = np.random.normal(40, 5, 100)
    negative_x = np.concatenate([negative_x1, negative_x2])
    negative_y = np.concatenate([negative_y1, negative_y2])

    # Positive class clusters
    positive_x1 = np.random.normal(50, 2, 10)
    positive_y1 = np.random.normal(25, 2, 10)
    positive_x2 = np.random.normal(55, 2, 10)
    positive_y2 = np.random.normal(50, 2, 10)
    positive_x3 = np.random.normal(45, 2, 10)
    positive_y3 = np.random.normal(50, 2, 10)
    positive_x4 = np.random.normal(50, 2, 10)
    positive_y4 = np.random.normal(15, 2, 10)
    positive_x5 = np.random.normal(50, 2, 5)
    positive_y5 = np.random.normal(30, 2, 5)
    positive_x = np.concatenate([positive_x1, positive_x2, positive_x3, positive_x4, positive_x5])
    positive_y = np.concatenate([positive_y1, positive_y2, positive_y3, positive_y4, positive_y5])

    # Buffer zones around negative clusters
    buffer_distance = 5
    neg1_x_min = negative_x1.min() - buffer_distance
    neg1_x_max = negative_x1.max() + buffer_distance
    neg1_y_min = negative_y1.min() - buffer_distance
    neg1_y_max = negative_y1.max() + buffer_distance
    neg2_x_min = negative_x2.min() - buffer_distance
    neg2_x_max = negative_x2.max() + buffer_distance
    neg2_y_min = negative_y2.min() - buffer_distance
    neg2_y_max = negative_y2.max() + buffer_distance

    keep_positive_mask = ~(((positive_x > neg1_x_min) & (positive_x < neg1_x_max) &
                            (positive_y > neg1_y_min) & (positive_y < neg1_y_max)) |
                           ((positive_x > neg2_x_min) & (positive_x < neg2_x_max) &
                            (positive_y > neg2_y_min) & (positive_y < neg2_y_max)))
    final_positive_x = positive_x[keep_positive_mask]
    final_positive_y = positive_y[keep_positive_mask]

    additional_positive_x = np.array([40, 42, 44, 55, 55, 58, 42, 47, 45, 40])
    additional_positive_y = np.array([50, 52, 54, 35, 30, 37, 48, 55, 25, 20])
    final_positive_x = np.concatenate([final_positive_x, additional_positive_x])
    final_positive_y = np.concatenate([final_positive_y, additional_positive_y])

    overlap_x_min = max(negative_x1.min(), negative_x2.min())
    overlap_x_max = min(negative_x1.max(), negative_x2.max())
    overlap_y_min = max(negative_y1.min(), negative_y2.min())
    overlap_y_max = min(negative_y1.max(), negative_y2.max())
    keep_negative_mask = ~((negative_x > overlap_x_min) & (negative_x < overlap_x_max) &
                           (negative_y > overlap_y_min) & (negative_y < overlap_y_max))
    final_negative_x = negative_x[keep_negative_mask]
    final_negative_y = negative_y[keep_negative_mask]

    additional_negative_x = np.array([50, 45, 50])
    additional_negative_y = np.array([60, 60, 55])
    final_negative_x = np.concatenate([final_negative_x, additional_negative_x])
    final_negative_y = np.concatenate([final_negative_y, additional_negative_y])

    X = np.vstack((np.column_stack((final_negative_x, final_negative_y)),
                   np.column_stack((final_positive_x, final_positive_y))))
    y = np.hstack((np.zeros(len(final_negative_x)), np.ones(len(final_positive_x))))
    return X, y


# =============================================================================
# Experiment Runner (stores full results, runtime, margin, synthetic info, intrusion)
# =============================================================================

def run_experiment(oversamplers):
    """
    Returns:
        results_df: DataFrame with columns [dataset, classifier, sampler, fold, AUC, GMean, IR]
        runtime_df: DataFrame with columns [dataset, sampler, fold, resample_time]
        margin_df: DataFrame with columns [dataset, sampler, fold, margin] (SVM only)
        synthetic_info: list of dicts for geometric analysis (SVM only, ALL folds)
        intrusion_df: DataFrame with columns [dataset, sampler, fold, intrusion_score]
    """
    results = []
    runtime = []
    margin = []
    synthetic_info = []
    intrusion = []

    for ds_name in DATASETS:
        print(f"\n=== Processing {ds_name} ===")
        try:
            X, y = load_keel_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"Skipping {ds_name}: {e}")
            continue

        n_min = np.sum(y == 1)
        n_splits = min(N_FOLDS, n_min)
        if n_splits < 2:
            print(f"Skipping {ds_name}: too few minority samples ({n_min}) for cross‑validation.")
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        n_maj = np.sum(y == 0)
        ir = n_maj / n_min if n_min > 0 else np.inf

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_raw, X_test = X[train_idx], X[test_idx]
            y_train_raw, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_raw = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test)

            for sampler_name, sampler_constructor in oversamplers.items():
                start_time = time.time()
                if sampler_name == 'None':
                    X_train_res = X_train_raw
                    y_train_res = y_train_raw
                    synth = np.array([])
                    resample_time = 0.0
                else:
                    sampler = sampler_constructor()
                    try:
                        if sampler_name.startswith('QRO'):
                            X_train_res, y_train_res = sampler.fit_resample(X_train_raw, y_train_raw)
                            synth = getattr(sampler, 'synthetic_samples_', None)
                        else:
                            X_train_res, y_train_res = sampler.fit_resample(X_train_raw, y_train_raw)
                            n_orig = len(X_train_raw)
                            synth = X_train_res[n_orig:] if len(X_train_res) > n_orig else np.array([])
                    except Exception as e:
                        print(f"Error with {sampler_name} on fold {fold}: {e}")
                        continue
                    resample_time = time.time() - start_time

                # Loop over classifiers
                for clf_name, clf in CLASSIFIERS.items():
                    clf.fit(X_train_res, y_train_res)
                    y_proba = clf.predict_proba(X_test)[:, 1]
                    y_pred = clf.predict(X_test)

                    if len(np.unique(y_test)) == 2:
                        auc = roc_auc_score(y_test, y_proba)
                    else:
                        auc = np.nan

                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    gmean = np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0.0

                    results.append({
                        'dataset': ds_name,
                        'classifier': clf_name,
                        'sampler': sampler_name,
                        'fold': fold,
                        'AUC': auc,
                        'GMean': gmean,
                        'IR': ir
                    })

                    # Store margin for SVM
                    if clf_name == 'SVM':
                        decision_vals = clf.decision_function(X_test)
                        margin_estimate = np.mean(np.abs(decision_vals))
                        margin.append({
                            'dataset': ds_name,
                            'sampler': sampler_name,
                            'fold': fold,
                            'margin': margin_estimate
                        })

                        # Store synthetic info for geometric analysis (all folds)
                        if synth is not None and len(synth) > 0:
                            synthetic_info.append({
                                'dataset': ds_name,
                                'sampler': sampler_name,
                                'fold': fold,
                                'synthetic': synth,
                                'X_min_orig': X_train_raw[y_train_raw == 1]
                            })

                            # Compute intrusion score for this fold
                            X_maj_orig = X_train_raw[y_train_raw == 0]
                            dist_to_maj = cdist(synth, X_maj_orig).min(axis=1)
                            dist_to_min = cdist(synth, X_train_raw[y_train_raw == 1]).min(axis=1)
                            log_ratio = np.log((dist_to_min + 1e-12) / (dist_to_maj + 1e-12))
                            mean_log_ratio = np.mean(log_ratio)
                            intrusion.append({
                                'dataset': ds_name,
                                'sampler': sampler_name,
                                'fold': fold,
                                'intrusion_score': mean_log_ratio
                            })

                # Store runtime (once per sampler)
                runtime.append({
                    'dataset': ds_name,
                    'sampler': sampler_name,
                    'fold': fold,
                    'resample_time': resample_time
                })

    results_df = pd.DataFrame(results)
    runtime_df = pd.DataFrame(runtime)
    margin_df = pd.DataFrame(margin)
    intrusion_df = pd.DataFrame(intrusion)
    return results_df, runtime_df, margin_df, synthetic_info, intrusion_df


# =============================================================================
# Analysis Functions
# =============================================================================

def overall_auc_ranks(df, exclude_baseline=True):
    """
    Compute overall AUC ranks across classifiers:
    For each dataset, average AUC over classifiers, then rank samplers.
    """
    avg = df.groupby(['dataset', 'sampler'])['AUC'].mean().reset_index()
    if exclude_baseline:
        avg = avg[avg['sampler'] != 'None']
    avg['rank'] = avg.groupby('dataset')['AUC'].rank(ascending=False, method='min')
    overall_rank = avg.groupby('sampler')['rank'].mean().sort_values()
    return overall_rank


def plot_cd_diagram(avg_ranks, title, filename):
    """Create a publication‑ready CD diagram for overall AUC ranks."""
    methods = avg_ranks.index.tolist()
    ranks = avg_ranks.values
    order = np.argsort(ranks)
    methods_sorted = [methods[i] for i in order]
    ranks_sorted = ranks[order]

    df = pd.read_csv(os.path.join(RESULTS_DIR, 'full_results.csv'))
    avg = df.groupby(['dataset', 'sampler'])['AUC'].mean().reset_index()
    avg = avg[avg['sampler'] != 'None']
    rank_df = avg.pivot(index='dataset', columns='sampler', values='AUC')
    rank_df = rank_df.dropna()
    n_datasets = rank_df.shape[0]
    n_methods = rank_df.shape[1]

    if n_datasets == 0:
        print("Warning: No complete datasets. Skipping CD diagram.")
        return

    rank_matrix = rank_df.rank(axis=1, ascending=False, method='min').values
    friedman_stat, p_value = friedmanchisquare(*[rank_matrix[:, i] for i in range(n_methods)])
    print(f"Friedman p-value for overall AUC: {p_value}")

    nemenyi = sp.posthoc_nemenyi_friedman(rank_matrix)
    nemenyi.to_csv(os.path.join(RESULTS_DIR, 'nemenyi_auc_overall.csv'))

    q_alpha = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850, 7:2.948,
               8:3.031, 9:3.102, 10:3.164}.get(n_methods, 2.850)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    # Figure size: width 5in, height adaptive
    fig, ax = plt.subplots(figsize=(5, 2 + 0.25 * n_methods))
    x_min = min(ranks) - 0.5
    x_max = max(ranks) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, 1)
    ax.plot([x_min, x_max], [0, 0], color='black', linewidth=1.2)

    for i, (method, rank) in enumerate(zip(methods_sorted, ranks_sorted)):
        y_pos = 0.1 if i % 2 == 0 else -0.1
        ax.plot(rank, 0, 'o', color='steelblue', markersize=7)
        ax.text(rank, y_pos, method, ha='center', va='center', fontsize=8)

    cd_x = x_max - cd / 2
    ax.plot([cd_x - cd/2, cd_x + cd/2], [0.2, 0.2], color='crimson', linewidth=2)
    ax.text(cd_x, 0.28, f'CD = {cd:.3f}', ha='center', fontsize=8, color='crimson')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def variance_expansion_analysis(synthetic_info):
    """Compute variance ratio boxplot."""
    if not synthetic_info:
        print("No synthetic information available – skipping variance expansion analysis.")
        return

    records = []
    for info in synthetic_info:
        ds = info['dataset']
        sampler = info['sampler']
        X_min = info['X_min_orig']
        synth = info['synthetic']
        fold = info['fold']
        if len(X_min) < 2 or len(synth) < 2:
            continue
        cov_orig = np.cov(X_min.T)
        cov_synth = np.cov(synth.T)
        trace_orig = np.trace(cov_orig)
        if trace_orig < 1e-8:
            continue
        trace_synth = np.trace(cov_synth)
        ratio = trace_synth / trace_orig
        abs_diff = trace_synth - trace_orig
        records.append({'dataset': ds, 'sampler': sampler, 'fold': fold,
                        'variance_ratio': ratio, 'abs_diff': abs_diff})

    if not records:
        print("No valid variance ratios computed – skipping.")
        return

    df = pd.DataFrame(records)
    df_agg = df.groupby(['dataset', 'sampler'])['variance_ratio'].mean().reset_index()
    df_agg.to_csv(os.path.join(RESULTS_DIR, 'variance_ratio_agg.csv'), index=False)

    summary = df_agg.groupby('sampler')['variance_ratio'].agg(['mean', 'std']).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, 'variance_summary.csv'))
    print("\n=== Variance Expansion Ratio Summary (main) ===")
    print(summary)

    # Wilcoxon test QRO vs SMOTE
    qro = df_agg[df_agg['sampler'] == 'QRO'].set_index('dataset')['variance_ratio']
    smote = df_agg[df_agg['sampler'] == 'SMOTE'].set_index('dataset')['variance_ratio']
    common = qro.index.intersection(smote.index)
    if len(common) > 0:
        w_stat, w_p = wilcoxon(qro.loc[common], smote.loc[common])
        print(f"Wilcoxon test (QRO vs SMOTE variance ratio): p = {w_p:.4e}")
        with open(os.path.join(RESULTS_DIR, 'variance_wilcoxon.txt'), 'w') as f:
            f.write(f"Wilcoxon p-value: {w_p:.4e}")

        diff = qro.loc[common] - smote.loc[common]
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        print(f"Cohen's d (QRO - SMOTE): {d:.4f}")
        with open(os.path.join(RESULTS_DIR, 'variance_cohens_d.txt'), 'w') as f:
            f.write(f"Cohen's d: {d:.4f}")

        min_idx = df_agg.groupby('dataset')['variance_ratio'].idxmin()
        best_samplers = df_agg.loc[min_idx, 'sampler']
        qro_best = (best_samplers == 'QRO').mean() * 100
        print(f"QRO achieves lowest variance in {qro_best:.1f}% of datasets.")
        with open(os.path.join(RESULTS_DIR, 'variance_best_percent.txt'), 'w') as f:
            f.write(f"QRO lowest variance in {qro_best:.1f}% of datasets.")

    abs_df = df.groupby(['dataset', 'sampler'])['abs_diff'].mean().reset_index()
    abs_summary = abs_df.groupby('sampler')['abs_diff'].agg(['mean', 'std']).round(4)
    print("\n=== Internal: Absolute Deviation (trace_synth - trace_orig) ===")
    print(abs_summary)

    # Publication boxplot (width 6in)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_agg, x='sampler', y='variance_ratio', palette='Set2')
    plt.title('Variance Expansion Ratio', fontsize=11, fontweight='bold')
    plt.ylabel('Variance Ratio (synthetic / original)', fontsize=10)
    plt.xlabel('Oversampling Method', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'variance_boxplot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved variance_boxplot.pdf (main figure)")
    return df_agg


def intrusion_analysis(intrusion_df):
    """Analyze intrusion score, produce creative strip plot with mean markers."""
    if intrusion_df.empty:
        print("No intrusion data – skipping.")
        return

    # Aggregate per dataset-sampler (mean over folds)
    agg = intrusion_df.groupby(['dataset', 'sampler'])['intrusion_score'].mean().reset_index()
    agg.to_csv(os.path.join(RESULTS_DIR, 'intrusion_agg.csv'), index=False)

    summary = agg.groupby('sampler')['intrusion_score'].agg(['mean', 'std']).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, 'intrusion_summary.csv'))
    print("\n=== Intrusion Score Summary (main) ===")
    print("(higher score = more intrusion, worse)")
    print(summary)

    # Wilcoxon test QRO vs QRO-NoDensity
    qro = agg[agg['sampler'] == 'QRO'].set_index('dataset')['intrusion_score']
    nodens = agg[agg['sampler'] == 'QRO-NoDensity'].set_index('dataset')['intrusion_score']
    common = qro.index.intersection(nodens.index)
    if len(common) > 0:
        w_stat, w_p = wilcoxon(qro.loc[common], nodens.loc[common])
        print(f"Wilcoxon test (QRO vs QRO-NoDensity intrusion): p = {w_p:.4e}")
        with open(os.path.join(RESULTS_DIR, 'intrusion_wilcoxon.txt'), 'w') as f:
            f.write(f"Wilcoxon p-value: {w_p:.4e}")

    # ---- Main figure: strip plot with jitter and mean markers ----
    plt.figure(figsize=(8, 5))
    # Strip plot (points with jitter)
    sns.stripplot(data=agg, x='sampler', y='intrusion_score', jitter=True, alpha=0.6, color='gray', size=4)
    # Compute means
    means = agg.groupby('sampler')['intrusion_score'].mean().reset_index()
    # Plot mean markers and connecting line
    plt.plot(means['sampler'], means['intrusion_score'], marker='D', linestyle='-', color='red', linewidth=2, markersize=8, label='Mean')
    # Add horizontal line at zero for reference
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Oversampling Method', fontsize=11)
    plt.ylabel('Intrusion Score (log(d_min/d_maj))', fontsize=11)
    plt.title('Intrusion Scores Across Datasets', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'intrusion_strip_plot.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved intrusion_strip_plot.pdf (main figure)")

    # ---- Supplementary: paired line plot QRO vs QRO-NoDensity ----
    paired = agg[agg['sampler'].isin(['QRO', 'QRO-NoDensity'])].pivot(index='dataset', columns='sampler', values='intrusion_score')
    paired = paired.dropna()
    datasets = paired.index.tolist()
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, paired['QRO'].values, marker='o', linestyle='-', linewidth=1.5, color='steelblue', label='QRO')
    ax.plot(x, paired['QRO-NoDensity'].values, marker='s', linestyle='--', linewidth=1.5, color='crimson', label='QRO-NoDensity')
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Intrusion Score (log(d_min/d_maj))', fontsize=10)
    ax.set_title('Intrusion Score: Effect of Density Weighting', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'intrusion_paired_line.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved intrusion_paired_line.pdf ")


def sensitivity_study():
    """q-sensitivity study on 5 datasets using QRO_NoDensity."""
    print("\n=== q-Sensitivity Study ===")
    q_values = [1.1, 1.3, 1.5, 1.8, 2.0]
    k = 5
    beta = 0.1
    alpha = 0.5

    results = []
    for ds_name in SENSITIVITY_DATASETS:
        print(f"  Dataset: {ds_name}")
        try:
            X, y = load_keel_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"    Skipping {ds_name}: {e}")
            continue
        n_min = np.sum(y == 1)
        if n_min < 3:
            continue
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(skf.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for q in q_values:
            sampler = QRO_NoDensity(k=k, q=q, beta=beta, alpha=alpha, random_state=RANDOM_STATE)
            try:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                synth = sampler.synthetic_samples_
            except Exception as e:
                continue
            X_min_orig = X_train[y_train == 1]
            if len(X_min_orig) >= 2 and len(synth) >= 2:
                cov_orig = np.cov(X_min_orig.T)
                cov_synth = np.cov(synth.T)
                trace_orig = np.trace(cov_orig)
                if trace_orig > 1e-8:
                    var_ratio = np.trace(cov_synth) / trace_orig
                else:
                    var_ratio = np.nan
            else:
                var_ratio = np.nan

            X_maj_train = X_train[y_train == 0]
            if len(synth) > 0 and len(X_min_orig) > 0 and len(X_maj_train) > 0:
                dist_to_maj = cdist(synth, X_maj_train).min(axis=1)
                dist_to_min = cdist(synth, X_min_orig).min(axis=1)
                log_ratio = np.mean(np.log((dist_to_min + 1e-12) / (dist_to_maj + 1e-12)))
            else:
                log_ratio = np.nan

            results.append({
                'dataset': ds_name,
                'q': q,
                'variance_ratio': var_ratio,
                'intrusion_log_ratio': log_ratio
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_results.csv'), index=False)

    agg = df.groupby('q')[['variance_ratio', 'intrusion_log_ratio']].mean().reset_index()

    # Publication figure: two subplots, overall width 7in
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    # Left: variance ratio
    ax = axes[0]
    ax.plot(agg['q'], agg['variance_ratio'], marker='o', linestyle='-', linewidth=1.5, color='steelblue')
    ax.set_xlabel('q', fontsize=10)
    ax.set_ylabel('Variance Ratio', fontsize=10)
    ax.set_title('Variance Ratio vs q', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(labelsize=9)
    # Right: intrusion log-ratio
    ax = axes[1]
    ax.plot(agg['q'], agg['intrusion_log_ratio'], marker='s', linestyle='-', linewidth=1.5, color='crimson')
    ax.set_xlabel('q', fontsize=10)
    ax.set_ylabel('Intrusion Score', fontsize=10)
    ax.set_title('Intrusion Score vs q', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sensitivity_aggregated.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved sensitivity_aggregated.pdf (main figure)")


def runtime_analysis(runtime_df):
    """Summarise resampling times per sampler."""
    summary = runtime_df.groupby('sampler')['resample_time'].agg(['mean', 'std']).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, 'runtime_summary.csv'))
    print("\n=== Runtime Summary (seconds) – main table ===")
    print(summary)
    per_ds = runtime_df.groupby(['dataset', 'sampler'])['resample_time'].mean().reset_index()
    per_ds.to_csv(os.path.join(RESULTS_DIR, 'runtime_full.csv'), index=False)


def ablation_study(results_df, synthetic_info):
    """
    Minimal ablation on TUNE_DATASETS.
    Compares QRO (full), QRO-NoDensity, and SMOTE.
    Reports AUC (SVM) and variance ratio.
    """
    print("\n=== Minimal Ablation Study (5 tuning datasets) ===")
    df_svm = results_df[(results_df['classifier'] == 'SVM') & (results_df['dataset'].isin(TUNE_DATASETS))]
    auc_agg = df_svm.groupby(['dataset', 'sampler'])['AUC'].mean().reset_index()
    auc_table = auc_agg.pivot(index='dataset', columns='sampler', values='AUC').round(4)
    relevant = ['QRO', 'QRO-NoDensity', 'SMOTE']
    auc_table = auc_table[relevant] if all(s in auc_table.columns for s in relevant) else auc_table

    var_records = []
    for info in synthetic_info:
        if info['dataset'] not in TUNE_DATASETS:
            continue
        if info['sampler'] not in relevant:
            continue
        ds = info['dataset']
        sampler = info['sampler']
        X_min = info['X_min_orig']
        synth = info['synthetic']
        if len(X_min) < 2 or len(synth) < 2:
            continue
        cov_orig = np.cov(X_min.T)
        cov_synth = np.cov(synth.T)
        trace_orig = np.trace(cov_orig)
        if trace_orig < 1e-8:
            continue
        trace_synth = np.trace(cov_synth)
        ratio = trace_synth / trace_orig
        var_records.append({'dataset': ds, 'sampler': sampler, 'fold': info['fold'], 'variance_ratio': ratio})

    if var_records:
        var_df = pd.DataFrame(var_records)
        var_agg = var_df.groupby(['dataset', 'sampler'])['variance_ratio'].mean().reset_index()
        var_table = var_agg.pivot(index='dataset', columns='sampler', values='variance_ratio').round(4)
        var_table = var_table[relevant] if all(s in var_table.columns for s in relevant) else var_table
    else:
        var_table = pd.DataFrame()

    ablation_result = pd.concat([auc_table, var_table], axis=1, keys=['AUC', 'VarRatio'])
    ablation_result.to_csv(os.path.join(RESULTS_DIR, 'ablation_results.csv'))
    print("\nAblation Results (AUC | Variance Ratio) – main table:")
    print(ablation_result)
    return ablation_result


# =============================================================================
# Toy Dataset Visualisation 
# =============================================================================

def generate_toy_figure():
    """Create a compact 2x2 figure illustrating synthetic samples on toy data."""
    X_toy, y_toy = generate_toy_dataset(seed=42)

    # Select four representative methods
    samplers = {
        'QRO': QRO(k=9, q=1.5, beta=0.1, alpha=0.5, random_state=42),
        'SMOTE': SMOTE(k_neighbors=5, random_state=42),
        'KMeans-SMOTE': KMeansSMOTE(kmeans_args={'n_clusters': 5}, random_state=42),
        'Borderline-SMOTE': BorderlineSMOTE(kind='borderline-2', k_neighbors=5, random_state=42)
    }

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    axes = axes.flatten()

    x_min, x_max = X_toy[:, 0].min() - 5, X_toy[:, 0].max() + 5
    y_min, y_max = X_toy[:, 1].min() - 5, X_toy[:, 1].max() + 5

    for ax, (name, sampler) in zip(axes, samplers.items()):
        X_res, y_res = sampler.fit_resample(X_toy, y_toy)
        n_orig = len(X_toy)
        synthetic_mask = np.zeros(len(X_res), dtype=bool)
        if len(X_res) > n_orig:
            synthetic_mask[n_orig:] = True

        # Original majority
        orig_maj = (y_res == 0) & (~synthetic_mask)
        ax.scatter(X_res[orig_maj, 0], X_res[orig_maj, 1],
                   c='#1f77b4', marker='o', s=20, alpha=0.7, edgecolors='k', linewidth=0.5)
        # Original minority
        orig_min = (y_res == 1) & (~synthetic_mask)
        ax.scatter(X_res[orig_min, 0], X_res[orig_min, 1],
                   c='red', marker='o', s=20, alpha=0.7, edgecolors='k', linewidth=0.5)
        # Synthetic minority
        syn_min = (y_res == 1) & synthetic_mask
        ax.scatter(X_res[syn_min, 0], X_res[syn_min, 1],
                   c='green', marker='X', s=30, alpha=0.9, linewidth=1)

        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Feature 1', fontsize=8)
        ax.set_ylabel('Feature 2', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Single legend for the whole figure
    handles = [
        plt.Line2D([], [], color='#1f77b4', marker='o', linestyle='None', markersize=4, label='Original Majority'),
        plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=4, label='Original Minority'),
        plt.Line2D([], [], color='green', marker='X', linestyle='None', markersize=4, label='Synthetic Minority')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=8)

    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(RESULTS_DIR, 'toy_comparison.pdf'), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, 'toy_comparison.png'), dpi=600)
    plt.close()
    print("Saved toy_comparison.pdf and toy_comparison.png")


# =============================================================================
# Performance Summary (mean scores and ranks per classifier)
# =============================================================================

def performance_summary(results_df):

    # Exclude 'None' baseline
    df = results_df[results_df['sampler'] != 'None'].copy()

    # Average across folds per dataset-classifier-sampler
    per_ds = df.groupby(['dataset', 'classifier', 'sampler'])[['AUC', 'GMean']].mean().reset_index()

    # For each classifier and method, compute overall mean across datasets
    mean_scores = per_ds.groupby(['classifier', 'sampler'])[['AUC', 'GMean']].mean().reset_index()

    # For each classifier and method, compute average rank across datasets
    # First rank within each dataset-classifier
    per_ds['rank_AUC'] = per_ds.groupby(['dataset', 'classifier'])['AUC'].rank(ascending=False, method='min')
    per_ds['rank_GMean'] = per_ds.groupby(['dataset', 'classifier'])['GMean'].rank(ascending=False, method='min')
    # Average ranks
    avg_ranks = per_ds.groupby(['classifier', 'sampler'])[['rank_AUC', 'rank_GMean']].mean().reset_index()

    # Merge mean scores and average ranks
    merged = pd.merge(mean_scores, avg_ranks, on=['classifier', 'sampler'])

    # Pivot to have methods as rows and classifiers as columns
    table = merged.pivot(index='sampler', columns='classifier')
    # Flatten MultiIndex columns
    new_cols = []
    for metric, classifier in table.columns:
        if metric == 'AUC':
            new_cols.append(f'{classifier}_AUC_mean')
        elif metric == 'GMean':
            new_cols.append(f'{classifier}_GMean_mean')
        elif metric == 'rank_AUC':
            new_cols.append(f'{classifier}_AUC_rank')
        elif metric == 'rank_GMean':
            new_cols.append(f'{classifier}_GMean_rank')
    table.columns = new_cols
    table = table.reset_index()

    # Compute overall ranks (across classifiers) for AUC
    overall_auc_rank = overall_auc_ranks(results_df)

    # Compute overall ranks for G-Mean
    avg_gmean = results_df.groupby(['dataset', 'sampler'])['GMean'].mean().reset_index()
    avg_gmean = avg_gmean[avg_gmean['sampler'] != 'None']
    avg_gmean['rank'] = avg_gmean.groupby('dataset')['GMean'].rank(ascending=False, method='min')
    overall_gmean_rank = avg_gmean.groupby('sampler')['rank'].mean().sort_values()

    # Merge overall ranks into table
    overall_df = pd.DataFrame({
        'sampler': overall_auc_rank.index,
        'Overall_AUC_rank': overall_auc_rank.values,
        'Overall_GMean_rank': overall_gmean_rank.values
    })
    table = pd.merge(table, overall_df, on='sampler', how='left')

    # Reorder columns for clarity
    cols = ['sampler']
    for clf in CLASSIFIERS.keys():
        cols.extend([f'{clf}_AUC_mean', f'{clf}_AUC_rank', f'{clf}_GMean_mean', f'{clf}_GMean_rank'])
    cols.extend(['Overall_AUC_rank', 'Overall_GMean_rank'])
    # Ensure all columns exist
    cols = [c for c in cols if c in table.columns]
    table = table[cols]

    # Round numeric columns
    for col in table.columns:
        if col != 'sampler':
            table[col] = table[col].round(4)

    # Save
    table.to_csv(os.path.join(RESULTS_DIR, 'performance_summary.csv'), index=False)
    print("\n=== Performance Summary (mean scores and ranks) saved to performance_summary.csv ===")
    return table


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Tuned parameters (replace with your actual tuned values)
    best_k, best_q, best_alpha, best_beta = 5, 1.5, 0.5, 0.1

    OVERSAMPLERS = {
        'None': lambda: None,
        'SMOTE': lambda: SMOTE(k_neighbors=5, random_state=42),
        'Borderline-SMOTE': lambda: BorderlineSMOTE(
            kind="borderline-2", k_neighbors=5, m_neighbors=10, random_state=42
        ),
        'ADASYN': lambda: ADASYN(random_state=42),
        'SMOTE-ENN': lambda: SMOTEENN(random_state=42),
        'SMOTE-TL': lambda: SMOTETomek(random_state=42),
        'KMeans-SMOTE': lambda: KMeansSMOTE(
            kmeans_args={'n_clusters': 5},
            random_state=42
        ),
        'QRO': lambda: QRO(k=best_k, q=best_q, beta=best_beta, alpha=best_alpha, random_state=42),
        'QRO-NoDensity': lambda: QRO_NoDensity(k=best_k, q=best_q, beta=best_beta, alpha=best_alpha, random_state=42)
    }

    print("Starting full benchmark...")
    results_df, runtime_df, margin_df, synth_info, intrusion_df = run_experiment(OVERSAMPLERS)

    # Save master results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'full_results.csv'), index=False)

    # ===== Predictive Performance (Main) =====
    overall_ranks = overall_auc_ranks(results_df)
    overall_ranks.to_csv(os.path.join(RESULTS_DIR, 'overall_auc_ranks.csv'))
    print("\n=== Overall AUC Ranks (main) ===")
    print(overall_ranks)

    plot_cd_diagram(overall_ranks, 'Critical Difference – Overall AUC', 'cd_auc_overall.pdf')

    # ===== predictive Tables =====
    per_ds = results_df.groupby(['dataset', 'classifier', 'sampler'])[['AUC', 'GMean']].mean().reset_index()
    per_ds_auc = per_ds.pivot_table(index=['dataset', 'classifier'], columns='sampler', values='AUC')
    per_ds_gmean = per_ds.pivot_table(index=['dataset', 'classifier'], columns='sampler', values='GMean')
    per_ds_auc.to_csv(os.path.join(RESULTS_DIR, 'per_dataset_auc.csv'))
    per_ds_gmean.to_csv(os.path.join(RESULTS_DIR, 'per_dataset_gmean.csv'))
    print("Saved per-dataset tables")

    # ===== Performance Summary =====
    performance_summary(results_df)

    # ===== Variance Expansion (Main) =====
    variance_expansion_analysis(synth_info)

    # ===== Intrusion Analysis (Main) =====
    intrusion_analysis(intrusion_df)

    # ===== q-Sensitivity (Main) =====
    sensitivity_study()

    # ===== Runtime (Main table) =====
    runtime_analysis(runtime_df)

    # ===== Minimal Ablation (Main table) =====
    ablation_study(results_df, synth_info)

    # ===== Toy Dataset Visualisation =====
    generate_toy_figure()

    print(f"\nAll results saved in '{RESULTS_DIR}'")
   
