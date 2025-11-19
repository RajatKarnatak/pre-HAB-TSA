#!/usr/bin/env python3
"""
Peak → Window → Ward Clustering → Prediction Strength (PS) k selection → Plots

Changes in this version:
  • Uses the TIME column (col 0) to compute N0:
        N0 = N0_COS_MEAN - N0_COS_AMP * cos(2π*(time mod YEAR_LENGTH_DAYS)/YEAR_LENGTH_DAYS)
  • Ignores any extra columns after Z (i.e., the old seasonality column is not used).
  • Clustering features remain stats-only: mean, max, mean derivative for each of [N0, N, P_N, P_T, Z].
  • PS-based k selection with simple two-step rule:
        k_coarse = argmax PS(k) over candidates;
        k_fine   = candidate with PS(k) closest above its threshold (null+margin or fixed).
"""

# ============================== USER CONFIG (edit here) ==============================

# Data & windowing
WIN_SIZE                = 100          # pre-peak window length (samples)
PEAK_Q                  = 99.9         # percentile for P_T peak threshold

# Time & seasonal N0 from time column
YEAR_LENGTH_DAYS        = 365.0        # period for seasonal cosine
N0_COS_MEAN             = 1.0          # vertical mean for N0(t)
N0_COS_AMP              = 0.8          # cosine amplitude for N0(t)

# Column mapping in fort.11 (only the first 5 columns are used)
#   0: time, 1: N, 2: P_N, 3: P_T, 4: Z
RAW_COL_TIME            = 0
RAW_COLS_BIO            = [1, 2, 3, 4]  # N, P_N, P_T, Z

# Variables and feature selection
VAR_NAMES               = ["N0", "N", "P_N", "P_T", "Z"]
FEAT_COLS               = [1, 2, 3, 4, 5]  # 1-based indices into VAR_NAMES; keep 1 (N0)
# Features used for clustering are fixed to [mean, max, mean d/dt] for each selected variable.

# k range for metrics (will auto-trim to n_windows - 1)
K_MIN                   = 2
K_MAX                   = 15

# Prediction Strength settings
PS_REPS                 = 20           # number of random 2-fold splits per k
PS_BASE_SEED            = 42           # base RNG seed for PS
PS_JOBS                 = __import__("os").cpu_count() or 4

# Threshold mode: choose ONE of the two below
PS_USE_NULL_BASELINE    = True         # if True, per-k threshold = PS_null(k) + STAB_MARGIN; else fixed threshold
STAB_MARGIN             = 0.05         # used only when PS_USE_NULL_BASELINE=True
PS_FIXED_THRESHOLD      = 0.70         # used only when PS_USE_NULL_BASELINE=False

# Null baseline PS settings
NULL_PS_REPS            = 10           # reps for null PS (column-permuted features)

# Optional evaluations (fixed defaults)
GAP_B                   = 20           # 0 disables gap statistic
GAP_JOBS                = __import__("os").cpu_count() or 4
PERM_CHI                = 0            # permutations for CHI p-value; 0 disables
MAX_SAMPLES             = None         # subsample size for heavy metrics evaluation; None disables

# Plotting options
SAVE_CONSENSUS_HEATMAPS = False        # unused here; kept for compatibility
DENDRO_MAX_WINDOWS      = 2500         # skip dendrogram if more windows than this

# Matplotlib aesthetics (feel free to tweak)
MPL_DPI                 = 120
MPL_FONT_SIZES          = dict(
    xtick=18, ytick=18, axes=30, title=18, legend=16
)

# Colors per cluster label (for plots)
COLOR_CLUST = {
    -1: "#aaaaaa",
     0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd",
     5: "#8c564b", 6: "#e377c2", 7: "#7f7f7f", 8: "#bcbd22", 9: "#17becf",
}

# ============================ END USER CONFIG =======================================

# --- prevent BLAS oversubscription early ---
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import find_peaks
import scipy.cluster.hierarchy as sch
from itertools import combinations
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    pairwise_distances
)
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# Style
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": MPL_DPI})
plt.rcParams.update({
    "xtick.labelsize": MPL_FONT_SIZES["xtick"],
    "ytick.labelsize": MPL_FONT_SIZES["ytick"],
    "axes.labelsize" : MPL_FONT_SIZES["axes"],
    "axes.titlesize" : MPL_FONT_SIZES["title"],
    "legend.fontsize": MPL_FONT_SIZES["legend"],
})
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)


# ------------------- Helpers -------------------
def extract_pre_raw_windows(df, peaks, window_size=50):
    wins, valid = [], []
    for p in peaks:
        s = p - window_size
        if s >= 0:
            wins.append(df.iloc[s:p].values)
            valid.append(p)
    return np.array(wins), valid

def compute_inertia(X, labels):
    val = 0.0
    for c in np.unique(labels):
        mem = X[labels == c]
        if mem.size:
            cen = mem.mean(axis=0)
            val += ((mem - cen) ** 2).sum()
    return val

def clustering_stability(X, k, R=20, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    def one(rep_seed):
        r = np.random.default_rng(rep_seed)
        idx = r.choice(n, size=int(frac*n), replace=True)
        uniq = np.unique(idx)
        lab  = AgglomerativeClustering(k, linkage="ward").fit_predict(X[uniq])
        co = 0; tot = 0
        for i in range(len(lab)):
            same = (lab == lab[i]).sum() - 1
            co  += same
            tot += len(lab) - 1
        return co/tot if tot else 0
    seeds = rng.integers(0, 1e9, R)
    vals  = Parallel(os.cpu_count() or 4, prefer="threads")(delayed(one)(s) for s in seeds)
    return float(np.mean(vals))

# ---- Validation indices (heavy ones made robust) ----
def dunn_index(X, labels):
    D = pairwise_distances(X)
    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]
    inter = np.min([D[i][:, j].min()
                    for i in clusters for j in clusters if not np.array_equal(i,j)])
    intra= np.max([D[c][:, c].max() for c in clusters])
    return inter / intra if intra>0 else np.nan

def dunn2_index(X, labels):
    D = pairwise_distances(X)
    clusters = [np.where(labels==c)[0] for c in np.unique(labels)]
    inters = [D[np.ix_(ci, cj)].mean() for ci, cj in combinations(clusters, 2)]
    intra  = max(D[np.ix_(c, c)].max() for c in clusters)
    return np.mean(inters)/intra if intra>0 else np.nan

def cop_index(X, labels):
    D    = pairwise_distances(X)
    uniq = np.unique(labels)
    flags= []
    for c in uniq:
        idx_c = np.where(labels == c)[0]
        idx_o = np.where(labels != c)[0]
        if len(idx_c) < 2 or len(idx_o) == 0: continue
        d_in_max  = D[idx_c][:, idx_c].max(axis=1)
        d_out_min = D[idx_c][:, idx_o].min(axis=1)
        flags.extend((d_out_min / (d_in_max + 1e-12)) < 1.0)
    return np.mean(flags) if flags else np.nan

def s_dbw_index(X, labels):
    """Robust S_Dbw: returns NaN on degenerate cases instead of raising."""
    try:
        labs = np.unique(labels); k = len(labs)
        if k < 2: return np.nan
        groups = [X[labels == c] for c in labs]
        if any(g.shape[0] < 2 for g in groups): return np.nan
        centroids = np.vstack([g.mean(axis=0) for g in groups])
        scatter_r = np.array([
            np.mean(np.linalg.norm(g - g.mean(axis=0), axis=1)) + 1e-12
            for g in groups
        ])
        global_scatter = np.mean(np.linalg.norm(X - X.mean(axis=0), axis=1)) + 1e-12
        scat_term = float(np.mean(scatter_r) / global_scatter)
        def count_within_radius(points, center, radius):
            d = np.linalg.norm(points - center, axis=1)
            return int(np.sum(d <= radius))
        dens_bw_sum = 0.0
        for i in range(k):
            Xi, ci, ri = groups[i], centroids[i], scatter_r[i]
            for j in range(i + 1, k):
                Xj, cj, rj = groups[j], centroids[j], scatter_r[j]
                mid = 0.5 * (ci + cj); R = max(ri, rj)
                dens_mid = count_within_radius(np.vstack([Xi, Xj]), mid, R)
                dens_i   = count_within_radius(Xi, ci, ri)
                dens_j   = count_within_radius(Xj, cj, rj)
                denom = max(dens_i, dens_j, 1)
                dens_bw_sum += dens_mid / denom
        dens_term = 2.0 * dens_bw_sum / (k * (k - 1))
        return scat_term + dens_term
    except Exception:
        return np.nan

def c_index(X, labels):
    D = pairwise_distances(X)
    pairs_within = [pair for c in np.unique(labels)
                    for pair in combinations(np.where(labels==c)[0], 2)]
    S = sum(D[i,j] for i,j in pairs_within)
    dists = np.sort(D, axis=None)
    m = len(pairs_within)
    S_min = dists[:m].sum()
    S_max = dists[-m:].sum()
    return (S - S_min)/(S_max - S_min + 1e-12)

def xie_beni_index(X, labels):
    clusters = np.unique(labels)
    centroids = np.vstack([X[labels==c].mean(axis=0) for c in clusters])
    num = sum(((X[labels==c]-centroids[i])**2).sum()
              for i,c in enumerate(clusters))
    D = pairwise_distances(centroids)
    np.fill_diagonal(D, np.inf)
    den = (np.min(D)**2) * X.shape[0]
    return num/den if den>0 else np.nan

def pbm_index(X, labels):
    clusters = np.unique(labels)
    centroids = np.vstack([X[labels==c].mean(axis=0) for c in clusters])
    oc = X.mean(axis=0)
    E = sum(np.linalg.norm(ci-oc) for ci in centroids)
    W = sum(np.linalg.norm(X[labels==c]-centroids[i], axis=1).sum()
            for i,c in enumerate(clusters))
    return ((E/len(clusters))/(W/X.shape[0]))**2 if W>0 else np.nan

def hartigan_index(inertias, ks, n):
    H = np.full_like(inertias, np.nan, dtype=float)
    for i in range(len(ks)-1):
        H[i] = ((inertias[i]/inertias[i+1]) - 1)*(n - ks[i] - 1)
    return H

def mcclain_rao_index(X, labels):
    D = pairwise_distances(X)
    n = X.shape[0]
    within, between = [], []
    for i in range(n-1):
        for j in range(i+1, n):
            (within if labels[i]==labels[j] else between).append(D[i,j])
    return np.mean(within)/np.mean(between) if between else np.nan

def ball_hall_index(X, labels):
    clusters = np.unique(labels)
    total_var = sum(((X[labels==c]-X[labels==c].mean(axis=0))**2).sum()
                    for c in clusters)
    return total_var/X.shape[0]

def rmsstd_index(X, labels):
    clusters = np.unique(labels)
    variances = [np.var(X[labels==c], axis=0).sum() for c in clusters]
    return np.sqrt(np.mean(variances))

def permutation_chi_index(X, labels, n_perms=100, rng=None):
    if n_perms <= 0: return np.nan
    rng = np.random.default_rng() if rng is None else rng
    real_chi = calinski_harabasz_score(X, labels)
    perms = [calinski_harabasz_score(X, rng.permutation(labels)) for _ in range(n_perms)]
    return np.mean(np.array(perms) >= real_chi)

def _safe_metric(name, k, func, default=np.nan):
    try:
        return func()
    except Exception as e:
        warnings.warn(f"k={k}: {name} failed ({e}); using NaN.")
        return default

# ---------- Prediction Strength ----------
def fit_labels_ward(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)

def _centroids(Z, lab, k):
    C = []
    for c in range(k):
        mem = Z[lab == c]
        if mem.size == 0:
            return None
        C.append(mem.mean(axis=0))
    return np.vstack(C)

def _assign_to_centroids(points, C):
    D = pairwise_distances(points, C)
    return D.argmin(axis=1)

def _ps_one_side(pred_labels, true_labels):
    uniq = np.unique(pred_labels)
    vals = []
    for c in uniq:
        J = np.where(pred_labels == c)[0]
        m = len(J)
        if m < 2:
            continue
        tl = true_labels[J]
        counts = np.bincount(tl)
        num = np.sum(counts * (counts - 1) // 2)
        den = m * (m - 1) // 2
        vals.append(num / max(den, 1))
    return float(min(vals)) if vals else np.nan

def prediction_strength_once(X, k, seed=42):
    """One PS estimate via a random 2-fold split and nearest-centroid mapping across halves."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 4 or k < 2 or k >= n:
        return np.nan
    idx = rng.permutation(n)
    A = idx[: n // 2]; B = idx[n // 2 :]

    labA = fit_labels_ward(X[A], k)
    labB = fit_labels_ward(X[B], k)
    CA = _centroids(X[A], labA, k)
    CB = _centroids(X[B], labB, k)
    if CA is None or CB is None:
        return np.nan

    predA_on_B = _assign_to_centroids(X[B], CA)
    predB_on_A = _assign_to_centroids(X[A], CB)

    ps1 = _ps_one_side(predA_on_B, labB)
    ps2 = _ps_one_side(predB_on_A, labA)
    return np.nanmin([ps1, ps2])

def prediction_strength_avg(X, k, reps=10, base_seed=42, n_jobs=4):
    seeds = np.random.SeedSequence(base_seed).spawn(reps)
    vals = Parallel(n_jobs=min(n_jobs, reps), prefer="threads")(
        delayed(prediction_strength_once)(X, k, int(s.generate_state(1)[0]))
        for s in seeds
    )
    arr = np.array(vals, dtype=float)
    return np.nanmean(arr), np.nanstd(arr), arr

# ---- Plot routines (legends fixed) ----
def plot_transition_matrix(labels, out_file):
    uniq = sorted(np.unique(labels))
    k = len(uniq)
    if k < 2: return
    label_map = {c:i for i,c in enumerate(uniq)}
    lab_seq = [label_map[l] for l in labels]
    T = np.zeros((k, k))
    for i in range(1, len(lab_seq)):
        T[lab_seq[i-1], lab_seq[i]] += 1
    T = np.divide(T, T.sum(1, keepdims=True), out=np.zeros_like(T),
                  where=T.sum(1, keepdims=True)!=0)
    plt.figure(figsize=(4,3))
    sns.heatmap(T, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("To"); plt.ylabel("From")
    plt.tight_layout(); plt.savefig(out_file); plt.close()

def growth_rate_figures(pre_raw_windows, cluster_labels, out_base="adv_plots"):
    uniq = np.unique(cluster_labels)
    if len(uniq)==0: return
    time_axis = np.arange(-WIN_SIZE, 0)
    xe2, xk, xaT, xb, xc, xe, xr, xlam, xs, xmu = 0.3,0.13,0.2,0.05,0.4,0.03,0.0,0.6,0.04,0.02
    xphiN = 0.7; xphiT = 1-xphiN; xConvZ = 0.2; xPTth = xe2; xsharp=50; xnu=0.005; xa_Z = 0.2
    fig_mean, ax_mean = plt.subplots(4,1,figsize=(7,9),sharex=True)
    for c in uniq:
        idx = np.where(cluster_labels==c)[0]
        block = pre_raw_windows[idx][:,:, [1,2,3,4]]  # N, P_N, P_T, Z
        if block.size == 0: continue
        N, PN, PT, Z = [block[:,:,j] for j in range(4)]
        xfac = 0.5*(1+np.tanh(xsharp*(PT-xPTth)))
        upN = xaT*N*PT/((xe+N)*(xb+xc*PN+xc*PT))*(1-xfac)
        graZ= -xphiT*xlam*(PT**2)*Z/(xmu**2+xphiN*(PN**2)+xphiT*(PT**2))
        upZ = xa_Z*PT*Z/(xnu+Z)*xfac
        inst= upN - xr*PT + xConvZ*upZ + graZ - (xs+xk)*PT
        series = (upN, graZ, upZ, inst)
        for j,arr in enumerate(series):
            m = arr.mean(0); s = arr.std(0)
            ax_mean[j].plot(time_axis, m, lw=3, label=f"Cl {c}" if j==0 else None)
            ax_mean[j].fill_between(time_axis, m-s, m+s, alpha=0.25)
    for j,ax in enumerate(ax_mean):
        ax.set_ylabel([r"$up_N$", r"$gr_Z$", r"$up_Z$", r"$inst$"][j])
        ax.grid(True)
        ax.set_xticks([-50,-40,-30,-20,-10,0])
    if len(uniq): ax_mean[0].legend(ncol=max(1,len(uniq)))
    ax_mean[-1].set_xlabel("Time (pre-peak)")
    fig_mean.tight_layout()
    fp=f"{out_base}/mean_rates_k{int(np.max(uniq)) if len(uniq)>0 else 'NA'}.pdf"
    fig_mean.savefig(fp); plt.close()

def mean_trace_plot(pre_raw_windows, cluster_labels, win_size,
                    out="adv_plots/mean_trace_per_cluster.pdf"):
    clusters  = np.unique(cluster_labels)
    t_axis    = np.arange(-win_size, 0)
    var_names = [r"$N_0$", r"$N$", r"$P_N$", r"$P_T$", r"$Z$"]
    getters = [
        lambda block: block[:, :, 0],
        lambda block: block[:, :, 1],
        lambda block: block[:, :, 2],
        lambda block: block[:, :, 3],
        lambda block: block[:, :, 4],
    ]
    nrows = len(getters)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 12), sharex=True)
    for vi, (var, getter) in enumerate(zip(var_names, getters)):
        ax = axes[vi]
        for c in clusters:
            idx = np.where(cluster_labels == c)[0]
            if idx.size == 0: continue
            block  = pre_raw_windows[idx]
            series = getter(block)
            mean   = series.mean(axis=0); std = series.std(axis=0)
            if var == r"$P_T$":
                mean_plot  = np.clip(mean, 1e-12, np.inf)
                lower_plot = np.clip(mean - std, 1e-12, np.inf)
                upper_plot = np.clip(mean + std, 1e-12, np.inf)
                ax.plot(t_axis, mean_plot, lw=5, label=f"Cl {c}" if vi==0 else None)
                ax.fill_between(t_axis, lower_plot, upper_plot, alpha=0.5)
            else:
                ax.plot(t_axis, mean, lw=5, label=f"Cl {c}" if vi==0 else None)
                ax.fill_between(t_axis, mean - std, mean + std, alpha=0.5)
        ax.set_ylabel(var); ax.grid(True)
        #if var == r"$P_T$":
        #    ax.set_yscale("log"); ax.set_ylim(1e-10, 1e-1)
    if len(clusters): axes[0].legend(ncol=max(1, len(clusters)))
    axes[-1].set_xlabel("Time (pre-peak)")
    fig.tight_layout(); fig.savefig(out); plt.close()

def generate_all_plots(labels, pre_raw, pre_time, tag):
    os.makedirs("adv_plots", exist_ok=True)
    try: growth_rate_figures(pre_raw, labels, out_base="adv_plots")
    except Exception as e: warnings.warn(f"Growth-rate figure failed: {e}")
    try: mean_trace_plot(pre_raw, labels, WIN_SIZE,
                         out=f"adv_plots/mean_trace_per_cluster_k{tag}.pdf")
    except Exception as e: warnings.warn(f"Mean-trace plot failed: {e}")
    try: plot_transition_matrix(labels, f"transition_matrix_k{tag}.pdf")
    except Exception as e: warnings.warn(f"Transition matrix plot failed: {e}")

# ------------------- CLI -------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-heavy', action='store_true',
                   help='Skip O(N^2) indices (Dunn, C, MR, COP, S_Dbw, etc.)')
    return p.parse_args()

# ------------------- Main -------------------
def main():
    args = parse_args()

    # ---- feature set: fixed to FEAT_COLS; N0 pinned ----
    if 1 not in FEAT_COLS:
        raise ValueError("FEAT_COLS must include 1 (N0 is pinned).")
    bad = [c for c in FEAT_COLS if c < 1 or c > 5]
    if bad:
        raise ValueError(f"FEAT_COLS has invalid indices {bad}; valid are 1..5 for [N0,N,P_N,P_T,Z]")
    print("Clustering feature columns (from [N0,N,P_N,P_T,Z]; stats only):", FEAT_COLS)

    print("[1] Loading data …")
    # Read at least the first 5 columns; ignore any extras (e.g., old season column)
    data = pd.read_csv("fort.11", sep=r"\s+", header=None, engine="python")
    if data.shape[1] < 5:
        raise ValueError("fort.11 must have at least 5 columns: time, N, P_N, P_T, Z")

    # Build [N0, N, P_N, P_T, Z] with N0 derived from TIME (column 0)
    time_vec = data.iloc[:, RAW_COL_TIME].to_numpy(dtype=float)
    season_phase = np.mod(time_vec, YEAR_LENGTH_DAYS)  # wrap time onto [0, YEAR_LENGTH_DAYS)
    N0 = N0_COS_MEAN - N0_COS_AMP * np.cos(2.0*np.pi*season_phase / YEAR_LENGTH_DAYS)

    bio = data.iloc[:, RAW_COLS_BIO].to_numpy(dtype=float)   # N, P_N, P_T, Z
    values_all = np.column_stack([N0, bio])  # shape (T, 5)

    print("[2] Detecting peaks & extracting windows …")
    thr = np.percentile(data.iloc[:,3], PEAK_Q)  # P_T is col 3 (0-based)
    peaks, _ = find_peaks(data.iloc[:,3], height=thr)

    # Windows over 5 variables and time
    pre_raw,  valid_peaks = extract_pre_raw_windows(pd.DataFrame(values_all), peaks, WIN_SIZE)
    pre_time, _            = extract_pre_raw_windows(data[[RAW_COL_TIME]], peaks, WIN_SIZE)
    print(f"• found {len(peaks)} peaks; valid windows={len(valid_peaks)}")

    if pre_raw.shape[0] < 3:
        raise RuntimeError(f"Only {pre_raw.shape[0]} valid windows; need ≥3 for metrics.")

    # Quick histos (optional)
    def quick_hist(x, title, fname):
        if len(x) == 0: return
        plt.figure(figsize=(6,3)); sns.histplot(x, stat="density", bins=50)
        plt.title(title); plt.tight_layout(); plt.savefig(fname); plt.close()
    quick_hist(data.iloc[:,3][data.iloc[:,3] > thr], f"P_T > {PEAK_Q}th pct", "PT_high_pct.pdf")
    quick_hist(np.diff(np.sort(peaks)), "Inter-peak intervals", "inter_peak_intervals.pdf")

    # ---------- FEATURES: stats only (mean, max, mean rate) ----------
    times = np.squeeze(pre_time, axis=2)  # (n_win, WIN_SIZE)
    time_is_monotone = np.all(np.diff(times, axis=1) > 0, axis=1)

    stat_cols = []
    for c in FEAT_COLS:  # 1..5 map to channels 0..4
        series = pre_raw[:, :, c-1]                        # (n_win, WIN_SIZE)
        mean_v = np.nan_to_num(series.mean(axis=1))
        max_v  = np.nan_to_num(series.max(axis=1))
        dseries = np.empty_like(series, dtype=float)
        for i in range(series.shape[0]):
            dseries[i] = np.gradient(series[i], times[i]) if time_is_monotone[i] else np.gradient(series[i])
        mean_d = np.nan_to_num(dseries.mean(axis=1))
        stat_cols.extend([mean_v, max_v, mean_d])

    flat_feat = np.column_stack(stat_cols).astype(np.float32)
    scaler = StandardScaler()
    flat_feat = scaler.fit_transform(flat_feat).astype(np.float32)

    # Subsample for heavy metrics (without changing final clustering)
    if (MAX_SAMPLES is not None) and (flat_feat.shape[0] > MAX_SAMPLES):
        rng = np.random.default_rng(0)
        idx_sub = rng.choice(flat_feat.shape[0], size=MAX_SAMPLES, replace=False)
        flat = flat_feat[idx_sub]
    else:
        idx_sub = None
        flat = flat_feat

    # ---- Metrics for k range ----
    print("[3] computing metrics for k range …")
    K_vals = np.arange(K_MIN, min(K_MAX, flat.shape[0]-1)+1)

    def cluster_labels(X, k):
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)

    def _all_metrics(k):
        try:
            lab = cluster_labels(flat, k)
            if np.bincount(lab).min() == 0:
                raise ValueError("Empty cluster produced; reduce k.")
            sil   = silhouette_score(flat, lab)
            dbi   = davies_bouldin_score(flat, lab)
            chi   = calinski_harabasz_score(flat, lab)
            inert = compute_inertia(flat, lab)
            stab  = clustering_stability(flat, k)
            if args.skip_heavy:
                dunn = dunn2 = cop = sdbw = cind = xb = pbm = mr = bh = rmsstd = np.nan
            else:
                dunn  = _safe_metric("Dunn",      k, lambda: dunn_index(flat, lab))
                dunn2 = _safe_metric("Dunn2",     k, lambda: dunn2_index(flat, lab))
                cop   = _safe_metric("COP",       k, lambda: cop_index(flat, lab))
                sdbw  = _safe_metric("S_Dbw",     k, lambda: s_dbw_index(flat, lab))
                cind  = _safe_metric("C-index",   k, lambda: c_index(flat, lab))
                xb    = _safe_metric("Xie-Beni",  k, lambda: xie_beni_index(flat, lab))
                pbm   = _safe_metric("PBM",       k, lambda: pbm_index(flat, lab))
                mr    = _safe_metric("McClainRao",k, lambda: mcclain_rao_index(flat, lab))
                bh    = _safe_metric("BallHall",  k, lambda: ball_hall_index(flat, lab))
                rmsstd= _safe_metric("RMSSTD",    k, lambda: rmsstd_index(flat, lab))
            chi_perm = permutation_chi_index(flat, lab, n_perms=PERM_CHI) if PERM_CHI>0 else np.nan
            return (k, lab, sil, dbi, chi, inert, stab, dunn, dunn2, cop, sdbw,
                    cind, xb, pbm, mr, bh, rmsstd, chi_perm, None)
        except Exception as e:
            warnings.warn(f"k={k}: metrics failed ({e}); skipping this k.")
            return (k, None, *(np.nan,)*15, str(e))

    N_JOBS = os.cpu_count() or 4
    results = Parallel(N_JOBS, prefer="threads")(delayed(_all_metrics)(int(k)) for k in K_vals)

    # filter valid
    results_valid = [r for r in results if r[1] is not None]
    if not results_valid:
        raise RuntimeError("All k values failed or produced invalid clustering. Try lowering the k range.")

    (Ks, labels_list, SIL, DBI, CHI, INERT, STAB,
     DUNN, DUNN2, COP, SDBW, CIND, XB, PBM, MR, BH, RMSSTD, CHI_PERM, _ERRS) = zip(*results_valid)

    K_vals = np.array(Ks)
    SIL, DBI, CHI, INERT, STAB, DUNN, DUNN2, COP, SDBW = map(np.asarray,
        (SIL, DBI, CHI, INERT, STAB, DUNN, DUNN2, COP, SDBW))
    CIND, XB, PBM, MR, BH, RMSSTD, CHI_PERM = map(np.asarray,
        (CIND, XB, PBM, MR, BH, RMSSTD, CHI_PERM))

    # Gap statistic (fixed defaults)
    if len(K_vals) > 0 and GAP_B > 0:
        print("[3b] Gap statistic …")
        def gap_statistic(X, ks, *, B=20, random_state=0, n_jobs=os.cpu_count()):
            rng     = np.random.default_rng(random_state)
            mins    = X.min(0); maxs = X.max(0)
            n, d    = X.shape
            def _within_dispersion(data, labels, k):
                W = 0.0
                for c in range(k):
                    mem = data[labels == c]
                    if mem.size:
                        centroid = mem.mean(0)
                        W += ((mem - centroid) ** 2).sum()
                return W
            def _one_k(k, seed):
                lab_r = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
                W_real = _within_dispersion(X, lab_r, k)
                rng_k = np.random.default_rng(seed)
                log_Wb = []
                for _ in range(B):
                    Xb = rng_k.uniform(mins, maxs, size=(n, d))
                    lab_b = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xb)
                    Wb = _within_dispersion(Xb, lab_b, k)
                    log_Wb.append(np.log(Wb))
                log_Wb = np.array(log_Wb)
                gap   = log_Wb.mean() - np.log(W_real)
                sk    = np.sqrt(((log_Wb - log_Wb.mean()) ** 2).sum() / (B - 1)) * np.sqrt(1 + 1 / B)
                return k, gap, sk, W_real
            seeds = rng.integers(0, 1e9, size=len(K_vals))
            outs = Parallel(n_jobs=GAP_JOBS)(
                delayed(_one_k)(int(k), int(s)) for k, s in zip(K_vals, seeds)
            )
            ks_out, gaps, sks, _ = zip(*outs)
            order = np.argsort(ks_out)
            return np.array(gaps)[order], np.array(sks)[order]
        GAP, GAP_err = gap_statistic(flat, K_vals, B=GAP_B, random_state=42, n_jobs=GAP_JOBS)
    else:
        GAP = np.full_like(K_vals, np.nan, float); GAP_err = GAP.copy()

    # Combined plot (normalized overlay)
    print("[3c] Plotting validation curves …")
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()
    ax1.plot(K_vals, SIL, "-o", label="Silhouette (↑)", color="C0")
    ax2.plot(K_vals, DBI, "--s", label="DBI (↓)", color="C3")
    ax2.plot(K_vals, (CHI-CHI.min())/(CHI.max()-CHI.min()+1e-12),
             "-^", label="CHI norm (↑)", color="C2")
    ax1.plot(K_vals, (INERT-INERT.min())/(INERT.max()-INERT.min()+1e-12),
             ":d", label="Inertia norm (↓)", color="black")
    if np.isfinite(GAP).any():
        ax1.plot(K_vals, GAP, "-x", label="Gap (↑)", color="magenta")
    ax2.plot(K_vals, STAB, "-*", label="Bootstrap (↑)", color="orange")
    if not args.skip_heavy:
        ax2.plot(K_vals, DUNN, "-p", label="Dunn (↑)", color="C4")
        ax2.plot(K_vals, COP, "-h", label="COP (↓)", color="C5")
        ax2.plot(K_vals, SDBW, "-v", label="S_Dbw (↓)", color="C6")
    ax1.set_xlabel("k"); ax1.set_ylabel("Left-axis metrics"); ax2.set_ylabel("Right-axis metrics")
    ax1.set_xticks(K_vals)
    ax1.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=3)
    fig.tight_layout(); fig.savefig("cluster_validation_combined.pdf"); plt.close()

    # Metrics dicts for candidate extraction
    metrics_all = {
        'sil': SIL, 'dbi': DBI, 'chi': CHI, 'gap': GAP, 'inert': INERT,
        'dunn': DUNN, 'dunn2': DUNN2, 'cop': COP, 'sdbw': SDBW,
        'c': CIND, 'xb': XB, 'pbm': PBM, 'mr': MR, 'bh': BH, 'rmsstd': RMSSTD,
        'hartigan': hartigan_index(INERT, K_vals, flat.shape[0])
    }
    higher = {
        'sil':True, 'dbi':False, 'chi':True, 'gap':True, 'inert':False,
        'dunn':True, 'dunn2':True, 'cop':False, 'sdbw':False,
        'c':False, 'xb':False, 'pbm':True, 'mr':False, 'bh':False, 'rmsstd':False,
        'hartigan':True
    }

    # ---------- Candidate ks from optimal basic indices ----------
    BASIC_METRICS = list(metrics_all.keys())  # use all above
    cand_set = set()
    for name in BASIC_METRICS:
        arr = np.asarray(metrics_all[name], float)
        if not np.isfinite(arr).any():
            continue
        idx = np.nanargmax(arr) if higher[name] else np.nanargmin(arr)
        cand_set.add(int(K_vals[idx]))

    cand_ks = sorted(cand_set)
    if not cand_ks:
        best_k = int(K_vals[np.nanargmax(SIL)])
        cand_ks = sorted(set([best_k] + [best_k-1, best_k+1]) & set(K_vals.tolist()))
    print(f"Index-derived candidate ks: {cand_ks}")

    # ---------- Prediction Strength on candidate ks ----------
    print("[PS] Computing prediction strength on candidates …")
    ps_mean = {}
    ps_std  = {}
    for k in cand_ks:
        m, s, _ = prediction_strength_avg(flat_feat, k, reps=PS_REPS, base_seed=PS_BASE_SEED, n_jobs=PS_JOBS)
        ps_mean[k] = m
        ps_std[k]  = s

    if PS_USE_NULL_BASELINE:
        # Null baseline via column permutation
        print("[PS] Computing null (permuted) prediction strength …")
        rng = np.random.default_rng(PS_BASE_SEED + 777)
        Xs = flat_feat.copy()
        for j in range(Xs.shape[1]):
            rng.shuffle(Xs[:, j])
        ps_null = {}
        for k in cand_ks:
            m0, _, _ = prediction_strength_avg(Xs, k, reps=NULL_PS_REPS, base_seed=PS_BASE_SEED+1000, n_jobs=PS_JOBS)
            ps_null[k] = m0
    else:
        ps_null = {k: np.nan for k in cand_ks}

    # ---------- Simple two-step rule ----------
    def k_threshold(k):
        return (ps_null[k] + STAB_MARGIN) if PS_USE_NULL_BASELINE else PS_FIXED_THRESHOLD

    # 1) k_coarse: most stable overall (max PS across ALL candidates)
    k_coarse = max(cand_ks, key=lambda k: ps_mean[k])

    # 2) k_fine: closest above its threshold (if any), tie-break to smaller k
    eligible = [k for k in cand_ks if np.isfinite(ps_mean[k]) and (ps_mean[k] >= k_threshold(k))]
    def margin_above(k): return ps_mean[k] - k_threshold(k)

    if eligible:
        sorted_eligible = sorted(eligible, key=lambda k: (margin_above(k), k))
        k_fine = sorted_eligible[0]
        if k_fine == k_coarse and len(sorted_eligible) > 1:
            k_fine = sorted_eligible[1]
    else:
        k_fine = k_coarse

    # Pretty print diagnostics
    def _r(x): return None if (x is None or not np.isfinite(x)) else float(np.round(x, 3))
    print("PS mean per k:", {int(k): _r(ps_mean[k]) for k in cand_ks})
    if PS_USE_NULL_BASELINE:
        print("Null PS per k:", {int(k): _r(ps_null[k]) for k in cand_ks})
        print(f"Threshold rule: PS(k) ≥ PS_null(k) + {STAB_MARGIN}")
    else:
        print(f"Threshold rule: PS(k) ≥ {PS_FIXED_THRESHOLD}")
    print(f"Chosen k_coarse={k_coarse} (max PS overall), k_fine={k_fine} (closest above threshold)")

    # Final ks to run (coarse + fine)
    ks_to_run = []
    for k in [k_coarse, k_fine]:
        if k is not None and k not in ks_to_run:
            ks_to_run.append(k)
    print(f"Final ks to run: {ks_to_run}")

    # Final clustering, plots per k (using stats features)
    first = True
    flat_full = pre_raw.reshape(pre_raw.shape[0], -1).astype(np.float32)  # for dendrogram only
    for k in ks_to_run:
        print(f"[5] Final clustering for k={k} …")
        labels_k = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(flat_feat)
        print(f"• counts=", dict(zip(*np.unique(labels_k, return_counts=True))))
        if first:
            if flat_full.shape[0] <= DENDRO_MAX_WINDOWS:
                plt.figure(figsize=(10,4))
                sch.dendrogram(sch.linkage(flat_full, 'ward'), truncate_mode="lastp", p=30,
                               show_leaf_counts=True, leaf_rotation=90.0)
                plt.tight_layout(); plt.savefig("dendrogram_truncated.pdf"); plt.close()
            else:
                warnings.warn("Too many windows for dendrogram; skipping.")
            first = False
        generate_all_plots(labels_k, pre_raw, pre_time, tag=str(k))

    print("\nAll outputs complete ✔")

if __name__ == "__main__":
    main()
