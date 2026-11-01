#!/usr/bin/env python3
"""
nn_morph_metrics.py
===================
Selection-agnostic morphology metrics for BYOL nearest-neighbour pairs.

For every sample in the test split, find its 2 nearest neighbours in
BYOL embedding space (cosine distance, via L2-normalised Euclidean KNN)
and score morphological similarity with four families of metrics.
No labels are used anywhere.

Usage
-----
    python scripts/nn_morph_metrics.py \\
        --run_dir outputs/byol_runs/<name>/data_seed_<D>/training_seed_<T>/cross_val_<K> \\
        [--figures_dir figures]

Outputs
-------
    figures/fig1_metric_distributions.png
    figures/fig2_ssim_vs_embedding_distance.png
    figures/fig3_extremes.png
    figures/fig4_cdf.png

Limitations
-----------
- Metrics are computed on 89×89 px LoTSS cutouts (~5 arcsec/px).
  Beam convolution smears fine structure; sub-beam morphology is unreliable.
- SSIM is noise-sensitive: faint sources near the noise floor yield
  artificially low scores regardless of true morphological similarity.
- Hu moments are dominated by the brightest flux component; multi-lobe
  sources where lobe brightness is unequal will score poorly.
- The concentration index assumes a single central source; resolved
  doubles produce misleading values.
- PCA captures pixel-space variance, not semantic structure; it serves
  only as a mid-range baseline between random and BYOL.
"""

import argparse
import pathlib
import sys

import cv2                                          # OpenCV, for Hu moments
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import numpy as np
from scipy.stats import wilcoxon                   # non-parametric paired test
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from skimage.metrics import structural_similarity  # SSIM

# ── Colour / style constants ──────────────────────────────────────────────────
C_BYOL   = "#2196F3"   # blue
C_RANDOM = "#FF7043"   # red-orange
C_PCA    = "#43A047"   # green
COLORS   = [C_BYOL, C_RANDOM, C_PCA]
LABELS   = ["BYOL NN", "Random", "PCA NN"]

# Metrics where HIGHER = more similar (displayed as-is)
# Metrics where LOWER  = more similar (displayed with a note)
METRIC_KEYS = ["ssim", "mse_neg", "ncc", "hu_dist", "ell_diff", "conc_diff", "cos_dist"]
METRIC_LABELS = [
    "SSIM ↑",
    "−MSE ↑",           # stored as negative MSE so higher = better everywhere
    "NCC ↑",
    "Hu-dist ↓",
    "Ellipticity diff ↓",
    "Concentration diff ↓",
    "Cosine dist ↓",
]
N_METRICS = len(METRIC_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _znorm(img: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation of a 2-D image."""
    mu, sd = float(img.mean()), float(img.std())
    return (img - mu) / (sd + 1e-8)


def _dihedral(img: np.ndarray):
    """Generator: yield the 8 images in the dihedral group D4
    (4 rotations × original / horizontal-flip).
    Radio galaxy orientation is arbitrary so we take the best score
    over all 8 transforms when computing similarity."""
    for k in range(4):
        rot = np.rot90(img, k)
        yield rot
        yield np.fliplr(rot)


def _best_dihedral(fn, img1: np.ndarray, img2: np.ndarray) -> float:
    """Return max(fn(img1, t)) for all 8 dihedral transforms t of img2."""
    return max(fn(img1, t) for t in _dihedral(img2))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index between two normalised images.
    skimage.metrics.structural_similarity jointly measures luminance,
    contrast, and structural correlations; returns ∈ [-1, 1]."""
    dr = float(a.max() - a.min())
    return float(structural_similarity(a, b, data_range=max(dr, 1e-6)))


def _mse_neg(a: np.ndarray, b: np.ndarray) -> float:
    """Negative MSE (stored negative so that higher = more similar,
    consistent with SSIM / NCC conventions)."""
    return -float(np.mean((a - b) ** 2))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation: cosine similarity of flattened images.
    NCC ∈ [-1, 1]; 1 = perfectly correlated pixel patterns."""
    af, bf = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(af / na, bf / nb))


def _hu_moments_log(img: np.ndarray) -> np.ndarray:
    """7 Hu moments in log scale for a 2-D image.
    cv2.moments() computes raw, central, and normalised spatial moments.
    cv2.HuMoments() derives 7 values that are invariant under rotation,
    scaling, and translation.  Log-scaling (with sign preservation)
    linearises the large dynamic range."""
    img8 = (np.clip(img, 0, None) * 255).astype(np.uint8)
    m    = cv2.moments(img8)
    hu   = cv2.HuMoments(m).flatten()          # shape (7,)
    return np.sign(hu) * np.log10(np.abs(hu) + 1e-15)


def _hu_dist(img1: np.ndarray, img2: np.ndarray) -> float:
    """L2 distance between log-Hu moment vectors (lower = more similar)."""
    return float(np.linalg.norm(_hu_moments_log(img1) - _hu_moments_log(img2)))


def _ellipticity(img: np.ndarray) -> float:
    """Ellipticity ε = 1 − √(λ_min / λ_max) from the intensity-weighted
    covariance matrix (second-order central moments μ20, μ02, μ11).
    ε = 0 is circular; ε → 1 is highly elongated."""
    w = np.clip(img, 0, None)
    total = w.sum()
    if total < 1e-8:
        return 0.0
    ys, xs = np.indices(img.shape, dtype=float)
    xc = (xs * w).sum() / total
    yc = (ys * w).sum() / total
    dx, dy = xs - xc, ys - yc
    mu20 = (dx ** 2 * w).sum() / total
    mu02 = (dy ** 2 * w).sum() / total
    mu11 = (dx * dy * w).sum() / total
    tr   = mu20 + mu02
    det  = mu20 * mu02 - mu11 ** 2
    disc = max(0.0, (tr / 2) ** 2 - det)
    lam1 = tr / 2 + np.sqrt(disc)
    lam2 = max(0.0, tr / 2 - np.sqrt(disc))
    if lam1 < 1e-12:
        return 0.0
    return float(1.0 - np.sqrt(lam2 / lam1))


def _concentration(img: np.ndarray, r_frac: float = 0.3) -> float:
    """Fraction of total flux within the inner r_frac of the image radius.
    Higher = more centrally concentrated (compact) source.
    Uses the geometric image centre as the origin."""
    w = np.clip(img, 0, None)
    total = w.sum()
    if total < 1e-8:
        return 0.0
    cy, cx = (np.array(img.shape) - 1) / 2.0
    ys, xs = np.indices(img.shape, dtype=float)
    r      = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_max  = np.sqrt(cx ** 2 + cy ** 2)
    return float(w[r <= r_frac * r_max].sum() / total)


def compute_pair_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Compute all 6 image-based metrics for one (query, neighbour) pair.
    Cosine distance is filled in by the caller from the embedding vectors."""
    img1n, img2n = _znorm(img1.astype(np.float32)), _znorm(img2.astype(np.float32))
    return {
        "ssim":     _best_dihedral(_ssim,     img1n, img2n),
        "mse_neg":  _best_dihedral(_mse_neg,  img1n, img2n),
        "ncc":      _best_dihedral(_ncc,      img1n, img2n),
        "hu_dist":  _hu_dist(img1.astype(np.float32), img2.astype(np.float32)),
        "ell_diff": abs(_ellipticity(img1.astype(np.float32)) -
                        _ellipticity(img2.astype(np.float32))),
        "conc_diff": abs(_concentration(img1.astype(np.float32)) -
                         _concentration(img2.astype(np.float32))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(run_dir: pathlib.Path):
    """Load images and projections for the given cross_val run directory.

    Directory layout assumed:
        run_dir = .../training_seed_T/cross_val_K/
        test_projections: run_dir/data/byol/test_projections.npy
        test_idx:         outputs/data_splits/<D>/cross_val_K/test_idx.npy
        images:           <repo_root>/data/preprocessed/lotss/images_filtered.npy
    """
    proj_path = run_dir / "data" / "byol" / "test_projections.npy"
    if not proj_path.exists():
        sys.exit(f"ERROR: projection file not found: {proj_path}")

    # Derive outputs root: cross_val_K -> training_seed_T -> data_seed_D ->
    #                       run_name -> byol_runs -> outputs
    outputs_root = run_dir.parent.parent.parent.parent.parent
    data_seed    = run_dir.parent.parent.name.replace("data_seed_", "")
    cv_name      = run_dir.name                            # e.g. cross_val_0

    idx_path = outputs_root / "data_splits" / data_seed / cv_name / "test_idx.npy"
    if not idx_path.exists():
        sys.exit(f"ERROR: test_idx not found: {idx_path}")

    imgs_path = outputs_root.parent / "data" / "preprocessed" / "lotss" / "images_filtered.npy"
    if not imgs_path.exists():
        sys.exit(f"ERROR: images not found: {imgs_path}")

    print(f"Loading images from {imgs_path} ...")
    all_imgs = np.load(imgs_path, mmap_mode="r").astype(np.float32)
    if all_imgs.max() > 1.5:
        all_imgs = all_imgs / 255.0

    test_idx  = np.load(idx_path)
    test_imgs = all_imgs[test_idx]                         # (N_test, H, W)

    print(f"Loading projections from {proj_path} ...")
    proj = np.load(proj_path).astype(np.float32)
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    proj_norm = proj / np.maximum(norms, 1e-10)            # L2-normalised

    print(f"  Test set: {len(test_idx)} samples, images {test_imgs.shape[1:]}")
    return test_imgs, proj_norm


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-neighbour retrieval
# ─────────────────────────────────────────────────────────────────────────────

def byol_nn(proj_norm: np.ndarray, k: int = 2):
    """Return indices and cosine distances for the k nearest neighbours of
    every sample in proj_norm (excluding itself).
    Using L2 on unit-sphere vectors is equivalent to cosine distance."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(proj_norm)
    dists, idxs = nn.kneighbors(proj_norm)
    # Row 0 is self (distance ≈ 0); drop it
    cos_dists = dists[:, 1:] / 2.0     # Euclidean²/2 = 1 − cosine_sim for unit vecs
    return idxs[:, 1:], cos_dists      # (N, k), (N, k)


def random_nn(n: int, k: int = 2, seed: int = 42) -> np.ndarray:
    """For each of n samples draw k random other indices (fixed seed)."""
    rng  = np.random.default_rng(seed)
    idxs = np.zeros((n, k), dtype=int)
    for i in range(n):
        pool = np.delete(np.arange(n), i)
        idxs[i] = rng.choice(pool, size=k, replace=False)
    return idxs


def pca_nn(test_imgs: np.ndarray, k: int = 2, n_components: int = 64):
    """Fit PCA on flattened test images and return k-NN indices in PCA space.
    PCA finds the directions of maximum pixel-variance; it is a
    mid-range baseline between random and BYOL."""
    flat  = test_imgs.reshape(len(test_imgs), -1)
    print(f"Fitting PCA ({n_components} components) on {flat.shape} ...")
    pca   = PCA(n_components=min(n_components, flat.shape[1]), random_state=0)
    codes = pca.fit_transform(flat)
    codes /= (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-10)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(codes)
    _, idxs = nn.kneighbors(codes)
    return idxs[:, 1:]


# ─────────────────────────────────────────────────────────────────────────────
# Batch metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(test_imgs, nn_idxs, cos_dists=None, desc=""):
    """Compute per-query mean of metrics over k neighbours.

    Returns dict {metric_key: 1-D array of length N_queries}.
    cos_dists may be None for baselines where it is not meaningful.
    """
    N, k = nn_idxs.shape
    accum = {key: np.zeros(N, dtype=np.float32) for key in METRIC_KEYS}

    for i in range(N):
        if i % 200 == 0:
            print(f"  {desc} {i}/{N} ...", end="\r", flush=True)
        for j in range(k):
            m = compute_pair_metrics(test_imgs[i], test_imgs[nn_idxs[i, j]])
            for key in m:
                accum[key][i] += m[key]
            if cos_dists is not None:
                accum["cos_dist"][i] += float(cos_dists[i, j])
            else:
                accum["cos_dist"][i] = np.nan
        for key in METRIC_KEYS:
            accum[key][i] /= k

    print(f"  {desc} done.              ")
    return accum


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_effect(a, b):
    """Wilcoxon signed-rank test + rank-biserial correlation (effect size r).
    Returns (p_value, effect_size_r).  Handles ties/zeros gracefully."""
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 10:
        return np.nan, np.nan
    try:
        res = wilcoxon(a, b, zero_method="wilcox", correction=False)
        p   = res.pvalue
        # rank-biserial r = 1 - 2*W / (n*(n+1)/2)  [effect size for Wilcoxon]
        n   = len(a)
        r   = 1.0 - 2.0 * res.statistic / (n * (n + 1) / 2.0)
        return float(p), float(r)
    except Exception:
        return np.nan, np.nan


def pval_str(p):
    if np.isnan(p):
        return "n.a."
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return f"p={p:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig1_distributions(metrics_byol, metrics_rand, metrics_pca, out_path):
    """One panel per metric: overlaid violin plots for BYOL / Random / PCA.
    Medians are marked. Each panel is annotated with the Wilcoxon p-value
    and rank-biserial effect size (BYOL vs Random)."""
    n_cols = 4
    n_rows = (N_METRICS + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2))
    axes_flat = axes.flatten()

    for ai, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
        ax = axes_flat[ai]
        all_data = [
            metrics_byol[key][np.isfinite(metrics_byol[key])],
            metrics_rand[key][np.isfinite(metrics_rand[key])],
            metrics_pca [key][np.isfinite(metrics_pca [key])],
        ]
        positions  = [p for p, d in zip([1, 2, 3], all_data) if len(d) > 1]
        data_sets  = [d for d in all_data if len(d) > 1]
        plot_cols  = [c for c, d in zip(COLORS, all_data) if len(d) > 1]
        if not data_sets:
            ax.set_title(label + " (no data)", fontsize=10)
            continue
        parts = ax.violinplot(data_sets, positions=positions,
                              showmedians=True, showextrema=False)
        for pc, col in zip(parts["bodies"], plot_cols):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Annotate with Wilcoxon (BYOL vs Random) and effect size
        p, r = wilcoxon_effect(metrics_byol[key], metrics_rand[key])
        ax.text(0.97, 0.97,
                f"BYOL vs Rand\n{pval_str(p)}, r={r:+.2f}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.7, pad=2, linewidth=0))

    # Hide unused panels
    for ax in axes_flat[N_METRICS:]:
        ax.set_visible(False)

    fig.suptitle("Morphology metric distributions for 2-NN pairs", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig2_ssim_vs_cosdist(metrics_byol, out_path):
    """2-D density (hexbin) of mean SSIM vs mean cosine distance per query."""
    x = metrics_byol["cos_dist"]
    y = metrics_byol["ssim"]
    mask = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    hb = ax.hexbin(x[mask], y[mask], gridsize=40, cmap="Blues",
                   mincnt=1, linewidths=0.2)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count", fontsize=10)

    # Pearson r for annotation
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    ax.text(0.97, 0.97, f"r = {r:+.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)

    ax.set_xlabel("Mean cosine distance (embedding)", fontsize=11)
    ax.set_ylabel("Mean SSIM (best dihedral)", fontsize=11)
    ax.set_title("SSIM vs embedding distance — BYOL 2-NN", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig3_extremes(test_imgs, nn_idxs_byol, metrics_byol, out_path, n_show=10):
    """Grid showing the n_show best and n_show worst queries by mean SSIM,
    each alongside their 2 nearest neighbours.
    Layout: two halves (best / worst), each n_show columns × 3 rows."""
    ssim_vals = metrics_byol["ssim"]
    valid     = np.where(np.isfinite(ssim_vals))[0]
    ranked    = valid[np.argsort(ssim_vals[valid])]
    worst_idxs = ranked[:n_show]                   # lowest SSIM
    best_idxs  = ranked[-n_show:][::-1]            # highest SSIM

    k = nn_idxs_byol.shape[1]                      # should be 2

    fig = plt.figure(figsize=(n_show * 1.4, 3 * 1.4 * 2 + 0.8))
    outer = mgs.GridSpec(2, 1, figure=fig, hspace=0.18)

    for half, (group_idxs, title) in enumerate([
        (best_idxs,  f"Best {n_show} by SSIM"),
        (worst_idxs, f"Worst {n_show} by SSIM"),
    ]):
        inner = mgs.GridSpecFromSubplotSpec(
            3, n_show, subplot_spec=outer[half],
            hspace=0.04, wspace=0.04,
        )
        for col, qi in enumerate(group_idxs):
            # Row 0 = query
            ax = fig.add_subplot(inner[0, col])
            ax.imshow(test_imgs[qi], cmap="viridis", origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("Query", fontsize=7)
            if half == 0 and col == n_show // 2:
                ax.set_title(title, fontsize=9, pad=4)
            elif half == 1 and col == n_show // 2:
                ax.set_title(title, fontsize=9, pad=4)
            ssim_v = float(ssim_vals[qi])
            ax.text(0.02, 0.98, f"{ssim_v:.2f}",
                    transform=ax.transAxes, fontsize=6,
                    color="white", va="top", ha="left",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1, linewidth=0))

            # Rows 1+ = neighbours
            for row in range(1, min(k + 1, 3)):
                ni  = int(nn_idxs_byol[qi, row - 1])
                ax2 = fig.add_subplot(inner[row, col])
                ax2.imshow(test_imgs[ni], cmap="viridis", origin="lower")
                ax2.set_xticks([]); ax2.set_yticks([])
                if col == 0:
                    ax2.set_ylabel(f"NN{row}", fontsize=7)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig4_cdf(metrics_byol, metrics_rand, metrics_pca, out_path):
    """Empirical CDFs of SSIM for the three retrieval conditions."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for mdict, col, lbl in zip(
        [metrics_byol, metrics_rand, metrics_pca], COLORS, LABELS
    ):
        vals = np.sort(mdict["ssim"][np.isfinite(mdict["ssim"])])
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, color=col, linewidth=1.8, label=lbl)
        # Mark median
        med = float(np.median(vals))
        med_cdf = float(np.interp(med, vals, cdf))
        ax.scatter([med], [med_cdf], color=col, s=50, zorder=5,
                   marker="D", edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Mean SSIM (best dihedral)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Empirical CDF of SSIM — BYOL vs baselines", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics_byol, metrics_rand, metrics_pca):
    print("\n── Metric medians (BYOL | Random | PCA) ──────────────────────────")
    for key, label in zip(METRIC_KEYS, METRIC_LABELS):
        mb = np.nanmedian(metrics_byol[key])
        mr = np.nanmedian(metrics_rand[key])
        mp = np.nanmedian(metrics_pca[key])
        print(f"  {label:<30s}  {mb:+7.4f}  {mr:+7.4f}  {mp:+7.4f}")
    print()
    print("── Wilcoxon test BYOL vs Random ──────────────────────────────────")
    for key, label in zip(METRIC_KEYS, METRIC_LABELS):
        p, r = wilcoxon_effect(metrics_byol[key], metrics_rand[key])
        print(f"  {label:<30s}  {pval_str(p):<10s}  r={r:+.3f}")
    print()
    print("Limitations: see module docstring.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, type=pathlib.Path,
                        help="Path to a cross_val_N directory "
                             "(must contain data/byol/test_projections.npy)")
    parser.add_argument("--figures_dir", default="figures", type=pathlib.Path,
                        help="Output directory for PNG figures (default: figures/)")
    parser.add_argument("--k", default=2, type=int,
                        help="Number of nearest neighbours (default: 2)")
    parser.add_argument("--pca_components", default=64, type=int,
                        help="PCA dimensionality for baseline (default: 64)")
    args = parser.parse_args()

    figures_dir = pathlib.Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    test_imgs, proj_norm = load_data(args.run_dir)

    # ── Retrieve neighbours ───────────────────────────────────────────────────
    print("Computing BYOL nearest neighbours ...")
    nn_idxs_byol, cos_dists = byol_nn(proj_norm, k=args.k)

    print("Sampling random neighbours ...")
    nn_idxs_rand = random_nn(len(test_imgs), k=args.k)

    print("Computing PCA nearest neighbours ...")
    nn_idxs_pca = pca_nn(test_imgs, k=args.k, n_components=args.pca_components)

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("Computing metrics — BYOL NN ...")
    m_byol = compute_all_metrics(test_imgs, nn_idxs_byol, cos_dists,  desc="BYOL")
    print("Computing metrics — Random NN ...")
    m_rand = compute_all_metrics(test_imgs, nn_idxs_rand, None,        desc="Rand")
    print("Computing metrics — PCA NN ...")
    m_pca  = compute_all_metrics(test_imgs, nn_idxs_pca,  None,        desc="PCA ")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig1_distributions(m_byol, m_rand, m_pca,
                       figures_dir / "fig1_metric_distributions.png")
    fig2_ssim_vs_cosdist(m_byol,
                         figures_dir / "fig2_ssim_vs_embedding_distance.png")
    fig3_extremes(test_imgs, nn_idxs_byol, m_byol,
                  figures_dir / "fig3_extremes.png")
    fig4_cdf(m_byol, m_rand, m_pca,
             figures_dir / "fig4_cdf.png")

    print_summary(m_byol, m_rand, m_pca)
    print(f"\nAll figures written to: {figures_dir.resolve()}/")
    print("\nRun command:")
    print(f"  python scripts/nn_morph_metrics.py --run_dir {args.run_dir}")


if __name__ == "__main__":
    main()
