"""
run_protege_sweep.py — Protege GP active-learning sweep over BYOL run directories.

For each run directory matching --run-glob under --outputs-root:
  1. Load labelled + unlabelled train projections, indices, and fixed test split.
  2. Build source names, labels (tier-scored human_label 1–4), and PCA features.
  3. Seed anomaly_scores from the labelled training set, run ScoreConverter.
  4. Run GP active learning (Protege) querying the unlabelled train pool.
  5. Compute recall-curve AUC over the unqueried eval set.
  6. Save recall_curve, protege_scores.parquet, and protege_summary.json.

After all runs, print a summary table sorted by AUC descending.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import multiprocessing as mp

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from astronomaly.anomaly_detection import human_loop_learning
from astronomaly.base.base_dataset import Dataset as _AstroDataset
from astronomaly.feature_extraction import shape_features as _sf

# ---------------------------------------------------------------------------
# Hardcoded paths and constants
# ---------------------------------------------------------------------------
CSV_PATH    = Path("/users/mbredber/p3_SUPLAT/data/metadata/lotss_classifications_horton_et_al_2025_filtered.csv")
LABELS_PATH  = Path("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss/labels_filtered.npy")
IMAGES_PATH  = Path("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss/images_filtered.npy")

LABEL_COLS = [
    "fri", "frii", "hybrid", "spiral", "relaxed",
    "cshaped", "sshaped", "misaligned", "wings", "xshaped",
    "straight", "multihotspots", "continuous", "banding", "onesided",
    "restarted", "cluster", "merger", "diffuse", "unknown",
]

SCORE_4 = ["xshaped", "unknown", "cluster", "merger"]
SCORE_3 = ["diffuse", "sshaped", "spiral"]
SCORE_2 = ["restarted", "onesided", "banding", "cshaped", "wings", "misaligned", "multihotspots", "relaxed"]
SCORE_1 = ["fri", "frii", "hybrid", "straight", "continuous"]
TIERS   = [(4, SCORE_4), (3, SCORE_3), (2, SCORE_2), (1, SCORE_1)]

PROTEGE_INITIAL_STEPS = 10


# ---------------------------------------------------------------------------
# ASTRONOMALY wrapper for numpy image arrays
# ---------------------------------------------------------------------------
class _NumpyImageDataset(_AstroDataset):
    def __init__(self, images, output_dir, **kwargs):
        super().__init__(output_dir=output_dir, **kwargs)
        self.images   = images
        self.index    = [str(i) for i in range(len(images))]
        self.metadata = pd.DataFrame(index=self.index)

    def get_sample(self, idx):
        return self.images[int(idx)]

    def get_display_data(self, idx):
        return {}


# ---------------------------------------------------------------------------
# Active-learning function (verbatim from notebook)
# ---------------------------------------------------------------------------
def run_GP_active_learning(features, labels, input_anomaly_scores, output_dir,
                           steps=10, initial_steps=None, N_labels=100, epsilon=0.5,
                           max_queries=None, record_timing=False):
    """Direct sklearn GP implementation.

    Speedups vs. the astronomaly wrapper:
    - Kernel hyperparameters are optimised only on the first fit; the fitted
      kernel is reused for all subsequent iterations (skips L-BFGS each time).
    - Acquisition scores are computed only for still-unlabelled sources.
    - Final trained_score for all sources is computed in one pass at the end.
    """
    feature_arr = features.values          # (n_train, n_features)
    n_total     = len(features)
    h_labels    = np.full(n_total, -1.0)   # -1 = unlabelled
    acq         = np.zeros(n_total)
    fitted_kernel = None
    gpr           = None
    fit_times     = []   # list of (n_labelled, elapsed_s)

    def _fit_and_acquire():
        nonlocal fitted_kernel, gpr
        labelled  = h_labels != -1
        X_train   = feature_arr[labelled]
        y_train   = h_labels[labelled]

        # First call: optimise hyperparameters.  Subsequent calls: fixed kernel.
        kernel = fitted_kernel if fitted_kernel is not None else (
            Matern(length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level_bounds=(1e-3, 1e1))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None if fitted_kernel is not None else 'fmin_l_bfgs_b',
        )
        gpr.fit(X_train, y_train)
        if fitted_kernel is None:
            fitted_kernel = gpr.kernel_

        # Acquisition only for unlabelled sources — no need to predict on known points.
        unlabelled = ~labelled
        if unlabelled.any():
            mean_u, std_u = gpr.predict(feature_arr[unlabelled], return_std=True)
            max_val = float(y_train.max())
            z = (mean_u - max_val - epsilon) / (std_u + 1e-9)
            acq[unlabelled] = (mean_u - max_val - epsilon) * ndtr(z) + std_u * norm.pdf(z)

    # ── Initial seeding ──────────────────────────────────────────────────────
    if initial_steps is not None:
        seed_names = input_anomaly_scores['score'].nlargest(initial_steps).index
        for name in seed_names:
            h_labels[features.index.get_loc(name)] = float(labels.loc[name, 'human_label'])
        _t0 = time.perf_counter()
        _fit_and_acquire()
        if record_timing:
            fit_times.append((int((h_labels != -1).sum()), time.perf_counter() - _t0))

    # ── Active learning loop ─────────────────────────────────────────────────
    while True:
        unlabelled_pos = np.where(h_labels == -1)[0]
        if len(unlabelled_pos) == 0:
            break
        if max_queries is not None and int((h_labels != -1).sum()) >= max_queries:
            break
        top_k = np.argsort(acq[unlabelled_pos])[-steps:]
        for pos in unlabelled_pos[top_k]:
            h_labels[pos] = float(labels.loc[features.index[pos], 'human_label'])
        _t0 = time.perf_counter()
        _fit_and_acquire()
        _elapsed = time.perf_counter() - _t0
        _n_lab = int((h_labels != -1).sum())
        print(f"    [{_n_lab}/{n_total} labelled]  fit={_elapsed:.2f}s", flush=True)
        if record_timing:
            fit_times.append((_n_lab, _elapsed))

    # ── Final prediction on all sources ─────────────────────────────────────
    trained_scores = gpr.predict(feature_arr, return_std=False)
    active_output  = pd.DataFrame({
        'trained_score': trained_scores,
        'acquisition':   acq,
        'human_label':   h_labels,
    }, index=features.index)

    return active_output, gpr, fit_times


# ---------------------------------------------------------------------------
# Baselines (complexity + ellipses) — keyed by data_seed
# ---------------------------------------------------------------------------
def compute_baselines(outputs_root, data_dir, csv_df, labels_all, data_seed,
                      train_idx=None, force=False):
    """Complexity (featurecount) and Ellipses+IsoForest baselines.

    Both are purely unsupervised — no labels used, IsoForest fitted on all images.
    Keyed by data_seed; reused across all runs sharing the same test split.
    If train_idx is provided, also computes train-set curves (complexity_train /
    ellipses_train) using the same fitted IsoForest.
    """
    out_path = outputs_root / f"baselines_{data_seed}.json"
    test_idx_path = data_dir / "test_idx.npy"
    test_hash = hashlib.sha256(open(test_idx_path, "rb").read()).hexdigest()

    if out_path.exists() and not force:
        with open(out_path) as fh:
            cached = json.load(fh)
        if cached.get("test_idx_hash") == test_hash:
            needs_train = train_idx is not None and "complexity_train" not in cached
            if not needs_train:
                print(f"  baselines: loaded from cache (seed={data_seed})", flush=True)
                return cached

    print(f"  baselines: computing (seed={data_seed}) ...", flush=True)
    test_idx = np.load(test_idx_path)
    n_eval   = len(test_idx)
    x        = np.arange(1, n_eval + 1)

    # Build test human labels (for recall curve evaluation only)
    test_lraw = labels_all[test_idx]
    test_ldf  = pd.DataFrame(test_lraw.astype(bool), columns=LABEL_COLS)
    test_hl   = np.ones(n_eval, dtype=int)
    for sv, cols in reversed(TIERS):
        test_hl[test_ldf[cols].any(axis=1).values] = sv
    n_pos = int((test_hl >= 3).sum())

    # ── Baseline 1: Complexity — featurecount from CSV ────────────────────────
    fc         = csv_df["featurecount"].values[test_idx]
    comp_order = np.argsort(fc)[::-1]
    cum_comp   = np.cumsum((test_hl[comp_order] >= 3).astype(int))
    auc_comp   = float(np.trapezoid(cum_comp, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0
    print(f"  baselines: complexity AUC={auc_comp:.4f}", flush=True)

    # ── Baseline 2: Ellipses + IsolationForest on all images ─────────────────
    images    = np.load(IMAGES_PATH).astype(np.float32) / 255.0
    ell_cache = outputs_root / f"_ell_cache_{data_seed}"
    ell_cache.mkdir(exist_ok=True)

    ds  = _NumpyImageDataset(images, output_dir=str(ell_cache), force_rerun=False)
    ext = _sf.EllipseFitFeatures(percentiles=[90, 70, 50, 0], channel=0,
                                  output_dir=str(ell_cache), force_rerun=False)
    all_ell = ext.run_on_dataset(ds)   # shape (N_all, n_features), string int index

    iforest = IsolationForest(n_estimators=200, random_state=data_seed)
    iforest.fit(all_ell.values)

    # Score test sources (negate: higher = more anomalous)
    test_ell   = all_ell.iloc[test_idx]
    ell_scores = -iforest.decision_function(test_ell.values)
    ell_order  = np.argsort(ell_scores)[::-1]
    cum_ell    = np.cumsum((test_hl[ell_order] >= 3).astype(int))
    auc_ell    = float(np.trapezoid(cum_ell, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0
    print(f"  baselines: ellipses  AUC={auc_ell:.4f}", flush=True)

    result = {
        "test_idx_hash": test_hash, "data_seed": data_seed,
        "n_eval": n_eval, "n_pos": n_pos,
        "complexity": {"recall": cum_comp.tolist(), "x": x.tolist(), "auc": auc_comp},
        "ellipses":   {"recall": cum_ell.tolist(),  "x": x.tolist(), "auc": auc_ell},
    }

    # ── Optional: train-set curves (same IsoForest, train split) ─────────────
    if train_idx is not None:
        n_train = len(train_idx)
        x_tr    = np.arange(1, n_train + 1)

        train_lraw = labels_all[train_idx]
        train_ldf  = pd.DataFrame(train_lraw.astype(bool), columns=LABEL_COLS)
        train_hl   = np.ones(n_train, dtype=int)
        for sv, cols in reversed(TIERS):
            train_hl[train_ldf[cols].any(axis=1).values] = sv
        n_pos_tr = int((train_hl >= 3).sum())

        fc_tr    = csv_df["featurecount"].values[train_idx]
        comp_tr  = np.argsort(fc_tr)[::-1]
        cum_comp_tr = np.cumsum((train_hl[comp_tr] >= 3).astype(int))
        auc_comp_tr = float(np.trapezoid(cum_comp_tr, x_tr) / (n_train * n_pos_tr)) if n_pos_tr > 0 else 0.0

        train_ell   = all_ell.iloc[train_idx]
        ell_sc_tr   = -iforest.decision_function(train_ell.values)
        ell_tr      = np.argsort(ell_sc_tr)[::-1]
        cum_ell_tr  = np.cumsum((train_hl[ell_tr] >= 3).astype(int))
        auc_ell_tr  = float(np.trapezoid(cum_ell_tr, x_tr) / (n_train * n_pos_tr)) if n_pos_tr > 0 else 0.0

        result["complexity_train"] = {"recall": cum_comp_tr.tolist(), "x": x_tr.tolist(),
                                      "auc": auc_comp_tr, "n_pos": n_pos_tr}
        result["ellipses_train"]   = {"recall": cum_ell_tr.tolist(),  "x": x_tr.tolist(),
                                      "auc": auc_ell_tr,  "n_pos": n_pos_tr}
        print(f"  baselines: train complexity AUC={auc_comp_tr:.4f}  "
              f"train ellipses AUC={auc_ell_tr:.4f}", flush=True)

    with open(out_path, "w") as fh:
        json.dump(result, fh)
    return result


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------
def process_run(run_dir: Path, epsilon: float,
                steps: int, suffix: str, csv_df: pd.DataFrame, labels_all: np.ndarray,
                use_pca: bool = False, max_queries: int = None, timing_plot: bool = False,
                pca_components: int = None, outputs_root: Path = None, force: bool = False):
    # --- Load data_seed from BYOL checkpoint ---
    ckpt = torch.load(run_dir / "byol_model_best.pt", map_location="cpu", weights_only=False)
    data_seed = int(ckpt["config"]["data_seed"])
    np.random.seed(data_seed)
    print(f"  data_seed={data_seed}", flush=True)

    data_dir     = run_dir / "data"
    protege_dir  = run_dir / "protege"
    protege_dir.mkdir(parents=True, exist_ok=True)

    # --- Load projections and indices ---
    lab_proj  = np.load(data_dir / "labelled_train_projections.npy")
    lab_idx   = np.load(data_dir / "labelled_train_idx.npy")
    unlab_idx = np.load(data_dir / "unlabelled_train_idx.npy")
    test_idx  = np.load(data_dir / "test_idx.npy")
    test_proj = np.load(data_dir / "test_projections.npy")

    unlab_proj_path = data_dir / "unlabelled_train_projections.npy"
    if unlab_proj_path.exists() and len(lab_idx) > 0 and len(unlab_idx) > 0:
        unlab_proj  = np.load(unlab_proj_path)
        train_proj  = np.concatenate([lab_proj, unlab_proj], axis=0)
        train_idx   = np.concatenate([lab_idx, unlab_idx])
    elif len(lab_idx) == 0:
        # f=0: train_projections.npy is the full (unlabelled) set
        train_proj = lab_proj
        train_idx  = unlab_idx
    else:
        # f=1: train_projections.npy is the full (labelled) set
        train_proj = lab_proj
        train_idx  = lab_idx

    all_proj = np.concatenate([train_proj, test_proj], axis=0)
    all_idx  = np.concatenate([train_idx, test_idx])

    baselines = None
    if outputs_root is not None:
        baselines = compute_baselines(outputs_root, data_dir, csv_df, labels_all, data_seed,
                                      train_idx=train_idx, force=force)

    # --- Source names (CSV row order aligned with labels_filtered.npy) ---
    source_names = csv_df.iloc[all_idx]["Source_Name"].values

    # --- Labels ---
    labels_split  = labels_all[all_idx]
    labels_npy_df = pd.DataFrame(labels_split.astype(bool), columns=LABEL_COLS)
    human_labels  = np.ones(len(labels_npy_df), dtype=int)
    for score_val, cols in reversed(TIERS):
        mask = labels_npy_df[cols].any(axis=1).values
        human_labels[mask] = score_val
    labels_df = pd.DataFrame({"human_label": human_labels}, index=source_names)

    # --- Features: optional PCA, then standardise components ---
    if use_pca:
        if pca_components is not None:
            pca = PCA(n_components=pca_components, svd_solver="full")
            proj_final = pca.fit_transform(all_proj)
            print(f"  PCA: {all_proj.shape[1]}D -> {proj_final.shape[1]}D  "
                  f"(fixed {pca_components} components)", flush=True)
        else:
            pca = PCA(n_components=0.95, svd_solver="full")
            proj_final = pca.fit_transform(all_proj)
            print(f"  PCA: {all_proj.shape[1]}D -> {proj_final.shape[1]}D  "
                  f"(explained var >= 0.95)", flush=True)
        # Standardise PCA components — each has a different variance (eigenvalue),
        # so a second pass is needed for the isotropic Matern kernel to work properly.
        proj_final = StandardScaler().fit_transform(proj_final)
    else:
        proj_final = all_proj
        print(f"  Features: {proj_final.shape[1]}D (no PCA)", flush=True)
    features_pca = pd.DataFrame(proj_final, index=source_names)

    # --- Split features into train and test (train = first n_train rows) ---
    n_train_rows   = len(train_idx)
    features_train = features_pca.iloc[:n_train_rows]
    features_test  = features_pca.iloc[n_train_rows:]

    # --- PCA-based seeding: sort by first PCA component, pick equally spaced ---
    _sorted_order   = np.argsort(features_train.values[:, 0])
    _seed_positions = np.linspace(0, n_train_rows - 1, PROTEGE_INITIAL_STEPS, dtype=int)
    _seed_rows      = _sorted_order[_seed_positions]
    seed_names      = features_train.index[_seed_rows]

    anomaly_scores = pd.DataFrame(
        {"score": np.zeros(n_train_rows)}, index=features_train.index
    )
    anomaly_scores.loc[seed_names, "score"] = (
        labels_df.loc[seed_names, "human_label"].values.astype(float)
    )

    # --- ScoreConverter ---
    score_converter = human_loop_learning.ScoreConverter(
        force_rerun=True, output_dir=str(protege_dir)
    )
    anomaly_scores = score_converter.run(anomaly_scores)

    print(f"  GP active learning: initial_steps={PROTEGE_INITIAL_STEPS} (PCA-seeded), "
          f"steps={steps}, epsilon={epsilon}", flush=True)

    active_output, gpr, fit_times = run_GP_active_learning(
        features_train, labels_df.loc[features_train.index], anomaly_scores,
        output_dir=str(protege_dir),
        steps=steps,
        initial_steps=PROTEGE_INITIAL_STEPS,
        epsilon=epsilon,
        max_queries=max_queries,
        record_timing=timing_plot,
    )

    # Score test sources with the fitted GP (no contamination)
    test_scores = gpr.predict(features_test.values)
    test_output = pd.DataFrame({
        "trained_score": test_scores,
        "human_label":   -2,
    }, index=features_test.index)

    # --- Recall curve (eval = fixed test split) ---
    eval_sources = test_output.index
    n_eval       = len(test_output)

    if n_eval == 0:
        print(f"  WARNING: no eval sources (all sources were queried). "
              f"AUC cannot be computed for this run.", flush=True)
        auc   = float("nan")
        n_pos = 0
    else:
        true_labels = labels_df.loc[eval_sources, "human_label"]
        true_pos    = (true_labels >= 3).astype(int)
        n_pos       = int(true_pos.sum())

        eval_scores = test_output.loc[eval_sources, "trained_score"]
        sorted_idx  = eval_scores.sort_values(ascending=False).index
        sorted_pos  = true_pos.loc[sorted_idx].values
        cum_found   = np.cumsum(sorted_pos)
        x           = np.arange(1, n_eval + 1)

        auc = float(np.trapezoid(cum_found, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0

    # --- Train AUC (GP scores on the full train split vs ground-truth labels) ---
    train_labels  = labels_df.loc[active_output.index, "human_label"]
    train_pos     = (train_labels >= 3).astype(int)
    n_pos_train   = int(train_pos.sum())
    if n_pos_train > 0:
        sorted_train  = active_output["trained_score"].sort_values(ascending=False)
        cum_train     = np.cumsum(train_pos.loc[sorted_train.index].values)
        x_train       = np.arange(1, n_train_rows + 1)
        train_auc     = float(np.trapezoid(cum_train, x_train) / (n_train_rows * n_pos_train))
    else:
        train_auc = 0.0

    # --- Save recall curve plot ---
    if n_eval > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, cum_found, label="Protege (BYOL)", linewidth=2)
        ax.plot(x, x * (n_pos / n_eval), "k--", label="Random baseline",
                linewidth=1.5, alpha=0.7)
        ax.axvline(x=n_train_rows, color="grey", linestyle=":", linewidth=1.2,
                   alpha=0.7, label=f"n_train={n_train_rows}")
        ax.set_xlabel("Sources inspected (ranked by Protege score)")
        ax.set_ylabel("Interesting sources found (label >= 3)")
        ax.set_title(f"{run_dir.name}  (AUC={auc:.4f}, {n_pos} positives in eval set of {n_eval})")
        if baselines:
            ax.plot(baselines["complexity"]["x"], baselines["complexity"]["recall"],
                    color="grey", linewidth=1.2, linestyle="--",
                    label=f"Complexity (AUC={baselines['complexity']['auc']:.3f})")
            ax.plot(baselines["ellipses"]["x"], baselines["ellipses"]["recall"],
                    color="grey", linewidth=1.2, linestyle=":",
                    label=f"Ellipses (AUC={baselines['ellipses']['auc']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(protege_dir / f"recall_curve{suffix}.png", dpi=120)
        plt.close(fig)

    # --- Save train recall curve plot ---
    if n_pos_train > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_train, cum_train, label="Protege (BYOL, train)", linewidth=2)
        ax.plot(x_train, x_train * (n_pos_train / n_train_rows), "k--",
                label="Random baseline", linewidth=1.5, alpha=0.7)
        if baselines and "complexity_train" in baselines:
            ax.plot(baselines["complexity_train"]["x"], baselines["complexity_train"]["recall"],
                    color="grey", linewidth=1.2, linestyle="--",
                    label=f"Complexity (AUC={baselines['complexity_train']['auc']:.3f})")
            ax.plot(baselines["ellipses_train"]["x"], baselines["ellipses_train"]["recall"],
                    color="grey", linewidth=1.2, linestyle=":",
                    label=f"Ellipses (AUC={baselines['ellipses_train']['auc']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Train sources ranked by Protege score")
        ax.set_ylabel("Interesting sources found (label >= 3)")
        ax.set_title(f"{run_dir.name} — train  "
                     f"(AUC={train_auc:.4f}, {n_pos_train} positives in {n_train_rows})")
        plt.tight_layout()
        fig.savefig(protege_dir / f"recall_curve_train{suffix}.png", dpi=120)
        plt.close(fig)

    # --- Timing plot ---
    if timing_plot and fit_times:
        _ns, _ts = zip(*fit_times)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(_ns, _ts, marker='o', markersize=3, linewidth=1.5)
        ax.set_xlabel("Labelled training points")
        ax.set_ylabel("GP fit time (s)")
        ax.set_title(f"{run_dir.name}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(protege_dir / f"timing_plot{suffix}.png", dpi=120)
        plt.close(fig)

    # --- Move astronomaly.log → logs/protege.log ---
    import logging as _logging, shutil as _shutil
    _logs_dir = run_dir / "logs"
    _logs_dir.mkdir(parents=True, exist_ok=True)
    for _h in list(_logging.getLogger().handlers):
        if isinstance(_h, _logging.FileHandler) and 'astronomaly.log' in _h.baseFilename:
            _h.close()
            _logging.getLogger().removeHandler(_h)
            _shutil.move(_h.baseFilename, _logs_dir / f"protege{suffix}.log")
            break

    # --- Save scores and summary ---
    combined_output = pd.concat([active_output, test_output])
    combined_output.to_parquet(protege_dir / f"protege_scores{suffix}.parquet")

    summary = {
        "run_dir":          str(run_dir),
        "data_seed":        data_seed,
        "n_labelled_seed":  PROTEGE_INITIAL_STEPS,
        "pca_seeded":       True,
        "steps":            steps,
        "epsilon":          epsilon,
        "n_eval":           n_eval,
        "n_eval_positives": n_pos,
        "test_auc":         None if (isinstance(auc, float) and auc != auc) else auc,
        "train_auc":        train_auc,
        "pca_components":   int(proj_final.shape[1]),
    }
    with open(protege_dir / f"protege_summary{suffix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    auc_str = f"{auc:.4f}" if (isinstance(auc, float) and auc == auc) else "N/A"
    print(f"  -> test_AUC={auc_str}  train_AUC={train_auc:.4f}  eval={n_eval}  positives={n_pos}", flush=True)
    return auc, train_auc, n_eval, n_pos


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be top-level for pickling)
# ---------------------------------------------------------------------------
def _worker_process_run(args):
    rd, epsilon, steps, suffix, csv_df, labels_all, use_pca, max_queries, timing_plot, pca_components, outputs_root, force = args
    m      = re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)
    f_val  = float(m.group(1))
    sw_val = float(m.group(2))
    print(f"[{rd.name}]  f={f_val}  sw={sw_val}", flush=True)
    try:
        auc, train_auc, n_eval, n_pos = process_run(rd, epsilon, steps, suffix, csv_df, labels_all,
                                                     use_pca=use_pca, max_queries=max_queries,
                                                     timing_plot=timing_plot, pca_components=pca_components,
                                                     outputs_root=outputs_root, force=force)
        return dict(name=rd.name, f=f_val, sw=sw_val, auc=auc, train_auc=train_auc,
                    n_eval=n_eval, n_pos=n_pos)
    except Exception as exc:
        import traceback
        print(f"  ERROR in {rd.name}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Protege GP active-learning sweep over BYOL run directories."
    )
    parser.add_argument("--outputs-root",      default="outputs",
                        help="Root directory containing run subdirectories.")
    parser.add_argument("--run-glob",          default="run_no_val_run_f*_sw*",
                        help="Glob pattern for run directories.")
    parser.add_argument("--epsilon",           type=float, default=2.0,
                        help="GP acquisition epsilon: exploration-exploitation trade-off (0=exploit, 3=paper default).")
    parser.add_argument("--steps",             type=int, default=10,
                        help="Sources labelled per GP iteration.")
    parser.add_argument("--output-suffix",     default="",
                        help="Suffix appended to output filenames (e.g. 'ei05').")
    parser.add_argument("--no-pca",            action="store_false", dest="pca",
                        help="Disable PCA dimensionality reduction before Protege GP (on by default).")
    parser.add_argument("--force",             action="store_true",
                        help="Re-run even if protege_summary already exists.")
    parser.add_argument("--workers",           type=int, default=1,
                        help="Number of parallel worker processes (default: 1).")
    parser.add_argument("--max-queries",       type=int, default=None,
                        help="Stop active learning after this many labelled points per run "
                             "(default: label the full train set).")
    parser.add_argument("--timing-plot",       action="store_true",
                        help="Save a GP fit-time vs labelled-points plot for each run.")
    parser.add_argument("--pca-components",    type=int, default=None,
                        help="Fixed number of PCA components (overrides 95%% variance threshold).")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    suffix       = f"_{args.output_suffix}" if args.output_suffix else ""

    # Pre-load shared data once
    csv_df     = pd.read_csv(CSV_PATH)
    labels_all = np.load(LABELS_PATH)

    # Discover run directories
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs if re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)]
    if not run_dirs:
        print(f"No run directories found matching '{args.run_glob}' under {outputs_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs)} run directories.\n")

    # Separate already-done runs (load cached results) from runs that need processing.
    results      = []
    worker_args  = []
    for rd in run_dirs:
        m      = re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)
        f_val  = float(m.group(1))
        sw_val = float(m.group(2))
        summary_path = rd / "protege" / f"protege_summary{suffix}.json"
        if summary_path.exists() and not args.force:
            print(f"[{rd.name}]  skipping (already done — use --force to rerun)", flush=True)
            with open(summary_path) as fh:
                s = json.load(fh)
            results.append(dict(name=rd.name, f=f_val, sw=sw_val,
                                auc=s.get("test_auc", s.get("auc")),
                                train_auc=s.get("train_auc"),
                                n_eval=s.get("n_eval", 0),
                                n_pos=s.get("n_eval_positives", 0)))
            print()
        else:
            worker_args.append((rd, args.epsilon, args.steps, suffix, csv_df, labels_all,
                                args.pca, args.max_queries, args.timing_plot, args.pca_components,
                                outputs_root, args.force))

    if worker_args:
        n_workers = min(args.workers, len(worker_args))
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                worker_results = pool.map(_worker_process_run, worker_args)
        else:
            worker_results = [_worker_process_run(a) for a in worker_args]
        results.extend(r for r in worker_results if r is not None)

    # Summary table
    if results:
        results.sort(key=lambda r: r["auc"] if (isinstance(r["auc"], float) and r["auc"] == r["auc"]) else -1.0, reverse=True)
        print("\n" + "=" * 83)
        print(f"{'Run':<45}  {'f':>5}  {'sw':>6}  {'test_AUC':>8}  {'train_AUC':>9}  {'n_eval':>7}  {'n_pos':>6}")
        print("-" * 83)
        for r in results:
            test_s  = f"{r['auc']:>8.4f}"       if (isinstance(r.get('auc'),       float) and r['auc']       == r['auc'])       else "     N/A"
            train_s = f"{r['train_auc']:>9.4f}"  if (isinstance(r.get('train_auc'), float) and r['train_auc'] == r['train_auc']) else "      N/A"
            print(f"{r['name']:<45}  {r['f']:>5}  {r['sw']:>6}  {test_s}  {train_s}  {r['n_eval']:>7}  {r['n_pos']:>6}")
        print("=" * 83)


if __name__ == "__main__":
    main()
