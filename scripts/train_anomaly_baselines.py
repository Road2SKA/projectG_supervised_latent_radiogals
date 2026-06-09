"""
train_baselines.py — Complexity & Ellipses baselines for Protege benchmarking.

Computes two baselines against Protege (train_protege_list.py):
  1. Complexity — featurecount (morphological feature count from Horton et al.) as
     the sole GP feature; seeded with 10 equally-spaced points along featurecount.
  2. Ellipses — ASTRONOMALY EllipseFitFeatures + IsolationForest scores as GP features;
     run N times to estimate variance.

Outputs mirror train_protege_list.py so the notebook can load them directly.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from astronomaly.anomaly_detection import human_loop_learning
from astronomaly.base.base_dataset import Dataset as _AstroDataset
from astronomaly.feature_extraction import shape_features as _sf
from astronomaly.anomaly_detection import isolation_forest as _isof

# ---------------------------------------------------------------------------
# Hardcoded paths and constants
# ---------------------------------------------------------------------------
CSV_PATH    = Path("/users/mbredber/p3_SUPLAT/data/metadata/lotss_classifications_horton_et_al_2025_filtered.csv")
LABELS_PATH = Path("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss/labels_filtered.npy")
IMAGES_PATH = Path("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss/images_filtered.npy")

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
# _NumpyImageDataset (ASTRONOMALY wrapper for numpy image arrays)
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
# Active-learning function (verbatim from train_protege_list.py)
# ---------------------------------------------------------------------------
def run_GP_active_learning(features, labels, input_anomaly_scores, output_dir,
                           steps=10, initial_steps=None, N_labels=100, epsilon=0.5,
                           max_queries=None, record_timing=False,
                           checkpoint_path=None, checkpoint_interval=100):
    """Direct sklearn GP implementation.

    - Kernel hyperparameters are re-optimised on every fit so the GP adapts
      as more labels are collected.
    - Acquisition scores are computed only for still-unlabelled sources.
    - Final trained_score for all sources is computed in one pass at the end.

    If checkpoint_path is given, h_labels is saved every checkpoint_interval
    iterations so the loop can resume after a job kill.
    """
    feature_arr = features.values          # (n_train, n_features)
    n_total     = len(features)
    h_labels    = np.full(n_total, -1.0)   # -1 = unlabelled
    acq         = np.zeros(n_total)
    gpr         = None
    fit_times   = []   # list of (n_labelled, elapsed_s)
    _iter_count = 0

    # ── Resume from checkpoint if available ──────────────────────────────────
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        h_labels = np.load(checkpoint_path)
        n_resumed = int((h_labels != -1).sum())
        print(f"    Resuming from checkpoint: {n_resumed}/{n_total} already labelled", flush=True)
        _fit_and_acquire_bootstrap = True
    else:
        _fit_and_acquire_bootstrap = False

    def _fit_and_acquire():
        nonlocal gpr
        labelled  = h_labels != -1
        X_train   = feature_arr[labelled]
        y_train   = h_labels[labelled]

        # Re-optimise kernel hyperparameters on every fit so the GP adapts as
        # the labelled set grows (freezing after the first fit locks in a
        # degenerate kernel when early seeds are uninformative).
        gpr = GaussianProcessRegressor(
            kernel=Matern() + WhiteKernel(),
            optimizer='fmin_l_bfgs_b',
        )
        gpr.fit(X_train, y_train)

        # Acquisition only for unlabelled sources — no need to predict on known points.
        unlabelled = ~labelled
        if unlabelled.any():
            mean_u, std_u = gpr.predict(feature_arr[unlabelled], return_std=True)
            max_val = float(y_train.max())
            z = (mean_u - max_val - epsilon) / (std_u + 1e-9)
            acq[unlabelled] = (mean_u - max_val - epsilon) * ndtr(z) + std_u * norm.pdf(z)

    # ── Bootstrap fit after checkpoint restore ───────────────────────────────
    if _fit_and_acquire_bootstrap:
        print(f"    Refitting GP from checkpoint state ...", flush=True)
        _fit_and_acquire()

    # ── Initial seeding (skipped if checkpoint already covers it) ────────────
    if initial_steps is not None and not _fit_and_acquire_bootstrap:
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
        _iter_count += 1
        print(f"    [{_n_lab}/{n_total} labelled]  fit={_elapsed:.2f}s", flush=True)
        if record_timing:
            fit_times.append((_n_lab, _elapsed))
        if checkpoint_path is not None and _iter_count % checkpoint_interval == 0:
            np.save(checkpoint_path, h_labels)

    # ── Final prediction on all sources ─────────────────────────────────────
    trained_scores = gpr.predict(feature_arr, return_std=False)
    active_output  = pd.DataFrame({
        'trained_score': trained_scores,
        'acquisition':   acq,
        'human_label':   h_labels,
    }, index=features.index)

    return active_output, gpr, fit_times


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_labels_df(labels_all, all_idx, source_names):
    """Build a DataFrame with human_label (1-4) indexed by Source_Name."""
    labels_split  = labels_all[all_idx]
    labels_npy_df = pd.DataFrame(labels_split.astype(bool), columns=LABEL_COLS)
    human_labels  = np.ones(len(labels_npy_df), dtype=int)
    for score_val, cols in reversed(TIERS):
        mask = labels_npy_df[cols].any(axis=1).values
        human_labels[mask] = score_val
    return pd.DataFrame({"human_label": human_labels}, index=source_names)


def _compute_auc_and_plot(test_output, labels_df, n_train, output_dir, title_prefix,
                          mean_curve=None, std_curve=None):
    """Compute AUC on test_output and save recall curve plot.

    If mean_curve/std_curve are provided, draw a shaded mean±std band instead
    of a single line (used for aggregate ellipses plot).
    """
    eval_sources = test_output.index
    n_eval       = len(test_output)

    if n_eval == 0:
        print("  WARNING: no eval sources, AUC cannot be computed.", flush=True)
        return float("nan"), 0

    true_labels = labels_df.loc[eval_sources, "human_label"]
    true_pos    = (true_labels >= 3).astype(int)
    n_pos       = int(true_pos.sum())

    eval_scores = test_output["trained_score"]
    sorted_idx  = eval_scores.sort_values(ascending=False).index
    sorted_pos  = true_pos.loc[sorted_idx].values
    cum_found   = np.cumsum(sorted_pos)
    x           = np.arange(1, n_eval + 1)

    auc = float(np.trapezoid(cum_found, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))
    if mean_curve is not None and std_curve is not None:
        x_agg = np.arange(1, len(mean_curve) + 1)
        ax.plot(x_agg, mean_curve, label="Ellipses (mean)", linewidth=2, color="tab:orange")
        ax.fill_between(x_agg, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.25, color="tab:orange", label="±1 std")
    else:
        ax.plot(x, cum_found, label="Protege", linewidth=2)
    ax.plot(x, x * (n_pos / n_eval), "k--", label="Random baseline", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Sources inspected (ranked by Protege score)")
    ax.set_ylabel("Interesting sources found (label >= 3)")
    ax.set_title(f"{title_prefix}  (AUC={auc:.4f}, {n_pos} positives in eval set of {n_eval})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(output_dir) / "recall_curve.png", dpi=120)
    plt.close(fig)

    return auc, n_pos


# ---------------------------------------------------------------------------
# Complexity baseline
# ---------------------------------------------------------------------------
def run_complexity(base_dir, data_dir, csv_df, labels_all, images,
                   train_idx, test_idx, epsilon, steps, max_queries, force):
    out_dir = base_dir / "complexity"
    summary_path = out_dir / "protege_summary.json"
    if summary_path.exists() and not force:
        print("  complexity: already done — skipping (use --force to rerun)", flush=True)
        with open(summary_path) as fh:
            return json.load(fh)

    out_dir.mkdir(parents=True, exist_ok=True)
    print("  complexity: building features from featurecount ...", flush=True)

    all_idx      = np.concatenate([train_idx, test_idx])
    source_names = csv_df.iloc[all_idx]["Source_Name"].values
    labels_df    = _build_labels_df(labels_all, all_idx, source_names)

    # Feature: featurecount (scalar), shape (N, 1) after reshape
    feat_raw = csv_df["featurecount"].values[all_idx].reshape(-1, 1).astype(float)
    scaler   = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_raw)
    features_df = pd.DataFrame(feat_scaled, index=source_names)

    n_train        = len(train_idx)
    features_train = features_df.iloc[:n_train]
    features_test  = features_df.iloc[n_train:]

    # Seed: 10 equally-spaced points along featurecount (ascending)
    sorted_order   = np.argsort(features_train.values[:, 0])
    seed_positions = np.linspace(0, n_train - 1, PROTEGE_INITIAL_STEPS, dtype=int)
    seed_rows      = sorted_order[seed_positions]
    seed_names     = features_train.index[seed_rows]

    anomaly_scores = pd.DataFrame(
        {"score": np.zeros(n_train)}, index=features_train.index
    )
    anomaly_scores.loc[seed_names, "score"] = (
        labels_df.loc[seed_names, "human_label"].values.astype(float)
    )

    score_converter = human_loop_learning.ScoreConverter(
        force_rerun=True, output_dir=str(out_dir)
    )
    anomaly_scores = score_converter.run(anomaly_scores)

    print(f"  complexity: GP active learning  steps={steps}  epsilon={epsilon}", flush=True)
    ckpt_path = out_dir / "gp_checkpoint.npy"
    active_output, gpr, _ = run_GP_active_learning(
        features_train, labels_df.loc[features_train.index], anomaly_scores,
        output_dir=str(out_dir),
        steps=steps,
        initial_steps=PROTEGE_INITIAL_STEPS,
        epsilon=epsilon,
        max_queries=max_queries,
        checkpoint_path=str(ckpt_path),
    )
    ckpt_path.unlink(missing_ok=True)

    test_scores = gpr.predict(features_test.values)
    test_output = pd.DataFrame({
        "trained_score": test_scores,
        "acquisition":   0.0,
        "human_label":   -2.0,
    }, index=features_test.index)

    combined = pd.concat([active_output, test_output])
    combined.to_parquet(out_dir / "protege_scores.parquet")

    auc, n_pos = _compute_auc_and_plot(
        test_output, labels_df, n_train, out_dir,
        title_prefix="Complexity baseline"
    )

    n_eval = len(test_output)
    summary = {
        "method":           "complexity",
        "data_seed":        int(data_dir.name.replace("anomaly_baseline_seed", "")),
        "n_labelled_seed":  PROTEGE_INITIAL_STEPS,
        "steps":            steps,
        "epsilon":          epsilon,
        "n_eval":           n_eval,
        "n_eval_positives": n_pos,
        "auc":              auc,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"  complexity: AUC={auc:.4f}  eval={n_eval}  positives={n_pos}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Ellipses baseline (single run)
# ---------------------------------------------------------------------------
def run_ellipses_single(run_dir, csv_df, labels_all, images,
                        train_idx, test_idx, run_idx,
                        epsilon, steps):
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "protege_summary.json"

    all_idx      = np.concatenate([train_idx, test_idx])
    source_names = csv_df.iloc[all_idx]["Source_Name"].values
    labels_df    = _build_labels_df(labels_all, all_idx, source_names)

    n_train = len(train_idx)

    # Build ellipse features on all sources (train + test)
    tmp_dir = run_dir / "_ell_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"    run {run_idx:02d}: extracting EllipseFit features ...", flush=True)
    images_split = images[all_idx]
    dataset    = _NumpyImageDataset(images_split, output_dir=str(tmp_dir), force_rerun=True)
    extractor  = _sf.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50, 0], channel=0,
        output_dir=str(tmp_dir), force_rerun=True
    )
    ell_feats = extractor.run_on_dataset(dataset)

    # Drop sources whose ellipse features are all-NaN (fit failed)
    nan_mask = ell_feats.isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        print(f"    run {run_idx:02d}: dropping {n_nan}/{len(ell_feats)} NaN feature rows",
              flush=True)
        ell_feats = ell_feats[~nan_mask]

    # Re-derive train/test membership from surviving string indices
    surviving      = np.array([int(s) for s in ell_feats.index])
    is_train       = surviving < n_train
    surv_train_str = [str(i) for i in surviving[is_train]]
    surv_test_str  = [str(i) for i in surviving[~is_train]]

    # Rebuild source_names and labels_df for surviving sources only
    surv_all_idx   = all_idx[surviving]
    source_names   = csv_df.iloc[surv_all_idx]["Source_Name"].values
    labels_df      = _build_labels_df(labels_all, surv_all_idx, source_names)
    str_to_name    = {str(s): source_names[k] for k, s in enumerate(surviving)}

    ell_feats_named = ell_feats.rename(index=str_to_name)
    features_train  = ell_feats_named.loc[[str_to_name[s] for s in surv_train_str]]
    features_test   = ell_feats_named.loc[[str_to_name[s] for s in surv_test_str]]

    # Scale
    scaler = StandardScaler()
    features_train = pd.DataFrame(
        scaler.fit_transform(features_train.values),
        index=features_train.index,
        columns=features_train.columns,
    )
    features_test = pd.DataFrame(
        scaler.transform(features_test.values),
        index=features_test.index,
        columns=features_test.columns,
    )

    # Random seeding: pick PROTEGE_INITIAL_STEPS sources uniformly at random.
    # run_idx seeds the RNG so each run gets a different but reproducible seed set.
    rng = np.random.default_rng(run_idx)
    seed_mask = np.zeros(len(features_train))
    seed_mask[rng.choice(len(features_train), size=PROTEGE_INITIAL_STEPS, replace=False)] = 1.0
    anomaly_scores = pd.DataFrame({"score": seed_mask}, index=features_train.index)

    print(f"    run {run_idx:02d}: GP active learning  steps={steps}  epsilon={epsilon}", flush=True)
    ckpt_path = run_dir / "gp_checkpoint.npy"
    active_output, gpr, _ = run_GP_active_learning(
        features_train, labels_df.loc[features_train.index], anomaly_scores,
        output_dir=str(run_dir),
        steps=steps,
        initial_steps=PROTEGE_INITIAL_STEPS,
        epsilon=epsilon,
        checkpoint_path=str(ckpt_path),
    )
    ckpt_path.unlink(missing_ok=True)

    test_scores = gpr.predict(features_test.values)
    test_output = pd.DataFrame({
        "trained_score": test_scores,
        "acquisition":   0.0,
        "human_label":   -2.0,
    }, index=features_test.index)

    combined = pd.concat([active_output, test_output])
    combined.to_parquet(run_dir / "protege_scores.parquet")

    # Compute recall curve arrays for aggregation (don't save plot per-run)
    eval_sources = test_output.index
    n_eval       = len(test_output)
    true_labels  = labels_df.loc[eval_sources, "human_label"]
    true_pos     = (true_labels >= 3).astype(int)
    n_pos        = int(true_pos.sum())
    sorted_idx   = test_output["trained_score"].sort_values(ascending=False).index
    sorted_pos   = true_pos.loc[sorted_idx].values
    cum_found    = np.cumsum(sorted_pos)
    x            = np.arange(1, n_eval + 1)
    auc          = float(np.trapezoid(cum_found, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0

    summary = {
        "method":           "ellipses",
        "run_idx":          run_idx,
        "steps":            steps,
        "epsilon":          epsilon,
        "n_eval":           n_eval,
        "n_eval_positives": n_pos,
        "auc":              auc,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"    run {run_idx:02d}: AUC={auc:.4f}  eval={n_eval}  positives={n_pos}", flush=True)
    return auc, cum_found.tolist(), n_pos, n_eval


# ---------------------------------------------------------------------------
# Ellipses aggregate: mean±std plot + summary
# ---------------------------------------------------------------------------
def run_ellipses(base_dir, data_dir, csv_df, labels_all, images,
                 train_idx, test_idx, epsilon, steps, n_runs, force):
    ell_dir = base_dir / "ellipses"
    agg_path = ell_dir / "aggregate_summary.json"

    all_idx      = np.concatenate([train_idx, test_idx])
    source_names = csv_df.iloc[all_idx]["Source_Name"].values
    labels_df    = _build_labels_df(labels_all, all_idx, source_names)
    n_eval_total = len(test_idx)

    aucs        = []
    cum_curves  = []

    for i in range(n_runs):
        run_dir      = ell_dir / f"run_{i:02d}"
        summary_path = run_dir / "protege_summary.json"

        if summary_path.exists() and not force:
            print(f"    run {i:02d}: already done — skipping", flush=True)
            with open(summary_path) as fh:
                s = json.load(fh)
            aucs.append(s["auc"])
            # Reconstruct cum_found from parquet for aggregation
            df = pd.read_parquet(run_dir / "protege_scores.parquet")
            test_part = df[df["human_label"] == -2.0]
            tl  = labels_df.loc[test_part.index, "human_label"]
            tp  = (tl >= 3).astype(int)
            si  = test_part["trained_score"].sort_values(ascending=False).index
            cum_curves.append(np.cumsum(tp.loc[si].values).tolist())
        else:
            auc, cum_found, n_pos, n_eval = run_ellipses_single(
                run_dir, csv_df, labels_all, images,
                train_idx, test_idx, run_idx=i,
                epsilon=epsilon, steps=steps,
            )
            aucs.append(auc)
            cum_curves.append(cum_found)

    # Aggregate
    auc_arr = np.array(aucs)
    mean_auc = float(np.mean(auc_arr))
    std_auc  = float(np.std(auc_arr))

    # Mean ± std recall curve
    min_len = min(len(c) for c in cum_curves)
    curves  = np.array([c[:min_len] for c in cum_curves])
    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    # Use last run's test_output for AUC/n_pos labels on aggregate plot
    last_run_dir = ell_dir / f"run_{n_runs - 1:02d}"
    last_df = pd.read_parquet(last_run_dir / "protege_scores.parquet")
    test_part = last_df[last_df["human_label"] == -2.0]
    tl  = labels_df.loc[test_part.index, "human_label"]
    n_pos_last = int((tl >= 3).sum())

    # Aggregate plot
    x = np.arange(1, min_len + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mean_curve, label=f"Ellipses (mean AUC={mean_auc:.4f})", linewidth=2, color="tab:orange")
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.25, color="tab:orange", label="±1 std")
    ax.plot(x, x * (n_pos_last / min_len), "k--", label="Random baseline", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Sources inspected (ranked by Protege score)")
    ax.set_ylabel("Interesting sources found (label >= 3)")
    ax.set_title(f"Ellipses baseline — {n_runs} runs  "
                 f"(mean AUC={mean_auc:.4f} ± {std_auc:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(ell_dir / "recall_curve.png", dpi=120)
    plt.close(fig)

    agg_summary = {
        "method":    "ellipses",
        "n_runs":    n_runs,
        "mean_auc":  mean_auc,
        "std_auc":   std_auc,
        "all_aucs":  aucs,
        "steps":     steps,
        "epsilon":   epsilon,
    }
    with open(agg_path, "w") as fh:
        json.dump(agg_summary, fh, indent=2)

    print(f"  ellipses: mean AUC={mean_auc:.4f} ± {std_auc:.4f}  ({n_runs} runs)", flush=True)
    return agg_summary


# ---------------------------------------------------------------------------
# BYOL + IsolationForest baseline
# ---------------------------------------------------------------------------
def run_byol_isoforest(run_dir, csv_df, labels_all,
                       use_pca=True, pca_components=None, force=False):
    """IsolationForest on BYOL embeddings — same features as Protege, no labels needed.

    Uses the identical PCA + StandardScaler pipeline as train_protege_list.py so
    the only difference vs. Protege is the absence of GP active learning.
    Outputs are saved under <run_dir>/baselines/byol_isoforest/.
    """
    out_dir      = run_dir / "baselines" / "byol_isoforest"
    summary_path = out_dir / "protege_summary.json"

    if summary_path.exists() and not force:
        print(f"  byol_isoforest [{run_dir.name}]: already done — skipping "
              "(use --force to rerun)", flush=True)
        with open(summary_path) as fh:
            return json.load(fh)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"

    # ── Load projections (identical logic to train_protege_list.py) ──────────
    lab_proj  = np.load(data_dir / "labelled_train_projections.npy")
    lab_idx   = np.load(data_dir / "labelled_train_idx.npy")
    unlab_idx = np.load(data_dir / "unlabelled_train_idx.npy")
    test_idx  = np.load(data_dir / "test_idx.npy")
    test_proj = np.load(data_dir / "test_projections.npy")

    unlab_proj_path = data_dir / "unlabelled_train_projections.npy"
    if unlab_proj_path.exists() and len(lab_idx) > 0 and len(unlab_idx) > 0:
        unlab_proj = np.load(unlab_proj_path)
        train_proj = np.concatenate([lab_proj, unlab_proj], axis=0)
        train_idx  = np.concatenate([lab_idx, unlab_idx])
    elif len(lab_idx) == 0:
        train_proj = lab_proj
        train_idx  = unlab_idx
    else:
        train_proj = lab_proj
        train_idx  = lab_idx

    all_idx      = np.concatenate([train_idx, test_idx])
    source_names = csv_df.iloc[all_idx]["Source_Name"].values
    labels_df    = _build_labels_df(labels_all, all_idx, source_names)
    n_train      = len(train_idx)
    train_names  = source_names[:n_train]
    test_names   = source_names[n_train:]

    # ── PCA + standardise (identical to Protege) ─────────────────────────────
    all_proj = np.concatenate([train_proj, test_proj], axis=0)
    if use_pca:
        if pca_components is not None:
            pca = PCA(n_components=pca_components, svd_solver="full")
        else:
            pca = PCA(n_components=0.95, svd_solver="full")
        all_proj = pca.fit_transform(all_proj)
        print(f"  byol_isoforest [{run_dir.name}]: PCA -> {all_proj.shape[1]}D", flush=True)

    all_proj = StandardScaler().fit_transform(all_proj)
    train_feats = all_proj[:n_train]
    test_feats  = all_proj[n_train:]

    # ── Fit IsolationForest on train, score test ──────────────────────────────
    print(f"  byol_isoforest [{run_dir.name}]: fitting IsolationForest ...", flush=True)
    iforest = IsolationForest(random_state=42)
    iforest.fit(train_feats)
    # decision_function: lower (more negative) = more anomalous
    test_scores = iforest.decision_function(test_feats)

    # ── Recall curve ─────────────────────────────────────────────────────────
    sorted_order = np.argsort(test_scores)          # ascending: most anomalous first
    true_labels  = labels_df.loc[test_names, "human_label"].values
    sorted_pos   = (true_labels[sorted_order] >= 3).astype(int)
    n_pos        = int(sorted_pos.sum())
    n_eval       = len(test_names)
    cum_found    = np.cumsum(sorted_pos)
    x            = np.arange(1, n_eval + 1)
    auc          = float(np.trapezoid(cum_found, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0

    # ── Save scores (negate so higher = more anomalous, consistent with Protege) ──
    test_output = pd.DataFrame({
        "trained_score": -test_scores,
        "human_label":   -2.0,
    }, index=test_names)
    test_output.to_parquet(out_dir / "protege_scores.parquet")

    summary = {
        "method":           "byol_isoforest",
        "run_dir":          str(run_dir),
        "n_eval":           n_eval,
        "n_eval_positives": n_pos,
        "auc":              auc,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"  byol_isoforest [{run_dir.name}]: AUC={auc:.4f}  "
          f"eval={n_eval}  positives={n_pos}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Complexity, Ellipses, and BYOL-IsolationForest baselines."
    )
    parser.add_argument("--data-seed",      type=int,   default=42,
                        help="Random seed for 70/30 train/test split (complexity/ellipses).")
    parser.add_argument("--outputs-root",   default="outputs",
                        help="Root directory for output.")
    parser.add_argument("--method",         default="both",
                        choices=["complexity", "ellipses", "both", "byol_isoforest"],
                        help="Which baseline(s) to run.")
    parser.add_argument("--epsilon",        type=float, default=0.5,
                        help="GP acquisition epsilon (complexity/ellipses).")
    parser.add_argument("--steps",          type=int,   default=10,
                        help="Labels per GP iteration (complexity/ellipses).")
    parser.add_argument("--max-queries",    type=int,   default=None,
                        help="Stop after N labelled points (complexity/ellipses).")
    parser.add_argument("--n-runs",         type=int,   default=10,
                        help="Number of ellipses runs (for variance estimate).")
    parser.add_argument("--run-glob",       default="run_*_f*_sw*",
                        help="Glob pattern for BYOL run directories (byol_isoforest only).")
    parser.add_argument("--no-pca",         action="store_false", dest="pca",
                        help="Disable PCA for byol_isoforest (on by default).")
    parser.add_argument("--pca-components", type=int,   default=None,
                        help="Fixed PCA components for byol_isoforest (default: 95%% variance).")
    parser.add_argument("--force",          action="store_true",
                        help="Re-run even if outputs exist.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)

    print(f"Loading data ...", flush=True)
    csv_df     = pd.read_csv(CSV_PATH)
    labels_all = np.load(LABELS_PATH)

    if args.method == "byol_isoforest":
        run_dirs = sorted([
            rd for rd in outputs_root.glob(args.run_glob)
            if rd.is_dir() and re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)
        ])
        if not run_dirs:
            print(f"No run directories found matching '{args.run_glob}' under {outputs_root}",
                  flush=True)
            sys.exit(1)
        print(f"Found {len(run_dirs)} run directories.\n")

        results = []
        for rd in run_dirs:
            summary = run_byol_isoforest(
                rd, csv_df, labels_all,
                use_pca=args.pca, pca_components=args.pca_components,
                force=args.force,
            )
            if summary:
                m = re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)
                results.append(dict(
                    name=rd.name,
                    f=float(m.group(1)), sw=float(m.group(2)),
                    auc=summary["auc"],
                    n_eval=summary["n_eval"],
                    n_pos=summary["n_eval_positives"],
                ))

        if results:
            results.sort(key=lambda r: r["auc"], reverse=True)
            print("\n" + "=" * 75)
            print(f"{'Run':<45}  {'f':>5}  {'sw':>5}  {'AUC':>7}  {'n_eval':>6}  {'n_pos':>5}")
            print("-" * 75)
            for r in results:
                print(f"{r['name']:<45}  {r['f']:>5.2f}  {r['sw']:>5.1f}  "
                      f"{r['auc']:>7.4f}  {r['n_eval']:>6}  {r['n_pos']:>5}")
            print("=" * 75)

    else:
        images       = np.load(IMAGES_PATH).astype(np.float32) / 255.0
        base_dir     = outputs_root / f"anomaly_baseline_seed{args.data_seed}"
        data_dir_out = base_dir / "data"
        data_dir_out.mkdir(parents=True, exist_ok=True)

        # 70/30 split — same logic as train_byol.py
        all_idx = np.arange(len(images))
        train_idx, test_idx = train_test_split(all_idx, test_size=0.30, random_state=args.data_seed)

        np.save(data_dir_out / "train_idx.npy", train_idx)
        np.save(data_dir_out / "test_idx.npy",  test_idx)
        print(f"Split: {len(train_idx)} train / {len(test_idx)} test  (seed={args.data_seed})",
              flush=True)

        if args.method in ("complexity", "both"):
            print("\n[Complexity baseline]", flush=True)
            run_complexity(
                base_dir, data_dir_out, csv_df, labels_all, images,
                train_idx, test_idx,
                epsilon=args.epsilon, steps=args.steps,
                max_queries=args.max_queries, force=args.force,
            )

        if args.method in ("ellipses", "both"):
            print(f"\n[Ellipses baseline — {args.n_runs} runs]", flush=True)
            run_ellipses(
                base_dir, data_dir_out, csv_df, labels_all, images,
                train_idx, test_idx,
                epsilon=args.epsilon, steps=args.steps,
                n_runs=args.n_runs, force=args.force,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
