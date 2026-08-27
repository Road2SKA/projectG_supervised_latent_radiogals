"""Compute Ellipses GP and blob_doh complexity baselines for a given data split.

Writes outputs/anomaly_baselines/baselines_{data_seed}.json in the format
expected by the rare_object_detection.ipynb recall-curve cell, so that
multiple seeds can be aggregated for error bars.

Usage
-----
  python scripts/compute_baselines.py --data-seed 3
  python scripts/compute_baselines.py --data-seed 3 --force   # overwrite cache

The ellipse feature cache from compute_noise_robustness.py is reused when
present (outputs/anomaly_baselines/_ell_cache_{seed}/EllipseFitFeatures_output.parquet).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astronomaly.base.base_dataset import Dataset as _AstroDataset
from astronomaly.feature_extraction import shape_features as _sf
from skimage.feature import blob_doh
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from suplat.utils.class_weights import LABEL_COLS, TIERS

IMAGES_PATH  = ROOT / "data/preprocessed/lotss/images_filtered.npy"
LABELS_PATH  = ROOT / "data/preprocessed/lotss/labels_filtered.npy"
OUTPUTS_ROOT = ROOT / "outputs"
BASELINE_DIR = OUTPUTS_ROOT / "anomaly_baselines"

POSITIVE_THRESHOLD = 3
DOH_PARAMS = {"min_sigma": 1.0, "max_sigma": 15, "num_sigma": 10, "threshold": 0.005}


# ── Astronomaly dataset wrapper ───────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_human_labels(labels_raw):
    ldf = pd.DataFrame(labels_raw.astype(bool), columns=LABEL_COLS)
    hl  = np.ones(len(labels_raw), dtype=int)
    for sv, cols in reversed(TIERS):
        hl[ldf[cols].any(axis=1).values] = sv
    return hl


def recall_auc(scores, true_hl, pos_thresh=POSITIVE_THRESHOLD):
    order  = np.argsort(scores)[::-1]
    binary = (true_hl[order] >= pos_thresh).astype(int)
    cum    = np.cumsum(binary)
    n_pos  = int(binary.sum())
    if n_pos == 0:
        return np.zeros(len(cum), dtype=int), np.arange(1, len(cum) + 1), 0.0
    x   = np.arange(1, len(cum) + 1)
    auc = float(np.trapezoid(cum, x) / (len(cum) * n_pos))
    return cum, x, auc


def extract_ell_features(images_f32, cache_dir, force_rerun=False):
    """Load ellipse features from cache or compute and cache them."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet = cache_dir / "EllipseFitFeatures_output.parquet"
    if parquet.exists() and not force_rerun:
        df = pd.read_parquet(parquet)
        print(f"  Loaded ellipse features from cache ({len(df)} sources)")
        return df
    print(f"  Extracting ellipse features ({len(images_f32)} images) …")
    ds  = _NumpyImageDataset(images_f32, output_dir=str(cache_dir), force_rerun=False)
    ext = _sf.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50, 0], channel=0,
        output_dir=str(cache_dir), force_rerun=force_rerun,
        upper_limit=150,
    )
    return ext.run_on_dataset(ds)


def fit_ell_gp(tr_ell_df, y_lab):
    """Fit ellipses GP on labeled train. Returns (gpr, scaler, train_medians)."""
    tr_nan     = tr_ell_df.isna().any(axis=1)
    tr_medians = tr_ell_df[~tr_nan].median()
    n_nan      = int(tr_nan.sum())
    if n_nan:
        print(f"  Ellipses train: imputing {n_nan}/{len(tr_ell_df)} NaN rows")
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(tr_ell_df.fillna(tr_medians).values)
    gpr = GaussianProcessRegressor(
        kernel=Matern(length_scale_bounds=(1e-2, 1e2))
               + WhiteKernel(noise_level_bounds=(1e-3, 1e1)),
        normalize_y=True,
    )
    gpr.fit(X_tr, y_lab)
    return gpr, scaler, tr_medians


def predict_ell_gp(te_ell_df, gpr, scaler, tr_medians):
    return gpr.predict(scaler.transform(te_ell_df.fillna(tr_medians).values))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute recall-curve baselines per data seed")
    parser.add_argument("--data-seed", type=int, required=True,
                        help="Data seed (used to locate splits and name output file)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    out_path = BASELINE_DIR / f"baselines_{args.data_seed}.json"
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check cache ───────────────────────────────────────────────────────────
    if out_path.exists() and not args.force:
        with open(out_path) as fh:
            cached = json.load(fh)
        has_ell  = cached.get("ellipses_gp") is not None
        has_comp = cached.get("complexity") is not None
        if has_ell and has_comp:
            print(f"Output exists and complete: {out_path}\nUse --force to overwrite.")
            return
        print(f"Output exists but incomplete (ell={has_ell}, comp={has_comp}), continuing …")
    else:
        cached = {}

    splits_dir = OUTPUTS_ROOT / "data_splits" / str(args.data_seed)
    print(f"\ndata_seed={args.data_seed}  splits: {splits_dir}")

    # ── Load split indices (f=1 runs) ─────────────────────────────────────────
    lab_idx  = np.load(splits_dir / "labelled_train_idx_f1.npy")
    test_idx = np.load(splits_dir / "test_idx.npy")

    labels_all = np.load(LABELS_PATH)
    n_total    = labels_all.shape[0]

    # Extend test_idx if it only covers the Protege budget subset
    _n_train = len(lab_idx)
    if len(test_idx) < (n_total - _n_train):
        test_idx = np.setdiff1d(np.arange(n_total), lab_idx)
        print(f"Derived full test split: {len(test_idx)} entries")
    else:
        print(f"Test set size: {len(test_idx)}")

    test_idx_hash = hashlib.sha256(
        open(splits_dir / "test_idx.npy", "rb").read()
    ).hexdigest()

    lab_hl  = build_human_labels(labels_all[lab_idx])
    test_hl = build_human_labels(labels_all[test_idx])
    n_pos   = int((test_hl >= POSITIVE_THRESHOLD).sum())
    print(f"Labeled train: {len(lab_idx)}  test positives (tier≥{POSITIVE_THRESHOLD}): {n_pos}")
    y_lab = lab_hl.astype(float)

    result = {
        "test_idx_hash": test_idx_hash,
        "data_seed":     args.data_seed,
        "n_eval":        int(len(test_idx)),
        "n_pos":         n_pos,
    }

    # ── Ellipse features ──────────────────────────────────────────────────────
    print("\n── Ellipse features ──")
    images_mmap   = np.load(IMAGES_PATH, mmap_mode="r")
    all_images_f  = images_mmap.astype(np.float32) / 255.0
    ell_cache_dir = BASELINE_DIR / f"_ell_cache_{args.data_seed}"
    all_ell = extract_ell_features(all_images_f, ell_cache_dir,
                                   force_rerun=args.force)
    del all_images_f

    # ── Ellipses GP ───────────────────────────────────────────────────────────
    print("\n── Ellipses GP ──")
    tr_ell  = all_ell.iloc[lab_idx]
    te_ell  = all_ell.iloc[test_idx]
    gpr_ell, ell_sc, ell_med = fit_ell_gp(tr_ell, y_lab)
    print(f"Fitted on {len(tr_ell)} labeled sources")

    ell_scores = predict_ell_gp(te_ell, gpr_ell, ell_sc, ell_med)
    cum, x, auc = recall_auc(ell_scores, test_hl)
    print(f"Ellipses GP AUC={auc:.4f}")
    result["ellipses_gp"] = {
        "recall": cum.tolist(),
        "x":      x.tolist(),
        "auc":    auc,
    }

    # ── blob_doh complexity ───────────────────────────────────────────────────
    print("\n── blob_doh complexity ──")
    test_images = np.array(images_mmap[test_idx]).astype(np.float32)
    del images_mmap

    def _doh_score(img):
        lo, hi = img.min(), img.max()
        norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
        return float(len(blob_doh(norm, **DOH_PARAMS)))

    doh_scores = np.array([_doh_score(test_images[i]) for i in range(len(test_images))],
                          dtype=float)
    del test_images
    cum, x, auc = recall_auc(doh_scores, test_hl)
    print(f"blob_doh complexity AUC={auc:.4f}")
    result["complexity"] = {
        "recall": cum.tolist(),
        "x":      x.tolist(),
        "auc":    auc,
        "method": "blob_doh",
        "params": DOH_PARAMS,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
