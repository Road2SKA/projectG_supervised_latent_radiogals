"""Merged noise robustness sweep: Complexity, BYOL GP, BYOL+Complexity GP, Ellipses GP.

All methods use the "train on clean, evaluate on noisy" paradigm. Noisy test
images are generated once per sigma level and shared across all baselines.

  Complexity        : blob_doh blob count on noisy test images (direct score).
  BYOL GP           : GP fitted on clean labeled-train encodings; predicts on
                      noisy re-encoded test images.
  BYOL+Complexity GP: same, but each feature vector is augmented with its
                      (standardised) blob count.
  Ellipses GP       : GP fitted on clean labeled-train ellipse features; predicts
                      on ellipse features re-extracted from noisy test images.

Backwards-compatible individual JSONs are also written so existing notebook cells
continue to work without modification.

Outputs
-------
  outputs/anomaly_baselines/noise_robustness/noise_robustness.json
  outputs/anomaly_baselines/noise_robustness/comp_noise_counts.npz
  outputs/anomaly_baselines/noise_robustness/comp_noise_robustness.json  (compat)
  outputs/anomaly_baselines/noise_robustness/byol_noise_robustness.json  (compat)
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astronomaly.base.base_dataset import Dataset as _AstroDataset
from astronomaly.feature_extraction import shape_features as _sf
from skimage.feature import blob_doh
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from suplat.models.byol_models import BYOLEfficientNetB0
from suplat.utils.class_weights import LABEL_COLS, TIERS

IMAGES_PATH    = ROOT / "data/preprocessed/lotss/images_filtered.npy"
LABELS_PATH    = ROOT / "data/preprocessed/lotss/labels_filtered.npy"
BYOL_RUNS_ROOT = ROOT / "outputs/byol_runs"
OUT_ROOT       = ROOT / "outputs/anomaly_baselines/noise_robustness"

NOISE_SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]


# ── Astronomaly dataset wrapper ────────────────────────────────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_human_labels(labels_raw):
    ldf = pd.DataFrame(labels_raw.astype(bool), columns=LABEL_COLS)
    hl  = np.ones(len(labels_raw), dtype=int)
    for sv, cols in reversed(TIERS):
        hl[ldf[cols].any(axis=1).values] = sv
    return hl


def recall_auc(scores, true_hl, pos_thresh=3):
    order  = np.argsort(scores)[::-1]
    binary = (true_hl[order] >= pos_thresh).astype(int)
    cum    = np.cumsum(binary)
    n_pos  = int(binary.sum())
    if n_pos == 0:
        return 0.0
    x = np.arange(1, len(cum) + 1)
    return float(np.trapezoid(cum, x) / (len(cum) * n_pos))


def add_noise(images, sigma_rel, rng):
    if sigma_rel == 0.0:
        return images.astype(np.float32)
    per_std = images.astype(np.float32).std(axis=(-2, -1), keepdims=True).clip(1e-6)
    return (images.astype(np.float32)
            + rng.normal(0, sigma_rel, images.shape).astype(np.float32) * per_std)


def blob_doh_complexity(image_f32):
    lo, hi = image_f32.min(), image_f32.max()
    img_norm = (image_f32 - lo) / (hi - lo) if hi > lo else np.zeros_like(image_f32)
    return float(len(blob_doh(img_norm, min_sigma=1, max_sigma=15,
                              num_sigma=10, threshold=0.005)))


def extract_ell_features(images_01, cache_dir, force_rerun=False):
    """Run EllipseFitFeatures on images_01 (float [0,1]), caching to cache_dir."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds  = _NumpyImageDataset(images_01, output_dir=str(cache_dir),
                              force_rerun=force_rerun)
    ext = _sf.EllipseFitFeatures(percentiles=[90, 80, 70, 60, 50, 0], channel=0,
                                  output_dir=str(cache_dir), force_rerun=force_rerun,
                                  upper_limit=150)
    return ext.run_on_dataset(ds)


def fit_ell_gp(tr_ell_df, y_lab):
    """Fit ellipses GP on labeled train. Returns (gp, scaler, train_medians)."""
    tr_nan     = tr_ell_df.isna().any(axis=1)
    tr_medians = tr_ell_df[~tr_nan].median()
    n_nan_tr   = int(tr_nan.sum())
    if n_nan_tr:
        print(f"  Ellipses train: imputing {n_nan_tr}/{len(tr_ell_df)} NaN rows with column medians")
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Merged noise robustness sweep")
    parser.add_argument("--byol-run",  type=str, required=True,
                        help="Name of the BYOL run directory under outputs/byol_runs/.")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--force",     action="store_true",
                        help="Overwrite existing outputs.")
    args = parser.parse_args()

    # Extract f-label suffix from run name (e.g. _f1 or _f0.1)
    _fm = re.search(r'_f([\d.]+)', args.byol_run)
    f_tag = f"_f{_fm.group(1)}" if _fm else ""

    rd             = BYOL_RUNS_ROOT / args.byol_run
    seed_dir       = rd / f"seed{args.seed}"
    BYOL_OUT_DIR   = seed_dir / "data" / "anomaly"
    BASELINE_OUT_DIR = ROOT / "outputs/anomaly_baselines"
    BYOL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = BYOL_OUT_DIR / "noise_robustness.json"
    if out_json.exists() and not args.force:
        print(f"Output exists: {out_json}\nUse --force to overwrite.")
        return

    splits_dir = ROOT / "outputs/data_splits" / str(args.data_seed)

    # ── Load split ────────────────────────────────────────────────────────────
    lab_idx  = np.load(splits_dir / f"labelled_train_idx{f_tag}.npy")
    unl_idx  = np.load(splits_dir / f"unlabelled_train_idx{f_tag}.npy")
    test_idx = np.load(splits_dir / "test_idx.npy")

    labels_all = np.load(LABELS_PATH)
    n_total    = labels_all.shape[0]
    _n_train   = len(lab_idx) + len(unl_idx)
    if len(test_idx) < (n_total - _n_train):
        train_idx = np.concatenate([lab_idx, unl_idx])
        test_idx  = np.setdiff1d(np.arange(n_total), train_idx)
        print(f"Derived full test split: {len(test_idx)} entries")
    else:
        print(f"Test set size: {len(test_idx)}")

    test_idx_sha256 = hashlib.sha256(test_idx.astype(np.int64).tobytes()).hexdigest()
    print(f"test_idx sha256: {test_idx_sha256[:16]}…")

    lab_hl  = build_human_labels(labels_all[lab_idx])
    test_hl = build_human_labels(labels_all[test_idx])
    print(f"Labeled train: {len(lab_idx)}  (tier≥3: {(lab_hl >= 3).sum()})")
    print(f"Test positives (tier≥3): {(test_hl >= 3).sum()}")
    y_lab = lab_hl.astype(float)

    # ── BYOL model ────────────────────────────────────────────────────────────
    # Load model first: labelled_train_encodings.npy may contain all train
    # sources (not just the 532 labeled ones) depending on how train_byol.py
    # was invoked, so we re-encode labeled train images directly to guarantee
    # the correct 532-row matrix regardless of file convention.
    print("\n── Loading BYOL model ──")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(seed_dir / "byol_model_best.pt", map_location="cpu",
                        weights_only=False)
    model  = BYOLEfficientNetB0(
        projection_dim=ckpt["config"].get("projection_dim", 256),
        hidden_dim=ckpt["config"].get("hidden_dim", 4096),
    )
    sd = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    print(f"Model on {device}")

    def encode(imgs_01, batch=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs_01), batch):
                x = (torch.from_numpy(imgs_01[i:i+batch])
                     .unsqueeze(1).float().to(device))
                out.append(model.online_encoder(x).float().cpu().numpy())
        return np.vstack(out)

    # ── PCA fitted on all pre-saved encodings ─────────────────────────────────
    print("\n── BYOL setup ──")
    tr_e  = np.load(seed_dir / "data/byol/labelled_train_encodings.npy")
    te_e  = np.load(seed_dir / "data/byol/test_encodings.npy")
    unl_p = seed_dir / "data/byol/unlabelled_train_encodings.npy"
    if unl_p.exists():
        tr_e = np.concatenate([tr_e, np.load(unl_p)], 0)
    all_e = np.concatenate([tr_e, te_e], 0)

    pca = PCA(n_components=0.95, svd_solver="full").fit(all_e)
    sc  = StandardScaler().fit(pca.transform(all_e))
    print(f"PCA: {pca.n_components_} components  |  pre-saved test encodings: {len(te_e)}")

    # Re-encode both labeled train and clean test images through the model so
    # that Xtr_lab and Xte are always from the same encoder (online_encoder).
    # The pre-saved encodings above are used only for fitting the PCA/scaler.
    images_mmap = np.load(IMAGES_PATH, mmap_mode="r")
    lab_images  = np.array(images_mmap[lab_idx])
    lab_enc     = encode(lab_images.astype(np.float32) / 255.0)
    Xtr_lab     = sc.transform(pca.transform(lab_enc))
    print(f"Re-encoded {len(Xtr_lab)} labeled train sources → Xtr_lab {Xtr_lab.shape}")

    test_images_clean = np.array(images_mmap[test_idx])
    clean_enc = encode(test_images_clean.astype(np.float32) / 255.0)
    Xte       = sc.transform(pca.transform(clean_enc))
    print(f"Re-encoded {len(Xte)} clean test sources → Xte {Xte.shape}")

    gpr_byol = GaussianProcessRegressor(
        kernel=Matern(length_scale_bounds=(1e-2, 1e2))
               + WhiteKernel(noise_level_bounds=(1e-3, 1e1)),
        normalize_y=True,
    )
    gpr_byol.fit(Xtr_lab, y_lab)
    print(f"BYOL GP fitted on {len(Xtr_lab)} labeled sources")

    # σ=0 BYOL GP AUC: read from protege enc_pca summary (same enc→PCA feature space
    # as σ>0 re-encoding; proj_nopca would mix feature spaces on the same curve).
    _byol_sigma0_auc = None
    for _sfx in ["enc_pca", "proj_nopca"]:
        _psum_path = seed_dir / "data/protege" / f"protege_summary_{_sfx}.json"
        if _psum_path.exists():
            with open(_psum_path) as _fh:
                _byol_sigma0_auc = json.load(_fh)["test_auc"]
            print(f"σ=0 BYOL GP AUC read from protege summary ({_sfx}): {_byol_sigma0_auc:.4f}")
            break

    # ── Clean ellipse features → Ellipses GP ──────────────────────────────────
    print("\n── Ellipses setup ──")
    ell_clean_dir = ROOT / "outputs/anomaly_baselines" / f"_ell_cache_{args.data_seed}"
    all_images_f  = images_mmap.astype(np.float32) / 255.0
    all_ell = extract_ell_features(all_images_f, ell_clean_dir, force_rerun=False)
    del all_images_f
    print(f"Ellipse features shape: {all_ell.shape}")

    tr_ell_lab   = all_ell.iloc[lab_idx]
    te_ell_clean = all_ell.iloc[test_idx]
    gpr_ell, ell_sc, ell_med = fit_ell_gp(tr_ell_lab, y_lab)
    print(f"Ellipses GP fitted on {len(tr_ell_lab)} labeled sources")

    test_images = test_images_clean   # already loaded above
    del images_mmap

    rng = np.random.default_rng(args.seed)

    comp_auc_list   = []
    byol_auc_list   = []
    ell_auc_list    = []
    comp_scores_npz    = {}   # str(sigma) → float32 array, for npz

    # ── Sigma sweep ───────────────────────────────────────────────────────────
    print()
    for sigma in NOISE_SIGMAS:
        print(f"── σ={sigma} ──", flush=True)
        noisy = add_noise(test_images, sigma, rng)   # (n_test, H, W) float32

        # Complexity
        blob_scores = np.array(
            [blob_doh_complexity(noisy[i]) for i in range(len(noisy))],
            dtype=np.float32,
        )
        c_auc = recall_auc(blob_scores.astype(float), test_hl)
        comp_auc_list.append(c_auc)
        comp_scores_npz[str(sigma)] = blob_scores
        print(f"  Complexity       AUC={c_auc:.4f}  (mean blobs={blob_scores.mean():.2f})",
              flush=True)

        # BYOL GP
        if sigma == 0.0:
            Xte_cur = Xte
            if _byol_sigma0_auc is not None:
                b_auc = _byol_sigma0_auc
                print(f"  BYOL GP          AUC={b_auc:.4f}  (protege summary)", flush=True)
            else:
                b_auc = recall_auc(gpr_byol.predict(Xte_cur), test_hl)
                print(f"  BYOL GP          AUC={b_auc:.4f}", flush=True)
        else:
            enc     = encode(noisy / 255.0)
            Xte_cur = sc.transform(pca.transform(enc))
            b_auc   = recall_auc(gpr_byol.predict(Xte_cur), test_hl)
            print(f"  BYOL GP          AUC={b_auc:.4f}", flush=True)
        byol_auc_list.append(b_auc)

        # Ellipses GP
        if sigma == 0.0:
            n_nan_0 = int(te_ell_clean.isna().any(axis=1).sum())
            if n_nan_0:
                print(f"  Ellipses test σ=0: imputing {n_nan_0}/{len(te_ell_clean)} NaN rows with train medians",
                      flush=True)
            ell_scores = predict_ell_gp(te_ell_clean, gpr_ell, ell_sc, ell_med)
        else:
            noisy_ell_dir = BASELINE_OUT_DIR / f"_ell_sigma{sigma}_cache_{args.data_seed}"
            te_ell_noisy  = extract_ell_features(noisy / 255.0, noisy_ell_dir,
                                                  force_rerun=False)
            n_nan = int(te_ell_noisy.isna().any(axis=1).sum())
            if n_nan:
                print(f"  Ellipses test σ={sigma}: imputing {n_nan}/{len(te_ell_noisy)} NaN rows with train medians",
                      flush=True)
            ell_scores = predict_ell_gp(te_ell_noisy, gpr_ell, ell_sc, ell_med)
        e_auc = recall_auc(ell_scores, test_hl)
        ell_auc_list.append(e_auc)
        print(f"  Ellipses GP      AUC={e_auc:.4f}", flush=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    npz_path = BASELINE_OUT_DIR / "comp_noise_counts.npz"
    np.savez(npz_path, **comp_scores_npz)
    print(f"\nSaved NPZ  → {npz_path}")

    # Unified JSON
    result = {
        "version":             1,
        "noise_sigmas":        NOISE_SIGMAS,
        "complexity_auc": comp_auc_list,
        "byol_auc":       byol_auc_list,
        "ell_auc":        ell_auc_list,
        "byol_run":            args.byol_run,
        "data_seed":           args.data_seed,
        "rng_seed":            args.seed,
        "test_idx_sha256":     test_idx_sha256,
        "pca_components":      int(pca.n_components_),
        "n_labeled":           int(len(lab_idx)),
        "n_test":              int(len(test_idx)),
    }
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved JSON → {out_json}")

    # Backwards-compatible: comp_noise_robustness.json (read by notebook cell 40)
    with open(BASELINE_OUT_DIR / "comp_noise_robustness.json", "w") as fh:
        json.dump({
            "version":         4,
            "noise_sigmas":    NOISE_SIGMAS,
            "comp_auc":        comp_auc_list,
            "method":          "blob_doh_complexity",
            "data_seed":       args.data_seed,
            "rng_seed":        args.seed,
            "test_idx_sha256": test_idx_sha256,
        }, fh, indent=2)

    # Backwards-compatible: byol_noise_robustness.json (read by notebook cell 40)
    with open(BYOL_OUT_DIR / "byol_noise_robustness.json", "w") as fh:
        json.dump({
            "run":            args.byol_run,
            "data_seed":      args.data_seed,
            "rng_seed":       args.seed,
            "noise_sigmas":   NOISE_SIGMAS,
            "byol_auc":       byol_auc_list,
            "n_labeled":      int(len(lab_idx)),
            "n_test":         int(len(test_idx)),
            "pca_components": int(pca.n_components_),
        }, fh, indent=2)

    print("Saved backwards-compatible JSONs.")

    # ── Summary ───────────────────────────────────────────────────────────────
    col_w = 18
    headers = ["Complexity", "BYOL GP", "Ellipses GP"]
    print(f"\n{'σ':<6}", end="")
    for h in headers:
        print(f"{h:>{col_w}}", end="")
    print()
    print("─" * (6 + col_w * len(headers)))
    for i, sigma in enumerate(NOISE_SIGMAS):
        print(f"{sigma:<6.2f}", end="")
        for vals in [comp_auc_list, byol_auc_list, ell_auc_list]:
            print(f"{vals[i]:>{col_w}.4f}", end="")
        print()


if __name__ == "__main__":
    main()
