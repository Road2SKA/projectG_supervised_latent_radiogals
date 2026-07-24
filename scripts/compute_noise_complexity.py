"""Compute connected-component complexity AUC at various noise levels.

For σ=0: uses catalogue featurecount from CSV (matching existing 0.730 reference).
For σ>0: adds noise to test images and counts connected components above a
         threshold_nsigma × per-image noise RMS threshold.

Output: outputs/anomaly_baselines/noise_robustness/comp_noise_robustness.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from suplat.utils.class_weights import LABEL_COLS, TIERS

# ── Hardcoded paths ───────────────────────────────────────────────────────────
IMAGES_PATH = ROOT / "data/preprocessed/lotss/images_filtered.npy"
LABELS_PATH = ROOT / "data/preprocessed/lotss/labels_filtered.npy"
CSV_PATH    = ROOT / "data/metadata/lotss_classifications_horton_et_al_2025_filtered.csv"
OUTPUT_PATH = ROOT / "outputs/anomaly_baselines/noise_robustness/comp_noise_robustness.json"

NOISE_SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def build_human_labels(labels_raw):
    """Convert raw bool label matrix to integer human-label tier scores."""
    ldf = pd.DataFrame(labels_raw.astype(bool), columns=LABEL_COLS)
    hl = np.ones(len(labels_raw), dtype=int)
    for sv, cols in reversed(TIERS):
        hl[ldf[cols].any(axis=1).values] = sv
    return hl


def add_noise(images, sigma_rel, rng):
    """Add Gaussian noise scaled by per-image std."""
    if sigma_rel == 0.0:
        return images.astype(np.float32)
    per_std = images.astype(np.float32).std(axis=(-2, -1), keepdims=True).clip(1e-6)
    return (images.astype(np.float32)
            + rng.normal(0, sigma_rel, images.shape).astype(np.float32) * per_std)


def count_components(image_f32, threshold_nsigma=3.0, min_pixels=4):
    """Count connected components above threshold_nsigma × background RMS."""
    med = np.median(image_f32)
    below = image_f32[image_f32 < med]
    noise_rms = below.std() if len(below) > 0 else 1e-6
    if noise_rms == 0.0:
        noise_rms = 1e-6
    binary = image_f32 > (threshold_nsigma * noise_rms)
    labeled, n = ndi.label(binary)
    if n == 0:
        return 0
    sizes = np.bincount(labeled.ravel())[1:]
    return int((sizes >= min_pixels).sum())


def recall_auc(scores, true_hl, pos_thresh=3):
    """Area under the cumulative recall curve (higher = better ranker)."""
    order  = np.argsort(scores)[::-1]
    binary = (true_hl[order] >= pos_thresh).astype(int)
    cum    = np.cumsum(binary)
    n_pos  = int(binary.sum())
    if n_pos == 0:
        return 0.0
    x = np.arange(1, len(cum) + 1)
    return float(np.trapezoid(cum, x) / (len(cum) * n_pos))


def main():
    parser = argparse.ArgumentParser(description="Noise complexity AUC sweep")
    parser.add_argument("--data_seed",        type=int,   default=42)
    parser.add_argument("--threshold_nsigma", type=float, default=3.0)
    parser.add_argument("--min_pixels",       type=int,   default=4)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--force",            action="store_true")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not args.force:
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Use --force to overwrite.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    splits_dir = ROOT / "outputs/data_splits" / str(args.data_seed)
    test_idx_path = splits_dir / "test_idx.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(f"test_idx.npy not found at {test_idx_path}")

    test_idx = np.load(test_idx_path)
    print(f"Test set size: {len(test_idx)}")

    labels_all = np.load(LABELS_PATH)
    test_lraw  = labels_all[test_idx]
    test_hl    = build_human_labels(test_lraw)
    print(f"Positive test sources (tier ≥ 3): {(test_hl >= 3).sum()}")

    images_all  = np.load(IMAGES_PATH, mmap_mode="r")
    test_images = images_all[test_idx]   # (N_test, 89, 89) uint8

    df = pd.read_csv(CSV_PATH)

    rng = np.random.default_rng(args.seed)

    comp_auc = []

    for sigma in NOISE_SIGMAS:
        print(f"σ={sigma} ...", end=" ", flush=True)

        if sigma == 0.0:
            # Use catalogue featurecount — matches the existing 0.730 reference
            fc = df["featurecount"].values[test_idx].astype(float)
            auc = recall_auc(fc, test_hl)
            print(f"AUC={auc:.4f}  (featurecount from CSV)")
        else:
            noisy = add_noise(test_images, sigma, rng)
            counts = np.array([
                count_components(noisy[i], args.threshold_nsigma, args.min_pixels)
                for i in range(len(noisy))
            ])
            auc = recall_auc(counts.astype(float), test_hl)
            print(f"AUC={auc:.4f}  "
                  f"(mean components={counts.mean():.2f}, "
                  f"zero-count={int((counts == 0).sum())})")

        comp_auc.append(auc)

    result = {
        "noise_sigmas":     NOISE_SIGMAS,
        "comp_auc":         comp_auc,
        "threshold_nsigma": args.threshold_nsigma,
        "min_pixels":       args.min_pixels,
        "data_seed":        args.data_seed,
        "rng_seed":         args.seed,
    }

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
