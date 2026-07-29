"""Parameter sweep for blob_log, blob_dog, and blob_doh on the clean test set.

For each (method, params) combination the score per test source is the blob
count detected on the [0, 1]-normalised image.  AUC is the area under the
cumulative recall curve (higher = better).

Outputs
-------
  outputs/anomaly_baselines/blob_sweep/blob_sweep_scores.npz   (cache)
  outputs/anomaly_baselines/blob_sweep/blob_sweep_results.json
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_doh, blob_log

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from suplat.utils.class_weights import LABEL_COLS, TIERS

IMAGES_PATH  = ROOT / "data/preprocessed/lotss/images_filtered.npy"
LABELS_PATH  = ROOT / "data/preprocessed/lotss/labels_filtered.npy"
OUTPUT_DIR   = ROOT / "outputs/anomaly_baselines/blob_sweep"
OUTPUT_PATH  = OUTPUT_DIR / "blob_sweep_results.json"
SCORES_PATH  = OUTPUT_DIR / "blob_sweep_scores.npz"

# ── Parameter grids ────────────────────────────────────────────────────────────

LOG_GRID = list(itertools.product(
    [0.5, 1.0, 2.0],   # min_sigma
    [5,   10,  15 ],   # max_sigma
    [0.05, 0.10, 0.15, 0.20, 0.30],  # threshold
))

DOG_GRID = list(itertools.product(
    [0.5, 1.0, 2.0],   # min_sigma
    [5,   10,  15 ],   # max_sigma
    [1.6, 2.0],         # sigma_ratio
    [0.01, 0.05, 0.10, 0.15, 0.20],  # threshold
))

DOH_GRID = list(itertools.product(
    [1.0, 2.0, 3.0],   # min_sigma
    [10,  15,  20 ],   # max_sigma
    [5,   10  ],        # num_sigma
    [0.001, 0.005, 0.01, 0.05],  # threshold
))


def _key_log(min_s, max_s, thr):
    return f"log_ms{min_s}_Ms{max_s}_t{thr}"

def _key_dog(min_s, max_s, sr, thr):
    return f"dog_ms{min_s}_Ms{max_s}_sr{sr}_t{thr}"

def _key_doh(min_s, max_s, ns, thr):
    return f"doh_ms{min_s}_Ms{max_s}_ns{ns}_t{thr}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_human_labels(labels_raw):
    ldf = pd.DataFrame(labels_raw.astype(bool), columns=LABEL_COLS)
    hl = np.ones(len(labels_raw), dtype=int)
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


def normalise(image):
    """Normalise image to [0, 1]; return zero array if flat."""
    img_min, img_max = image.min(), image.max()
    if img_max <= img_min:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - img_min) / (img_max - img_min)).astype(np.float32)


def count_blobs(fn, images):
    """Apply fn to each normalised image, return blob count array."""
    out = np.zeros(len(images), dtype=np.float64)
    for i, img in enumerate(images):
        try:
            out[i] = float(len(fn(normalise(img))))
        except Exception:
            out[i] = 0.0
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blob detector parameter sweep")
    parser.add_argument("--data_seed",   type=int, default=42)
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite output JSON.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached score arrays before running.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not args.force:
        print(f"Output exists: {OUTPUT_PATH}\nUse --force to overwrite.")
        return

    if args.clear_cache and SCORES_PATH.exists():
        SCORES_PATH.unlink()
        print("Cleared score cache (--clear-cache).")

    # ── Load test split ───────────────────────────────────────────────────────
    splits_dir = ROOT / "outputs/data_splits" / str(args.data_seed)
    test_idx   = np.load(splits_dir / "test_idx.npy")
    labels_all = np.load(LABELS_PATH)
    n_total    = labels_all.shape[0]

    lab_idx   = np.load(splits_dir / "labelled_train_idx.npy")
    unlab_idx = np.load(splits_dir / "unlabelled_train_idx.npy")
    _n_train  = len(lab_idx) + len(unlab_idx)
    if len(test_idx) < (n_total - _n_train):
        train_idx = np.concatenate([lab_idx, unlab_idx])
        test_idx  = np.setdiff1d(np.arange(n_total), train_idx)
        print(f"Derived full test split: {len(test_idx)} entries")
    else:
        print(f"Test set size: {len(test_idx)}")

    test_hl     = build_human_labels(labels_all[test_idx])
    images_all  = np.load(IMAGES_PATH, mmap_mode="r")
    test_images = images_all[test_idx].astype(np.float32)
    print(f"Positive test sources (tier ≥ 3): {(test_hl >= 3).sum()}\n")

    # ── Load score cache ──────────────────────────────────────────────────────
    scores_store = {}
    if SCORES_PATH.exists():
        cached = np.load(SCORES_PATH)
        scores_store.update({k: cached[k] for k in cached.files})
        print(f"Loaded {len(scores_store)} cached configs.\n")

    rows = []

    # ── blob_log sweep ────────────────────────────────────────────────────────
    print(f"blob_log: {len(LOG_GRID)} configurations")
    for min_s, max_s, thr in LOG_GRID:
        key = _key_log(min_s, max_s, thr)
        if key not in scores_store:
            scores_store[key] = count_blobs(
                lambda img, a=min_s, b=max_s, t=thr:
                    blob_log(img, min_sigma=a, max_sigma=b, threshold=t),
                test_images,
            )
        auc = recall_auc(scores_store[key], test_hl)
        zeros = int((scores_store[key] == 0).sum())
        mean  = float(scores_store[key].mean())
        rows.append({"method": "blob_log", "key": key, "auc": auc,
                     "mean": mean, "zeros": zeros,
                     "params": {"min_sigma": min_s, "max_sigma": max_s,
                                "threshold": thr}})
        print(f"  {key:<38} AUC={auc:.4f}  mean={mean:.2f}  zeros={zeros}")

    # ── blob_dog sweep ────────────────────────────────────────────────────────
    print(f"\nblob_dog: {len(DOG_GRID)} configurations")
    for min_s, max_s, sr, thr in DOG_GRID:
        key = _key_dog(min_s, max_s, sr, thr)
        if key not in scores_store:
            scores_store[key] = count_blobs(
                lambda img, a=min_s, b=max_s, r=sr, t=thr:
                    blob_dog(img, min_sigma=a, max_sigma=b,
                             sigma_ratio=r, threshold=t),
                test_images,
            )
        auc = recall_auc(scores_store[key], test_hl)
        zeros = int((scores_store[key] == 0).sum())
        mean  = float(scores_store[key].mean())
        rows.append({"method": "blob_dog", "key": key, "auc": auc,
                     "mean": mean, "zeros": zeros,
                     "params": {"min_sigma": min_s, "max_sigma": max_s,
                                "sigma_ratio": sr, "threshold": thr}})
        print(f"  {key:<38} AUC={auc:.4f}  mean={mean:.2f}  zeros={zeros}")

    # ── blob_doh sweep ────────────────────────────────────────────────────────
    print(f"\nblob_doh: {len(DOH_GRID)} configurations")
    for min_s, max_s, ns, thr in DOH_GRID:
        key = _key_doh(min_s, max_s, ns, thr)
        if key not in scores_store:
            scores_store[key] = count_blobs(
                lambda img, a=min_s, b=max_s, n=ns, t=thr:
                    blob_doh(img, min_sigma=a, max_sigma=b,
                             num_sigma=n, threshold=t),
                test_images,
            )
        auc = recall_auc(scores_store[key], test_hl)
        zeros = int((scores_store[key] == 0).sum())
        mean  = float(scores_store[key].mean())
        rows.append({"method": "blob_doh", "key": key, "auc": auc,
                     "mean": mean, "zeros": zeros,
                     "params": {"min_sigma": min_s, "max_sigma": max_s,
                                "num_sigma": ns, "threshold": thr}})
        print(f"  {key:<38} AUC={auc:.4f}  mean={mean:.2f}  zeros={zeros}")

    # ── Save score cache ──────────────────────────────────────────────────────
    np.savez(SCORES_PATH, **scores_store)
    print(f"\nScores cached → {SCORES_PATH}")

    # ── Print ranked table ────────────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda r: r["auc"], reverse=True)

    print(f"\n{'Rank':<5} {'Key':<40} {'Method':<10} {'AUC':>6} {'Mean':>7} {'Zeros':>6}")
    print("-" * 78)
    for rank, r in enumerate(rows_sorted[:30], 1):
        print(f"{rank:<5} {r['key']:<40} {r['method']:<10} "
              f"{r['auc']:>6.4f} {r['mean']:>7.2f} {r['zeros']:>6}")
    if len(rows_sorted) > 30:
        print(f"  … ({len(rows_sorted) - 30} more configs not shown)")
    print("-" * 78)

    # Per-method best
    print("\nBest per method:")
    for method in ("blob_log", "blob_dog", "blob_doh"):
        best = max((r for r in rows if r["method"] == method),
                   key=lambda r: r["auc"])
        print(f"  {method:<10}  AUC={best['auc']:.4f}  {best['key']}")
        print(f"             params={best['params']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "data_seed": args.data_seed,
        "n_configs": len(rows),
        "results":   rows_sorted,
    }
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
