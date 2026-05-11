"""
preprocess_mirabest.py

Converts MiraBest batches into 89x89 .npy crops for the SUPLAT BYOL encoder.

Your downloaded version (Zenodo 5588282, MiraBest_F format) already stores
binary labels: 0=FRI, 1=FRII. No filtering or remapping is needed — we load
the batches directly without going through MBFRConfident.

Total expected: 792 train + 88 test = 880 images (FRI=351, FRII=441 across
all batches, including test).

Expected input layout:
    data/raw/mirabest/
        batches/
            data_batch_1 ... data_batch_8   ← 88 images each
            test_batch                      ← 88 images
            batches.meta

Pipeline per image:
    1. Load batch via pickle
    2. Squeeze channel dim if present: (88,150,150,1) or (88,150,150) → (150,150)
    3. Resize 150x150 → 89x89 (bicubic, anti-aliased)
    4. Cast to float32
    5. Save as <split>_<index:04d>.npy
    6. Append row to labels.csv

Usage:
    python preprocess_mirabest.py                      # use all defaults
    python preprocess_mirabest.py \
        --root       data/raw/mirabest \
        --output_dir data/preprocessed/mirabest \
        --labels_csv data/metadata/mirabest_labels.csv
"""

import os
import pickle
import argparse

import numpy as np
import pandas as pd
from skimage.transform import resize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_SHAPE = (89, 89)

# Batch files in load order — 8 train batches + 1 test batch
TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 9)]
TEST_BATCHES  = ["test_batch"]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_batch(batch_dir, filename):
    """
    Load one MiraBest batch file and return images + labels.

    Parameters
    ----------
    batch_dir : str   path to the batches/ subdirectory
    filename  : str   e.g. "data_batch_1"

    Returns
    -------
    images : np.ndarray, shape (N, 150, 150)
    labels : list of int  (0=FRI, 1=FRII)
    """
    fpath = os.path.join(batch_dir, filename)
    with open(fpath, 'rb') as f:
        # encoding='latin1' required for Python-2-pickled files
        batch = pickle.load(f, encoding='latin1')

    images = np.array(batch['data'])   # may be a list in some versions
    labels = list(batch['labels'])

    # Normalise shape to (N, 150, 150) — squeeze channel dim if present
    if images.ndim == 4:
        images = images.squeeze(-1)   # (N, 150, 150, 1) → (N, 150, 150)

    return images, labels


def preprocess_image(img):
    """
    Resize a single 150x150 image to 89x89 and cast to float32.

    MiraBest images are already per-image normalised in the batch files,
    so no contrast-stretch is applied.

    Parameters
    ----------
    img : np.ndarray, shape (150, 150)

    Returns
    -------
    np.ndarray, shape (89, 89), dtype float32
    """
    resized = resize(img, OUTPUT_SHAPE,
                     order=3,               # bicubic interpolation
                     anti_aliasing=True,    # prevents aliasing when downsampling
                     preserve_range=True)   # keep original value range
    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(batch_dir, batch_files, split_name, output_dir, rows):
    """
    Load all batches for one split, preprocess each image, save .npy files.

    Parameters
    ----------
    batch_dir   : str        path to batches/ subdirectory
    batch_files : list[str]  filenames to load for this split
    split_name  : str        "train" or "test"
    output_dir  : str        where to write .npy files
    rows        : list       mutated in place — one dict per saved image
    """
    idx = 0

    for filename in batch_files:
        fpath = os.path.join(batch_dir, filename)
        if not os.path.exists(fpath):
            print(f"  WARNING: {filename} not found, skipping")
            continue

        images, labels = load_batch(batch_dir, filename)
        n_kept = 0

        for img, label in zip(images, labels):
            processed = preprocess_image(img)

            fname = f"{split_name}_{idx:04d}.npy"
            np.save(os.path.join(output_dir, fname), processed)

            rows.append({
                "filename":  fname,
                "split":     split_name,
                "label":     int(label),
                "label_str": "FRI" if label == 0 else "FRII",
                "source":    filename,
            })

            idx    += 1
            n_kept += 1

        print(f"    {filename}: {n_kept} images  "
              f"(FRI={labels.count(0)}, FRII={labels.count(1)})")

    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(root, output_dir, labels_csv):
    batch_dir = os.path.join(root, "batches")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    rows = []

    print(f"Processing {len(TRAIN_BATCHES)} training batches...")
    n_train = process_split(batch_dir, TRAIN_BATCHES, "train", output_dir, rows)

    print(f"\nProcessing test batch...")
    n_test = process_split(batch_dir, TEST_BATCHES, "test", output_dir, rows)

    df = pd.DataFrame(rows)
    df.to_csv(labels_csv, index=False)

    fri_n  = (df["label"] == 0).sum()
    frii_n = (df["label"] == 1).sum()
    print(f"\nDone.")
    print(f"  Train : {n_train}  |  Test : {n_test}  |  Total : {len(rows)}")
    print(f"  FRI={fri_n}, FRII={frii_n}")
    print(f"  Labels saved → {labels_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MiraBest → 89x89 .npy"
    )
    parser.add_argument("--root",
                        default="data/raw/mirabest",
                        help="Root dir containing batches/ subdir "
                             "(default: data/raw/mirabest)")
    parser.add_argument("--output_dir",
                        default="data/preprocessed/mirabest",
                        help="Where to save .npy crops "
                             "(default: data/preprocessed/mirabest)")
    parser.add_argument("--labels_csv",
                        default="data/metadata/mirabest_labels.csv",
                        help="Path for output labels CSV "
                             "(default: data/metadata/mirabest_labels.csv)")
    args = parser.parse_args()

    main(args.root, args.output_dir, args.labels_csv)