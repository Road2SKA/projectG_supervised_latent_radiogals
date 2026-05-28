"""
preprocess_first.py

Converts the RadioGalaxyDataset HDF5 file into 89x89 .npy crops for
the SUPLAT BYOL encoder.

Dataset facts (Griese et al. 2022, Zenodo 7120632):
  - 2158 images total, 300x300 pixels, from VLA FIRST survey
  - 4 classes: 0=FRI, 1=FRII, 2=Compact, 3=Bent
  - Pre-defined splits stored in the HDF5 file:
      Split_literature: "train" | "valid" | "test"
  - Class counts: FRI=495, FRII=924, Compact=391, Bent=348
    Test/valid sets: 50 samples per class each (balanced)

HDF5 file structure (galaxy_data_h5.h5):
    /data          : float32 array, shape (2158, 300, 300)
    /Label_literature : float64 array, values 0.0–3.0
    /Split_literature : bytes array, values b"train"|b"valid"|b"test"

Pipeline per image:
    1. Load all images + labels + splits from HDF5
    2. Resize 300x300 → 89x89 (bicubic, anti-aliased)
    3. Cast to float32
    4. Save as <split>_<index:04d>.npy
    5. Append row to labels.csv

Download:
    Clone https://github.com/floriangriese/RadioGalaxyDataset and copy
    galaxy_data_h5.h5 to data/raw/first/, or download
    galaxy_data.zip from Zenodo 7120632 and unzip it there.

Usage:
    python preprocess_first.py              # use defaults
    python preprocess_first.py \
        --h5_path    data/raw/first/galaxy_data_h5.h5 \
        --output_dir data/preprocessed/first \
        --labels_csv data/metadata/first_labels.csv
"""

import os
import argparse
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from skimage.transform import resize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_SHAPE = (89, 89)

# Integer label → string name
LABEL_NAMES = {0: "FRI", 1: "FRII", 2: "Compact", 3: "Bent"}

# The HDF5 stores splits as byte strings; map to plain strings
SPLIT_DECODE = {b"train": "train", b"valid": "valid", b"test": "test"}

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def preprocess_image(img):
    """
    Resize a single 300x300 image to 89x89 and cast to float32.

    RadioGalaxyDataset images are already normalised in the HDF5 file,
    so no contrast-stretch is applied here.

    Parameters
    ----------
    img : np.ndarray, shape (300, 300)

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
# Main
# ---------------------------------------------------------------------------

def main(h5_path, output_dir, labels_csv):
    if not os.path.exists(h5_path):
        print(f"HDF5 file not found: {h5_path}")
        print("Download galaxy_data_h5.h5 from Zenodo 7120632 or clone the "
              "GitHub repo and copy the file.")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    # --- Load everything from HDF5 in one pass ---
    # The file stores each sample as a separate group: data_0, data_1, ...
    # Each group contains: Img (300x300 uint8), Label_literature, Split_literature
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, 'r') as hf:
        n = len(hf)
        keys = [f"data_{i}" for i in range(n)]
        images = [hf[k]['Img'][()] for k in keys]
        labels = [int(hf[k]['Label_literature'][()]) for k in keys]
        splits = [hf[k]['Split_literature'][()] for k in keys]

    print(f"  Loaded {len(images)} images.")

    # --- Process each image ---
    rows      = []
    split_idx = Counter()   # per-split counter for unique filenames

    for img, label, split_bytes in zip(images, labels, splits):
        split = SPLIT_DECODE.get(split_bytes, split_bytes.decode())

        processed = preprocess_image(img)

        idx   = split_idx[split]
        fname = f"{split}_{idx:04d}.npy"
        np.save(os.path.join(output_dir, fname), processed)

        rows.append({
            "filename":  fname,
            "split":     split,
            "label":     int(label),
            "label_str": LABEL_NAMES.get(int(label), str(label)),
        })

        split_idx[split] += 1

    # --- Save labels CSV ---
    df = pd.DataFrame(rows)
    df.to_csv(labels_csv, index=False)

    # --- Summary ---
    print(f"\nDone. {len(rows)} images saved to {output_dir}")
    for split in ["train", "valid", "test"]:
        sdf = df[df["split"] == split]
        counts = sdf.groupby("label_str").size().to_dict()
        print(f"  {split:5s}: {len(sdf):4d} images  {counts}")
    print(f"  Labels saved → {labels_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess RadioGalaxyDataset HDF5 → 89x89 .npy"
    )
    parser.add_argument("--h5_path",
                        default="data/raw/first/galaxy_data_h5.h5",
                        help="Path to galaxy_data_h5.h5 "
                             "(default: data/raw/first/galaxy_data_h5.h5)")
    parser.add_argument("--output_dir",
                        default="data/preprocessed/first",
                        help="Where to save .npy crops "
                             "(default: data/preprocessed/first)")
    parser.add_argument("--labels_csv",
                        default="data/metadata/first_labels.csv",
                        help="Path for output labels CSV "
                             "(default: data/metadata/first_labels.csv)")
    args = parser.parse_args()

    main(args.h5_path, args.output_dir, args.labels_csv)