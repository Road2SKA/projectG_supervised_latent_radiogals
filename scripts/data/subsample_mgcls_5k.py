#!/usr/bin/env python3
"""
subsample_mgcls_5k.py

Creates a random 5,000-sample subset of the MGCLS-20k processed dataset:
  1. Samples 5,000 rows at random from mgcls_crops.csv
  2. Writes the subset to mgcls_5k_crops.csv
  3. Creates data/preprocessed/mgcls_5k/ and populates it with symlinks
     pointing to the corresponding files in data/preprocessed/mgcls_20k/

Safe to re-run: existing symlinks and the output CSV are skipped/overwritten.

Usage:
    python scripts/data/subsample_mgcls_5k.py
    python scripts/data/subsample_mgcls_5k.py --seed 123 --n 5000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main(meta_csv: str, output_csv: str, source_dir: str, output_dir: str,
         n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    df = pd.read_csv(meta_csv)
    if len(df) < n:
        raise ValueError(f"Only {len(df)} crops available; cannot sample {n}.")

    idx = rng.choice(len(df), size=n, replace=False)
    subset = df.iloc[idx].reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    subset.to_csv(output_csv, index=False)
    print(f"Wrote {len(subset)} rows -> {output_csv}")

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_created = 0
    for fname in subset["filename"]:
        dst = output_dir / fname
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(source_dir / fname, dst)
        n_created += 1

    n_total = len(subset)
    print(f"Created {n_created} new symlinks ({n_total - n_created} already existed) in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random 5k subsample of MGCLS-20k with symlinks"
    )
    parser.add_argument("--meta_csv",   default="data/metadata/mgcls_crops.csv",
                        help="Source metadata CSV (default: data/metadata/mgcls_crops.csv)")
    parser.add_argument("--output_csv", default="data/metadata/mgcls_5k_crops.csv",
                        help="Output subset CSV (default: data/metadata/mgcls_5k_crops.csv)")
    parser.add_argument("--source_dir", default="data/preprocessed/mgcls_20k",
                        help="Directory with full MGCLS-20k .npy files")
    parser.add_argument("--output_dir", default="data/preprocessed/mgcls_5k",
                        help="Directory where symlinks are created")
    parser.add_argument("--n",    type=int, default=5000,
                        help="Number of samples to draw (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    main(args.meta_csv, args.output_csv, args.source_dir, args.output_dir,
         args.n, args.seed)
