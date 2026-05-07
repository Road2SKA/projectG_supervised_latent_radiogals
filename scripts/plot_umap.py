#!/usr/bin/env python3
"""
plot_umap.py

UMAP visualisation of BYOL encoder embeddings for eval datasets.

Two plots are produced:
  1. umap_by_dataset.png  — each point coloured by source dataset
  2. umap_by_label.png    — each point coloured by integer label
                            (unlabelled points, label == -1, shown in grey)

UMAP 2D coordinates are also saved as umap_coords.npy for later reuse.

Usage:
    python scripts/plot_umap.py \\
        --datasets mgcls_5k mirabest radio_galaxy_dataset \\
        --embeddings_dir outputs/embeddings/run_id \\
        --output outputs/figures/run_id
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

SEED = 42

# Colorblind-friendly palette for datasets
DATASET_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def load_embeddings(
    embeddings_dir: Path, datasets: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_emb, all_lbl, all_ds = [], [], []
    for i, ds in enumerate(datasets):
        emb_path = embeddings_dir / f"{ds}_embeddings.npy"
        lbl_path = embeddings_dir / f"{ds}_labels.npy"
        if not emb_path.exists():
            print(f"  WARNING: {emb_path} not found — skipping {ds}")
            continue
        emb = np.load(emb_path)
        lbl = np.load(lbl_path)
        all_emb.append(emb)
        all_lbl.append(lbl)
        all_ds.append(np.full(len(emb), i, dtype=int))
    if not all_emb:
        raise RuntimeError(f"No embedding files found in {embeddings_dir}. Run extract_embeddings.py first.")
    return np.vstack(all_emb), np.concatenate(all_lbl), np.concatenate(all_ds)


def fit_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=SEED,
    )
    print(f"  Fitting UMAP on {len(embeddings)} samples...")
    return reducer.fit_transform(embeddings)


def plot_by_dataset(
    xy: np.ndarray,
    ds_idx: np.ndarray,
    datasets: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ds_name in enumerate(datasets):
        mask = ds_idx == i
        if not mask.any():
            continue
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=color, s=6, alpha=0.5, label=f"{ds_name} ({mask.sum()})",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP — coloured by dataset")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_label(
    xy: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    unique_labels = sorted(set(labels.tolist()))
    labelled = [l for l in unique_labels if l != -1]
    cmap = plt.cm.get_cmap("tab10", max(len(labelled), 1))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Unlabelled background first
    mask_unlabelled = labels == -1
    if mask_unlabelled.any():
        ax.scatter(
            xy[mask_unlabelled, 0], xy[mask_unlabelled, 1],
            c="lightgrey", s=4, alpha=0.3,
            label=f"unlabelled ({mask_unlabelled.sum()})",
        )

    for j, lbl in enumerate(labelled):
        mask = labels == lbl
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=[cmap(j)], s=10, alpha=0.7,
            label=f"label {lbl} ({mask.sum()})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP — coloured by label")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="UMAP visualisation of BYOL eval embeddings"
    )
    ap.add_argument("--datasets",       nargs="+", required=True,
                    help="Dataset names (must have matching *_embeddings.npy files)")
    ap.add_argument("--embeddings_dir", type=Path,  required=True,
                    help="Directory containing *_embeddings.npy / *_labels.npy")
    ap.add_argument("--output",         type=Path,  required=True,
                    help="Directory for output PNG files")
    ap.add_argument("--n_neighbors",    type=int,   default=15)
    ap.add_argument("--min_dist",       type=float, default=0.1)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    embeddings, labels, ds_idx = load_embeddings(args.embeddings_dir, args.datasets)
    print(f"Total: {len(embeddings)} samples from {len(args.datasets)} datasets")

    xy = fit_umap(embeddings, args.n_neighbors, args.min_dist)

    coords_path = args.output / "umap_coords.npy"
    np.save(coords_path, xy)
    print(f"Saved UMAP coordinates: {coords_path}")

    plot_by_dataset(xy, ds_idx, args.datasets, args.output / "umap_by_dataset.png")
    plot_by_label(xy, labels, args.output / "umap_by_label.png")


if __name__ == "__main__":
    main()
