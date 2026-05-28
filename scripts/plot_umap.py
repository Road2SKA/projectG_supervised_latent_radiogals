#!/usr/bin/env python3
"""
plot_umap.py

Two modes:

  1. --run-dir MODE: Reproduce the UMAP plots that train_byol.py would have
     generated for a completed run directory with saved projections/indices.
     Loads full labels (all 20 columns) from the original dataset.

     Usage:
         python scripts/plot_umap.py \\
             --run-dir outputs/run_sweep_f0.1_sw0.5_20260518_1834 \\
             [--data-dir data/preprocessed/lotss] \\
             [--no-outlier-plot]

  2. LEGACY MODE: UMAP from *_embeddings.npy / *_labels.npy files.

     Usage:
         python scripts/plot_umap.py \\
             --datasets mgcls_5k mirabest first \\
             --embeddings_dir outputs/embeddings/run_id \\
             --output outputs/figures/run_id
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# Default data directory (same default as train_byol.py)
DEFAULT_DATA_DIR = Path('/users/mbredber/p3_SUPLAT/data/preprocessed/lotss')

LABEL_RANGES = {
    'full':        (0, 20),
    'all':         (0, 20),
    'classical':   (0, 2),
    'initial':     (0, 5),
    'morphology':  (5, 16),
    'environment': (16, 20),
    'derived':     (19, 24),
}

CLASS_NAMES = {
    'initial':     ['FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles'],
    'morphology':  ['C-curvature', 'S-curvature', 'Misalignment', 'Wings', 'X-shaped',
                    'Straight jets', 'Multiple hotspots', 'Continuous jets', 'Banding',
                    'One-sided', 'Restarted'],
    'environment': ['Cluster', 'Merger', 'Diffuse emission', 'Unknown'],
    'derived':     ['Compact+hybrids', 'Hybrid FRI/FRII', 'Curved FRIs',
                    'Curved FRIIs', 'Straight+multi hotspots'],
}

# Colorblind-friendly palette for legacy mode datasets
DATASET_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


# =============================================================================
# RUN-DIR MODE
# =============================================================================

def run_dir_main(args):
    """Reproduce train_byol.py UMAP plots from a completed run directory."""
    from suplat.utils.plotting import fit_umap, plot_umap_single, plot_umap_outliers

    run_dir  = args.run_dir
    data_dir = run_dir / 'data'
    umap_dir = run_dir / 'figures' / 'umap'
    umap_dir.mkdir(parents=True, exist_ok=True)

    # ── Load projections ──────────────────────────────────────────────────────
    print("Loading projections...")
    train_proj = np.load(data_dir / 'train_projections.npy')
    val_proj   = np.load(data_dir / 'val_projections.npy')
    test_proj  = np.load(data_dir / 'test_projections.npy')
    print(f"  Train: {train_proj.shape}  Val: {val_proj.shape}  Test: {test_proj.shape}")

    # ── Load split indices ────────────────────────────────────────────────────
    labelled_train_idx = np.load(data_dir / 'labelled_train_idx.npy')
    val_idx            = np.load(data_dir / 'val_idx.npy')
    test_idx           = np.load(data_dir / 'test_idx.npy')

    # ── Load full 20-column labels from the original dataset ─────────────────
    labels_path = args.data_dir / 'labels_filtered.npy'
    if not labels_path.exists():
        print(f"ERROR: labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading full labels from {labels_path}...")
    labels_full = np.load(labels_path)
    print(f"  labels_full: {labels_full.shape}")

    # Per-split full labels aligned with the projections
    _lf_train = labels_full[labelled_train_idx][:len(train_proj)]
    _lf_val   = labels_full[val_idx][:len(val_proj)]
    _lf_test  = labels_full[test_idx][:len(test_proj)]

    _n_tr = len(train_proj)
    _n_va = len(val_proj)
    _n_te = len(test_proj)

    # ── "all" UMAP: unlabelled train (if any) + labelled train + val + test ──
    print("\nGenerating 'all' UMAP (train + val + test)...")
    _unlab_proj_path = data_dir / 'unlabelled_train_projections.npy'
    if _unlab_proj_path.exists():
        _unlab_proj = np.load(_unlab_proj_path)
        _n_ul = len(_unlab_proj)
        _lf_unlab = np.zeros((_n_ul, _lf_train.shape[1]), dtype=_lf_train.dtype)
        _all_proj = np.concatenate([_unlab_proj, train_proj, val_proj, test_proj])
        _all_lf   = np.concatenate([_lf_unlab, _lf_train, _lf_val, _lf_test])
        print(f"  Unlabelled train: {_unlab_proj.shape}")
    else:
        _n_ul = 0
        _all_proj = np.concatenate([train_proj, val_proj, test_proj])
        _all_lf   = np.concatenate([_lf_train, _lf_val, _lf_test])

    _n_all = _n_ul + _n_tr + _n_va + _n_te
    _mask_ul = np.zeros(_n_all, dtype=bool); _mask_ul[:_n_ul] = True
    _mask_tr = np.zeros(_n_all, dtype=bool); _mask_tr[_n_ul:_n_ul + _n_tr] = True
    _mask_va = np.zeros(_n_all, dtype=bool); _mask_va[_n_ul + _n_tr:_n_ul + _n_tr + _n_va] = True
    _mask_te = np.zeros(_n_all, dtype=bool); _mask_te[_n_ul + _n_tr + _n_va:] = True

    _, _all_2d = fit_umap(_all_proj, args.umap_n_neighbors, args.umap_min_dist, SEED)
    np.save(data_dir / 'umap_all_coords.npy', _all_2d)
    print(f"  Saved UMAP coordinates: {data_dir / 'umap_all_coords.npy'}")

    _split_masks_all = {}
    if _n_ul > 0:
        _split_masks_all['Unlabelled train'] = _mask_ul
    _split_masks_all.update({'Labelled train': _mask_tr, 'Val': _mask_va, 'Test': _mask_te})
    for _col in ('initial', 'morphology', 'train_labelled'):
        plot_umap_single(
            _all_2d, _all_lf, _col, CLASS_NAMES, LABEL_RANGES,
            title=f'All — {_col}',
            save_path=umap_dir / f'umap_all_{_col}.png',
            split_masks=_split_masks_all,
        )

    # ── "test" UMAP: test only ────────────────────────────────────────────────
    print("\nGenerating 'test' UMAP...")
    _, _test_2d = fit_umap(test_proj, args.umap_n_neighbors, args.umap_min_dist, SEED)
    np.save(data_dir / 'umap_test_coords.npy', _test_2d)

    _split_masks_test = {'Test': np.ones(_n_te, dtype=bool)}
    for _col in ('initial', 'morphology'):
        plot_umap_single(
            _test_2d, _lf_test, _col, CLASS_NAMES, LABEL_RANGES,
            title=f'Test — {_col}',
            save_path=umap_dir / f'umap_test_{_col}.png',
            split_masks=_split_masks_test,
        )

    # ── Outlier plot ──────────────────────────────────────────────────────────
    if not args.no_outlier_plot:
        images_path = args.data_dir / 'images_filtered.npy'
        if not images_path.exists():
            print(f"\nWARNING: images file not found ({images_path}), skipping outlier plot.")
        else:
            print("\nLoading images for outlier plot...")
            images = np.load(images_path).astype(np.float32) / 255
            train_images = images[labelled_train_idx][:_n_tr]
            train_labels = np.load(data_dir / 'train_labels.npy')[:_n_tr]
            plot_umap_outliers(
                _all_2d[:_n_tr],
                train_images,
                OUTPUT_DIR=umap_dir,
                labels=train_labels,
                save_prefix='umap_outliers',
            )

    print(f"\nUMAP plots saved to {umap_dir}/")


# =============================================================================
# LEGACY MODE
# =============================================================================

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
        raise RuntimeError(f"No embedding files found in {embeddings_dir}.")
    return np.vstack(all_emb), np.concatenate(all_lbl), np.concatenate(all_ds)


def fit_umap_legacy(embeddings: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=SEED,
    )
    print(f"  Fitting UMAP on {len(embeddings)} samples...")
    return reducer.fit_transform(embeddings)


def plot_by_dataset(xy, ds_idx, datasets, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ds_name in enumerate(datasets):
        mask = ds_idx == i
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=DATASET_COLORS[i % len(DATASET_COLORS)],
            s=6, alpha=0.5, label=f"{ds_name} ({mask.sum()})",
        )
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP — coloured by dataset")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_label(xy, labels, output_path):
    unique_labels = sorted(set(labels.tolist()))
    labelled = [l for l in unique_labels if l != -1]
    cmap = plt.cm.get_cmap("tab10", max(len(labelled), 1))
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_unlabelled = labels == -1
    if mask_unlabelled.any():
        ax.scatter(xy[mask_unlabelled, 0], xy[mask_unlabelled, 1],
                   c="lightgrey", s=4, alpha=0.3,
                   label=f"unlabelled ({mask_unlabelled.sum()})")
    for j, lbl in enumerate(labelled):
        mask = labels == lbl
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=[cmap(j)], s=10, alpha=0.7,
                   label=f"label {lbl} ({mask.sum()})")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP — coloured by label")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def legacy_main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading embeddings...")
    embeddings, labels, ds_idx = load_embeddings(args.embeddings_dir, args.datasets)
    print(f"Total: {len(embeddings)} samples from {len(args.datasets)} datasets")
    xy = fit_umap_legacy(embeddings, args.n_neighbors, args.min_dist)
    np.save(args.output / "umap_coords.npy", xy)
    print(f"Saved UMAP coordinates: {args.output / 'umap_coords.npy'}")
    plot_by_dataset(xy, ds_idx, args.datasets, args.output / "umap_by_dataset.png")
    plot_by_label(xy, labels, args.output / "umap_by_label.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="UMAP visualisation of BYOL encoder embeddings"
    )

    # ── run-dir mode ──────────────────────────────────────────────────────────
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run output directory (e.g. outputs/run_sweep_f0.1_sw0.5_*). "
                         "Activates run-dir mode.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"Dataset directory with images_filtered.npy / labels_filtered.npy "
                         f"(default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--no-outlier-plot", action="store_true",
                    help="Skip the outlier image panel (run-dir mode only)")
    ap.add_argument("--umap-n-neighbors", type=int, default=30,
                    help="UMAP n_neighbors (default: 30, same as train_byol.py)")
    ap.add_argument("--umap-min-dist",    type=float, default=0.1,
                    help="UMAP min_dist (default: 0.1)")

    # ── legacy mode ───────────────────────────────────────────────────────────
    ap.add_argument("--datasets",       nargs="+", default=None,
                    help="[Legacy] Dataset names (must have matching *_embeddings.npy files)")
    ap.add_argument("--embeddings_dir", type=Path,  default=None,
                    help="[Legacy] Directory containing *_embeddings.npy / *_labels.npy")
    ap.add_argument("--output",         type=Path,  default=None,
                    help="[Legacy] Directory for output PNG files")
    ap.add_argument("--n_neighbors",    type=int,   default=15,
                    help="[Legacy] UMAP n_neighbors (default: 15)")
    ap.add_argument("--min_dist",       type=float, default=0.1,
                    help="[Legacy] UMAP min_dist (default: 0.1)")

    args = ap.parse_args()

    if args.run_dir is not None:
        run_dir_main(args)
    elif args.datasets is not None and args.embeddings_dir is not None and args.output is not None:
        legacy_main(args)
    else:
        ap.print_help()
        print("\nERROR: provide either --run-dir or (--datasets + --embeddings_dir + --output).",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
