#!/usr/bin/env python3
"""
Generate pca_variance.png for every run* output directory that has projection
files but no existing pca_variance plot.

Usage:
    python scripts/backfill_pca_variance.py [--force] [--outputs-dir PATH]
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

SEED = 42
N_COMPONENTS = 128


def make_pca_plot(run_dir: Path, suffix: str = "", force: bool = False) -> bool:
    figures_dir = run_dir / "figures"
    out_path = figures_dir / f"pca_variance{suffix}.png"

    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists")
        return False

    data_dir = run_dir / "data"
    train_f = data_dir / f"labelled_train_projections{suffix}.npy"
    test_f  = data_dir / f"test_projections{suffix}.npy"
    unlab_f = data_dir / f"unlabelled_train_projections{suffix}.npy"

    if not train_f.exists() or not test_f.exists():
        print(f"  [skip] missing projection files for suffix '{suffix}'")
        return False

    parts = [np.load(train_f), np.load(test_f)]
    if unlab_f.exists():
        parts.insert(0, np.load(unlab_f))
    pca_all = np.concatenate(parts)

    n_components = min(N_COMPONENTS, pca_all.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(pca_all)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1 if cumvar[-1] >= 0.95 else f">{n_components}"
    n_99 = int(np.searchsorted(cumvar, 0.99)) + 1 if cumvar[-1] >= 0.99 else f">{n_components}"

    print(f"  dim={pca_all.shape[1]}d  n={len(pca_all)}  95%={n_95}d  99%={n_99}d  "
          f"coverage({n_components})={cumvar[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(max(6, n_components * 0.25), 4))
    ax.bar(np.arange(1, n_components + 1),
           pca.explained_variance_ratio_ * 100,
           color="tab:blue", width=1.0)
    if isinstance(n_95, int):
        ax.axvline(n_95, color="tab:orange", linestyle="--", linewidth=1.5,
                   label=f"95% variance ({n_95}d)")
    if isinstance(n_99, int):
        ax.axvline(n_99, color="tab:red", linestyle=":", linewidth=1.5,
                   label=f"99% variance ({n_99}d)")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(
        f"Latent space PCA variance  —  {pca_all.shape[1]}d projection\n"
        f"95% in {n_95}d  |  99% in {n_99}d  |  {n_components}-component coverage: {cumvar[-1]:.3f}"
    )
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Path to outputs directory (default: outputs/)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pca_variance plots")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    run_dirs = sorted(outputs_dir.glob("run*"))
    if not run_dirs:
        print(f"No run* directories found under {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run* directories under {outputs_dir}\n")
    generated = 0

    for run_dir in run_dirs:
        print(f"{run_dir.name}")
        # Detect all suffixes present (handles fold runs: _fold1, _fold2, …)
        train_files = sorted((run_dir / "data").glob("labelled_train_projections*.npy"))
        if not train_files:
            print("  [skip] no projection files found")
            continue
        for tf in train_files:
            suffix = tf.stem.replace("labelled_train_projections", "")
            if make_pca_plot(run_dir, suffix=suffix, force=args.force):
                generated += 1

    print(f"\nDone — generated {generated} plot(s).")


if __name__ == "__main__":
    main()
