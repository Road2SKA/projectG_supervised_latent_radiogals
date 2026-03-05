from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import umap


# =============================================================================
# HELPER: ASSIGN COLORS TO PURE CLASS SAMPLES
# =============================================================================

def get_pure_class_colors(
    labels: np.ndarray,
    label_type: str,
    CLASS_NAMES: dict,
) -> tuple[np.ndarray, list]:    
    """
    Assign colors to pure class samples, grey for non-pure.

    Args:
        labels:      (N, D) array of labels for current classification type
        label_type:  'initial', 'morphology', 'environment', or 'derived'
        CLASS_NAMES: dict mapping label_type -> list of class name strings

    Returns:
        colors:      (N,) array of color indices (-1 for grey, 0-K for pure classes)
        class_names: list of class names for legend
    """
    n_samples, n_classes = labels.shape
    colors = np.full(n_samples, -1, dtype=int)  # -1 = grey (non-pure)

    # A sample is "pure" if exactly one label equals 1
    for i in range(n_samples):
        label_vec = labels[i]
        n_ones = np.sum(label_vec == 1)
        if n_ones == 1:
            class_idx = np.where(label_vec == 1)[0][0]
            colors[i] = class_idx

    class_names = CLASS_NAMES.get(label_type, [f"Class {i}" for i in range(n_classes)])
    return colors, class_names


# =============================================================================
# MAIN PLOT FUNCTION
# =============================================================================

def plot_umap_pure_classes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title_suffix: str,
    save_prefix: str,
    split_name: str,
    args,
    SEED: int,
    LABEL_RANGES: dict,
    CLASS_NAMES: dict,
    OUTPUT_DIR: Path,
    train_labels_full: np.ndarray | None = None,
    test_labels_full: np.ndarray | None = None,
) -> None:
    """
    Generate UMAP plots for each classification type with pure class colouring.

    Args:
        embeddings:        (N, D) array of projected embeddings
        labels:            (N, L) label array (may be a subset if label_type != 'full')
        title_suffix:      string appended to plot title
        save_prefix:       filename prefix for saved PNGs
        split_name:        'train' or 'test' (used to select full labels when needed)
        args:              parsed argparse Namespace (uses label_type, umap_n_neighbors,
                           umap_min_dist)
        SEED:              random seed for UMAP reproducibility
        LABEL_RANGES:      dict mapping label_type -> (start, end) index tuple
        CLASS_NAMES:       dict mapping label_type -> list of class name strings
        OUTPUT_DIR:        pathlib.Path where PNG files are saved
        train_labels_full: (N_train, 20) full label array for train split (optional)
        test_labels_full:  (N_test,  20) full label array for test  split (optional)
    """
    print(f"  Computing UMAP for {title_suffix}...")

    # Fit UMAP once; reuse 2-D coordinates for all classification types
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric='euclidean',
        random_state=SEED,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # One plot per classification type
    for class_type in ['initial', 'morphology', 'environment', 'derived']:
        label_start, label_end = LABEL_RANGES[class_type]

        # Select the correct label columns for this classification type
        if args.label_type == 'full':
            # Full labels already available — just slice the right columns
            type_labels = labels[:, label_start:label_end]
        else:
            # Labels were pre-filtered; reload from the full label arrays
            if split_name == 'train':
                if train_labels_full is None:
                    continue
                full_labels = train_labels_full[:len(embeddings)]
            elif split_name == 'test':
                if test_labels_full is None:
                    continue
                full_labels = test_labels_full[:len(embeddings)]
            else:
                continue  # skip val set for UMAP
            type_labels = full_labels[:, label_start:label_end]

        # Colour assignment
        colors, class_names = get_pure_class_colors(type_labels, class_type, CLASS_NAMES)
        n_pure    = np.sum(colors >= 0)
        n_nonpure = np.sum(colors == -1)

        # ── Figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(
            f'UMAP - {title_suffix} - {class_type.capitalize()} Classification',
            fontsize=14,
        )

        # Non-pure samples in grey (background layer)
        mask_nonpure = colors == -1
        if np.any(mask_nonpure):
            ax.scatter(
                embedding_2d[mask_nonpure, 0],
                embedding_2d[mask_nonpure, 1],
                c='lightgrey', s=10, alpha=0.3,
                label=f'Non-pure ({n_nonpure})',
            )

        # Pure class samples with distinct colours
        n_classes = len(class_names)
        cmap = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')

        for class_idx in range(n_classes):
            mask_class = colors == class_idx
            n_class = np.sum(mask_class)
            if n_class > 0:
                ax.scatter(
                    embedding_2d[mask_class, 0],
                    embedding_2d[mask_class, 1],
                    c=[cmap(class_idx)], s=15, alpha=0.7,
                    label=f'{class_names[class_idx]} ({n_class})',
                )

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{n_pure} pure samples, {n_nonpure} non-pure')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        umap_path = OUTPUT_DIR / f'{save_prefix}_{class_type}.png'
        plt.savefig(umap_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved {class_type} to {umap_path}")
        plt.close()