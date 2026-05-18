from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import umap


# Fixed colour palette — consistent across runs and class types
_CMAP = plt.cm.get_cmap('tab20')
FIXED_COLORS = [_CMAP(i) for i in range(20)]


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
        if np.sum(label_vec == 1) == 1:
            colors[i] = np.where(label_vec == 1)[0][0]

    class_names = CLASS_NAMES.get(label_type, [f"Class {i}" for i in range(n_classes)])
    return colors, class_names


# =============================================================================
# HELPER: PLOT ONE CLASSIFICATION TYPE ON A GIVEN AXIS
# =============================================================================

def _plot_umap_ax(
    ax,
    embedding_2d: np.ndarray,
    colors: np.ndarray,
    class_names: list,
    class_type: str,
) -> None:
    n_pure = np.sum(colors >= 0)
    n_nonpure = np.sum(colors == -1)

    # Non-pure samples (background)
    mask_nonpure = colors == -1
    if np.any(mask_nonpure):
        ax.scatter(
            embedding_2d[mask_nonpure, 0], embedding_2d[mask_nonpure, 1],
            c='lightgrey', s=8, alpha=0.3,
        )

    # Pure class samples with fixed colours + in-figure centroid labels
    for class_idx, class_name in enumerate(class_names):
        mask_class = colors == class_idx
        n_class = np.sum(mask_class)
        if n_class > 0:
            color = FIXED_COLORS[class_idx]
            ax.scatter(
                embedding_2d[mask_class, 0], embedding_2d[mask_class, 1],
                c=[color], s=12, alpha=0.7,
            )
            cx = embedding_2d[mask_class, 0].mean()
            cy = embedding_2d[mask_class, 1].mean()
            ax.annotate(
                f'{class_name} ({n_class})', (cx, cy),
                fontsize=10.5, fontweight='bold',
                ha='center', va='center',
                color='black',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                          alpha=0.6, edgecolor='none'),
            )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'{class_type.capitalize()} | {n_pure} pure, {n_nonpure} non-pure')
    ax.grid(True, alpha=0.3)


# =============================================================================
# UMAP FITTING HELPER
# =============================================================================

def fit_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple:
    """Fit a UMAP reducer and return (reducer, embedding_2d)."""
    print(f"  Fitting UMAP on {len(embeddings)} samples...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=seed,
    )
    return reducer, reducer.fit_transform(embeddings)


# =============================================================================
# SINGLE-COLOURING UMAP PLOT
# =============================================================================

def plot_umap_single(
    embedding_2d: np.ndarray,
    labels_full: np.ndarray,
    colouring: str,
    CLASS_NAMES: dict,
    LABEL_RANGES: dict,
    title: str,
    save_path: Path,
    split_masks: dict | None = None,
) -> None:
    """
    Save one UMAP figure for a single colouring scheme.

    Args:
        embedding_2d:  (N, 2) pre-computed UMAP coordinates
        labels_full:   (N, 20) full label array aligned with embedding_2d
        colouring:     'initial', 'morphology', or 'train_labelled'
        CLASS_NAMES:   dict mapping label_type -> list of class name strings
        LABEL_RANGES:  dict mapping label_type -> (start, end) column indices
        title:         figure title
        save_path:     full path for the saved PNG
        split_masks:   dict {split_name: bool_array} — used for 'train_labelled' colouring
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if colouring in ('initial', 'morphology'):
        label_start, label_end = LABEL_RANGES[colouring]
        type_labels = labels_full[:, label_start:label_end]
        colors, class_names = get_pure_class_colors(type_labels, colouring, CLASS_NAMES)
        _plot_umap_ax(ax, embedding_2d, colors, class_names, colouring)
        ax.set_title(title)

    elif colouring == 'train_labelled':
        _split_colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
        if split_masks:
            for i, (split_name, mask) in enumerate(split_masks.items()):
                n = int(mask.sum())
                ax.scatter(
                    embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=_split_colors[i % len(_split_colors)],
                    s=10, alpha=0.7, label=f'{split_name} ({n})',
                )
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved {save_path.name}")
    plt.close()


# =============================================================================
# MAIN UMAP PLOT FUNCTION
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
    reducer=None,
) -> tuple:
    """
    Generate a 2×2 UMAP grid for all classification types with pure class colouring.
    Saves UMAP 2D coordinates as .npy for later reuse.

    Args:
        embeddings:         (N, D) array of projected embeddings
        labels:             (N, L) label array (may be a subset if label_type != 'full')
        title_suffix:       string appended to plot title
        save_prefix:        filename prefix for saved PNG and .npy
        split_name:         'train' or 'test'
        args:               parsed argparse Namespace
        SEED:               random seed for UMAP reproducibility
        LABEL_RANGES:       dict mapping label_type -> (start, end) index tuple
        CLASS_NAMES:        dict mapping label_type -> list of class name strings
        OUTPUT_DIR:         pathlib.Path where files are saved
        train_labels_full:  (N_train, 20) full label array for train split (optional)
        test_labels_full:   (N_test,  20) full label array for test  split (optional)
        reducer:            pre-fitted UMAP reducer; uses transform() instead of fit_transform()
        annotate_centroids: if True, annotate class centroids with class name labels

    Returns:
        (reducer, embedding_2d): fitted UMAP reducer and (N, 2) 2D coordinates
    """
    if reducer is None:
        print(f"  Fitting UMAP on {title_suffix}...")
        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric='euclidean',
            random_state=SEED,
        )
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        print(f"  Transforming {title_suffix} with pre-fitted UMAP...")
        embedding_2d = reducer.transform(embeddings)

    # 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'UMAP - {title_suffix}', fontsize=16)

    for idx, class_type in enumerate(['initial', 'morphology', 'environment', 'derived']):
        ax = axes[idx // 2, idx % 2]
        label_start, label_end = LABEL_RANGES[class_type]

        if args.label_type == 'full':
            type_labels = labels[:, label_start:label_end]
        else:
            if split_name == 'train':
                if train_labels_full is None:
                    ax.set_visible(False)
                    continue
                full_labels = train_labels_full[:len(embeddings)]
            elif split_name == 'test':
                if test_labels_full is None:
                    ax.set_visible(False)
                    continue
                full_labels = test_labels_full[:len(embeddings)]
            else:
                ax.set_visible(False)
                continue
            type_labels = full_labels[:, label_start:label_end]

        colors, class_names = get_pure_class_colors(type_labels, class_type, CLASS_NAMES)
        _plot_umap_ax(ax, embedding_2d, colors, class_names, class_type)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{save_prefix}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved UMAP grid to {save_path}")
    plt.close()

    return reducer, embedding_2d


# =============================================================================
# OVERLAY PLOT: TRAIN (BACKGROUND) + TEST (FOREGROUND) IN SAME UMAP SPACE
# =============================================================================

def plot_umap_overlay(
    train_2d: np.ndarray,
    test_2d: np.ndarray,
    train_labels_full: np.ndarray,
    test_labels_full: np.ndarray,
    LABEL_RANGES: dict,
    CLASS_NAMES: dict,
    OUTPUT_DIR: Path,
    save_prefix: str,
) -> None:
    """
    Plot train points (background, faded) and test points (foreground, opaque)
    in the same UMAP space as a 2×2 grid.

    Args:
        train_2d:          (N_train, 2) UMAP coordinates for train set
        test_2d:           (N_test,  2) test embeddings transformed into train UMAP space
        train_labels_full: (N_train, 20) full label array for train split
        test_labels_full:  (N_test,  20) full label array for test split
        LABEL_RANGES:      dict mapping label_type -> (start, end) index tuple
        CLASS_NAMES:       dict mapping label_type -> list of class name strings
        OUTPUT_DIR:        pathlib.Path where PNG is saved
        save_prefix:       filename prefix for saved PNG
    """
    print(f"  Generating train+test overlay UMAP...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('UMAP Overlay — Train (faded) + Test (solid) in train UMAP space', fontsize=16)

    for idx, class_type in enumerate(['initial', 'morphology', 'environment', 'derived']):
        ax = axes[idx // 2, idx % 2]
        label_start, label_end = LABEL_RANGES[class_type]

        train_type_labels = train_labels_full[:len(train_2d), label_start:label_end]
        test_type_labels  = test_labels_full[:len(test_2d),  label_start:label_end]

        train_colors, class_names = get_pure_class_colors(train_type_labels, class_type, CLASS_NAMES)
        test_colors,  _           = get_pure_class_colors(test_type_labels,  class_type, CLASS_NAMES)

        # ── Train: background layer (small, faded) ────────────────────────────
        mask = train_colors == -1
        if np.any(mask):
            ax.scatter(train_2d[mask, 0], train_2d[mask, 1],
                       c='lightgrey', s=5, alpha=0.15, zorder=1)

        for class_idx in range(len(class_names)):
            mask = train_colors == class_idx
            if np.any(mask):
                ax.scatter(train_2d[mask, 0], train_2d[mask, 1],
                           c=[FIXED_COLORS[class_idx]], s=5, alpha=0.15, zorder=2)

        # ── Test: foreground layer (larger, opaque, white edge) ───────────────
        mask = test_colors == -1
        n_nonpure = np.sum(mask)
        if np.any(mask):
            ax.scatter(test_2d[mask, 0], test_2d[mask, 1],
                       c='grey', s=25, alpha=0.6, zorder=3,
                       label=f'Non-pure ({n_nonpure})')

        for class_idx, class_name in enumerate(class_names):
            mask = test_colors == class_idx
            n_class = np.sum(mask)
            if np.any(mask):
                ax.scatter(test_2d[mask, 0], test_2d[mask, 1],
                           c=[FIXED_COLORS[class_idx]], s=30, alpha=0.9,
                           edgecolors='white', linewidths=0.4, zorder=4,
                           label=f'{class_name} ({n_class})')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{class_type.capitalize()} | test in train space')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{save_prefix}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved overlay to {save_path}")
    plt.close()


# =============================================================================
# UMAP OUTLIER IMAGES
# =============================================================================

ALL_CLASS_NAMES = [
    'FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles',
    'C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
    'Straight jets', 'Multi hotspots', 'Cont. jets', 'Banding', 'One-sided', 'Restarted',
    'Cluster', 'Merger', 'Diffuse', 'Unknown',
]


def plot_umap_outliers(
    train_2d: np.ndarray,
    images: np.ndarray,
    OUTPUT_DIR: Path,
    labels: np.ndarray | None = None,
    save_prefix: str = "umap_outliers",
    n_regions: int = 3,
    n_per_region: int = 3,
) -> None:
    """
    Show n_per_region examples from each of n_regions extreme UMAP regions.

    Layout: n_regions rows × (n_per_region image columns + 1 UMAP column).
    The UMAP panel (right) spans all rows and marks all selected points
    coloured by region: red (top), blue (middle), yellow (bottom).

    Args:
        train_2d:       (N, 2) UMAP coordinates, aligned with images
        images:         (N, H, W) original image array
        OUTPUT_DIR:     directory where PNG is saved
        labels:         (N, C) multi-hot label array aligned with images (optional)
        save_prefix:    filename prefix
        n_regions:      number of extreme regions (default 3)
        n_per_region:   examples per region (default 3)
    """
    centroid  = train_2d.mean(axis=0)
    distances = np.linalg.norm(train_2d - centroid, axis=1)

    # ── Select n_regions anchor points (maximally spread in UMAP space) ──────
    N_CAND   = max(60, n_regions * n_per_region * 6)
    cands    = np.argsort(distances)[::-1][:N_CAND].tolist()

    anchors  = [cands[0]]
    # anchor 1: farthest from anchor 0
    anchors.append(int(max(cands[1:], key=lambda c: np.linalg.norm(train_2d[c] - train_2d[anchors[0]]))))
    # anchor 2: maximises min-distance to both existing anchors
    anchors.append(int(max(
        (c for c in cands if c not in anchors),
        key=lambda c: min(np.linalg.norm(train_2d[c] - train_2d[a]) for a in anchors),
    )))

    # ── Assign candidates to nearest anchor, keep n_per_region each ──────────
    bucket = {a: [a] for a in anchors}
    for c in cands:
        if c in anchors:
            continue
        nearest = min(anchors, key=lambda a: np.linalg.norm(train_2d[c] - train_2d[a]))
        bucket[nearest].append(c)

    regions = []
    for anchor in anchors:
        members = sorted(bucket[anchor],
                         key=lambda c: np.linalg.norm(train_2d[c] - train_2d[anchor]))
        regions.append(members[:n_per_region])

    # ── Build subplot mosaic: image cols then shared UMAP col ─────────────────
    region_colors = ['red', 'royalblue', 'gold']
    mosaic = [[f'img{r}{c}' for c in range(n_per_region)] + ['umap']
              for r in range(n_regions)]

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(3.5 * (n_per_region + 1.4), 3.5 * n_regions),
        gridspec_kw={'width_ratios': [1] * n_per_region + [1.5]},
    )
    fig.suptitle('UMAP Region Samples', fontsize=14)

    ax_umap = axd['umap']
    ax_umap.scatter(train_2d[:, 0], train_2d[:, 1],
                    c='lightgrey', s=5, alpha=0.4, zorder=1)

    region_labels = ['Top', 'Middle', 'Bottom']
    for r, (region, color) in enumerate(zip(regions, region_colors)):
        for c, idx in enumerate(region):
            ax_img = axd[f'img{r}{c}']
            ax_img.imshow(images[idx], cmap='viridis', origin='lower')
            if labels is not None and idx < len(labels):
                active = [ALL_CLASS_NAMES[j] for j in range(min(labels.shape[1], len(ALL_CLASS_NAMES)))
                          if labels[idx, j] == 1]
                class_str = ', '.join(active) if active else '—'
            else:
                class_str = '—'
            ax_img.set_title(class_str, fontsize=7)
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax_img.axis('off')

            ax_umap.scatter(train_2d[idx, 0], train_2d[idx, 1],
                            c=color, s=90, marker='*', zorder=3,
                            edgecolors='black', linewidths=0.4)

        label = region_labels[r] if r < len(region_labels) else f'Region {r + 1}'
        ax_umap.scatter([], [], c=color, s=90, marker='*', label=label)

    ax_umap.set_xlabel('UMAP 1')
    ax_umap.set_ylabel('UMAP 2')
    ax_umap.set_title('UMAP Regions')
    ax_umap.legend(fontsize=9)
    ax_umap.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{save_prefix}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved outlier plot to {save_path}")
    plt.close()


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_curves(
    history: dict,
    best_val_loss: float,
    best_epoch: int,
    model_type: str,
    output_dir: Path,
    suffix: str = "",
    loss_mode: str = "both",
) -> None:
    """
    Plot and save training history curves.

    Args:
        history:       dict with keys 'train_loss', 'val_loss', 'lr', 'ema_decay',
                       optionally 'monitor_val_loss' and 'supervision_schedule'
        best_val_loss: best validation loss achieved
        best_epoch:    epoch at which best_val_loss was achieved
        model_type:    'convnet' or 'original' (used in plot title)
        output_dir:    directory where PNGs are saved
        loss_mode:     'both' or 'either' (determines supervision schedule label)
    """
    FS = 13   # base font size for axis labels / legend
    FS_T = 14 # subplot title font size
    FS_S = 18 # suptitle font size

    epochs = range(1, len(history['train_loss']) + 1)

    monitor_raw = history.get('monitor_val_loss')
    monitor_valid = (monitor_raw is not None
                     and any(v is not None for v in monitor_raw))
    monitor = monitor_raw if monitor_valid else None

    sched = history.get('supervision_schedule')
    sched_label = 'Supervision Weight' if loss_mode == 'both' else 'Pairing Probability'

    val_aug = history.get('val_aug_loss')
    val_fri = history.get('val_friend_loss')
    tr_aug  = history.get('train_aug_loss')
    tr_fri  = history.get('train_friend_loss')

    def _add_best(ax):
        ax.axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training History - {model_type.upper()} Model', fontsize=FS_S)

    # Top-left: combined losses overview
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train (total)', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val (total)', linewidth=2)
    if val_aug:
        axes[0, 0].plot(epochs, val_aug, 'r--', label='Val aug (★)', linewidth=1.5, alpha=0.8)
    if monitor:
        axes[0, 0].plot(epochs, monitor, color='orange', linestyle='--',
                        label='Val Monitor', linewidth=2, alpha=0.85)
    axes[0, 0].axhline(y=best_val_loss, color='g', linestyle='--',
                       label=f'Best ({best_val_loss:.4f})', alpha=0.7)
    axes[0, 0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2,
                       label=f'Best epoch ({best_epoch})', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=FS)
    axes[0, 0].set_ylabel('Loss', fontsize=FS)
    axes[0, 0].set_title('Combined Losses', fontsize=FS_T)
    axes[0, 0].legend(fontsize=FS)
    axes[0, 0].tick_params(labelsize=FS)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: L_aug and L_friend components for train and val
    if tr_aug:
        axes[0, 1].plot(epochs, tr_aug, 'b-', label='Train L_aug', linewidth=2)
    if tr_fri:
        axes[0, 1].plot(epochs, tr_fri, 'b--', label='Train L_friend', linewidth=1.5, alpha=0.8)
    if val_aug:
        axes[0, 1].plot(epochs, val_aug, 'r-', label='Val L_aug (★)', linewidth=2)
    if val_fri:
        axes[0, 1].plot(epochs, val_fri, 'r--', label='Val L_friend', linewidth=1.5, alpha=0.8)
    _add_best(axes[0, 1])
    axes[0, 1].set_xlabel('Epoch', fontsize=FS)
    axes[0, 1].set_ylabel('Loss', fontsize=FS)
    axes[0, 1].set_title('L_aug and L_friend Components', fontsize=FS_T)
    axes[0, 1].legend(fontsize=FS)
    axes[0, 1].tick_params(labelsize=FS)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: learning rate
    axes[1, 0].plot(epochs, history['lr'], 'orange', linewidth=2)
    _add_best(axes[1, 0])
    axes[1, 0].set_xlabel('Epoch', fontsize=FS)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=FS)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=FS_T)
    axes[1, 0].tick_params(labelsize=FS)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Bottom-right: supervision / prob schedule
    axes[1, 1].plot(epochs, sched if sched else [0] * len(list(epochs)), 'green', linewidth=2)
    axes[1, 1].set_ylabel(sched_label, fontsize=FS)
    axes[1, 1].set_title(f'{sched_label} Schedule', fontsize=FS_T)
    _add_best(axes[1, 1])
    axes[1, 1].set_xlabel('Epoch', fontsize=FS)
    axes[1, 1].tick_params(labelsize=FS)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves{suffix}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {output_dir / f'training_curves{suffix}.png'}")

    # Zoomed view of final 20%
    if len(history['train_loss']) > 10:
        start_idx = int(len(epochs) * 0.8)
        epochs_zoom = list(epochs)[start_idx:]

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle(f'Training History (Final 20%) - {model_type.upper()} Model', fontsize=FS_S)

        axes2[0].plot(epochs_zoom, history['train_loss'][start_idx:], 'b-', label='Train (total)', linewidth=2)
        axes2[0].plot(epochs_zoom, history['val_loss'][start_idx:], 'r-', label='Val (total)', linewidth=2)
        if val_aug:
            axes2[0].plot(epochs_zoom, val_aug[start_idx:], 'r--', label='Val aug (★)', linewidth=1.5, alpha=0.8)
        if monitor:
            axes2[0].plot(epochs_zoom, monitor[start_idx:], color='orange', linestyle='--',
                          label='Val Monitor', linewidth=2, alpha=0.85)
        axes2[0].axhline(y=best_val_loss, color='g', linestyle='--',
                         label=f'Best ({best_val_loss:.4f})', alpha=0.7)
        if best_epoch >= start_idx:
            axes2[0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2,
                             label=f'Best epoch ({best_epoch})', alpha=0.7)
        axes2[0].set_xlabel('Epoch', fontsize=FS)
        axes2[0].set_ylabel('Loss', fontsize=FS)
        axes2[0].set_title('Loss (Zoomed)', fontsize=FS_T)
        axes2[0].legend(fontsize=FS)
        axes2[0].tick_params(labelsize=FS)
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(epochs_zoom, tr_aug[start_idx:] if tr_aug else [], 'b-', label='Train L_aug', linewidth=2)
        axes2[1].plot(epochs_zoom, tr_fri[start_idx:] if tr_fri else [], 'b--', label='Train L_friend', linewidth=1.5, alpha=0.8)
        axes2[1].plot(epochs_zoom, val_aug[start_idx:] if val_aug else [], 'r-', label='Val L_aug (★)', linewidth=2)
        axes2[1].plot(epochs_zoom, val_fri[start_idx:] if val_fri else [], 'r--', label='Val L_friend', linewidth=1.5, alpha=0.8)
        if best_epoch >= start_idx:
            axes2[1].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)
        axes2[1].set_xlabel('Epoch', fontsize=FS)
        axes2[1].set_ylabel('Loss', fontsize=FS)
        axes2[1].set_title('L_aug / L_friend Components (Zoomed)', fontsize=FS_T)
        axes2[1].legend(fontsize=FS)
        axes2[1].tick_params(labelsize=FS)
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'training_curves_zoomed{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Zoomed training curves saved to {output_dir / f'training_curves_zoomed{suffix}.png'}")

    plt.close('all')
