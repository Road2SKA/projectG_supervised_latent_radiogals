from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import umap


def _sector_marker(theta1_deg: float, theta2_deg: float, n: int = 30) -> mpath.Path:
    """Return a unit pie-sector Path from theta1 to theta2 (degrees)."""
    angles = np.linspace(np.radians(theta1_deg), np.radians(theta2_deg), n)
    verts = [(0.0, 0.0)] + list(zip(np.cos(angles), np.sin(angles))) + [(0.0, 0.0)]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * n + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)


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
    type_labels: np.ndarray | None = None,
    simple_multilabel: bool = False,
) -> None:
    """
    simple_multilabel : if True, multi-label sources are drawn as open white
                        circles instead of pie-sector wedges.
    """


    n_pure = np.sum(colors >= 0)
    n_nonpure = np.sum(colors == -1)

    if type_labels is not None:
        n_classes = type_labels.shape[1]
        active_counts = type_labels.sum(axis=1)  # (N,)

        # Sources with no active label → light grey, same size as labelled points
        mask_unlabelled = active_counts == 0
        if np.any(mask_unlabelled):
            ax.scatter(
                embedding_2d[mask_unlabelled, 0], embedding_2d[mask_unlabelled, 1],
                c='lightgrey', s=15, alpha=0.2, zorder=1,
            )

        # Single-label (pure) sources: regular scatter per class
        for class_idx in range(n_classes):
            mask = (type_labels[:, class_idx] == 1) & (active_counts == 1)
            if np.any(mask):
                ax.scatter(
                    embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=[FIXED_COLORS[class_idx]], s=15, alpha=0.85, zorder=3,
                )

        # Multi-label sources
        multi_mask = active_counts > 1
        if simple_multilabel:
            # Draw as open white circles — fast and uncluttered
            if np.any(multi_mask):
                ax.scatter(
                    embedding_2d[multi_mask, 0], embedding_2d[multi_mask, 1],
                    facecolors='white', edgecolors='grey', linewidths=0.5,
                    s=15, alpha=0.7, zorder=3,
                )
        else:
            # Pie-sector wedges coloured by each active label
            multi_idx = np.where(multi_mask)[0]
            for i in multi_idx:
                x, y = embedding_2d[i]
                active = [ci for ci in range(n_classes) if type_labels[i, ci] == 1]
                n_active = len(active)
                angle_per = 360.0 / n_active
                for j, ci in enumerate(active):
                    theta1 = j * angle_per - 90
                    theta2 = (j + 1) * angle_per - 90
                    ax.scatter(x, y,
                               marker=_sector_marker(theta1, theta2),
                               c=[FIXED_COLORS[ci]], s=15, alpha=0.85, zorder=3)

        # Centroid labels: all sources with that label active, total count
        for class_idx, class_name in enumerate(class_names):
            active_mask = type_labels[:, class_idx] == 1
            n_total = np.sum(active_mask)
            if n_total > 0:
                color = FIXED_COLORS[class_idx]
                cx = embedding_2d[active_mask, 0].mean()
                cy = embedding_2d[active_mask, 1].mean()
                ax.annotate(
                    f'{class_name} ({n_total})', (cx, cy),
                    fontsize=13, fontweight='bold',
                    ha='center', va='center',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                              alpha=0.6, edgecolor='none'),
                )

    else:
        # Fallback: original pure-only behaviour
        mask_nonpure = colors == -1
        if np.any(mask_nonpure):
            ax.scatter(
                embedding_2d[mask_nonpure, 0], embedding_2d[mask_nonpure, 1],
                c='lightgrey', s=8, alpha=0.3,
            )

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
                    fontsize=13, fontweight='bold',
                    ha='center', va='center',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                              alpha=0.6, edgecolor='none'),
                )

    ax.set_xlabel('UMAP 1', fontsize=13)
    ax.set_ylabel('UMAP 2', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    # Place title in the emptiest corner
    xmin, xmax = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
    ymin, ymax = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    corner_counts = {
        'BL': np.sum((embedding_2d[:, 0] < xmid) & (embedding_2d[:, 1] < ymid)),
        'BR': np.sum((embedding_2d[:, 0] >= xmid) & (embedding_2d[:, 1] < ymid)),
        'TL': np.sum((embedding_2d[:, 0] < xmid) & (embedding_2d[:, 1] >= ymid)),
        'TR': np.sum((embedding_2d[:, 0] >= xmid) & (embedding_2d[:, 1] >= ymid)),
    }
    emptiest = min(corner_counts, key=corner_counts.get)
    tx = xmin + 0.02 * (xmax - xmin) if emptiest in ('BL', 'TL') else xmax - 0.02 * (xmax - xmin)
    ty = ymin + 0.03 * (ymax - ymin) if emptiest in ('BL', 'BR') else ymax - 0.03 * (ymax - ymin)
    ha = 'left' if emptiest in ('BL', 'TL') else 'right'
    va = 'bottom' if emptiest in ('BL', 'BR') else 'top'
    ax.text(tx, ty, f'{class_type.capitalize()} | {n_pure} pure, {n_nonpure} non-pure',
            fontsize=13, fontweight='bold', ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))


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
    simple_multilabel: bool = False,
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
    fig.suptitle(f'UMAP - {title_suffix}', fontsize=20)

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

        if type_labels.shape[1] == 0:
            ax.set_visible(False)
            continue

        colors, class_names = get_pure_class_colors(type_labels, class_type, CLASS_NAMES)
        _plot_umap_ax(ax, embedding_2d, colors, class_names, class_type,
                      type_labels=type_labels, simple_multilabel=simple_multilabel)

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
    n_outliers: int = 4,
) -> None:
    """
    Show the n most extreme outliers in UMAP space alongside their images.

    Outliers are defined as the points furthest from the centroid of train_2d.
    images must be in the same order as train_2d (use a non-shuffled loader).

    Each row in the grid shows the galaxy image (left) and a UMAP overview
    with the outlier highlighted (right).

    Args:
        train_2d:    (N, 2) UMAP coordinates, aligned with images
        images:      (N, H, W) original image array
        OUTPUT_DIR:  directory where PNG is saved
        labels:      (N, C) multi-hot label array aligned with images (optional)
        save_prefix: filename prefix
        n_outliers:  number of outliers to show (default 4)
    """
    centroid = train_2d.mean(axis=0)
    distances = np.linalg.norm(train_2d - centroid, axis=1)

    # Greedy diverse selection: start with the most extreme point, then
    # iteratively pick the next candidate that is farthest from all already-
    # selected outliers in UMAP space. This avoids showing the same source
    # multiple times when duplicate or near-duplicate images cluster at the
    # same extreme UMAP position.
    all_by_dist = np.argsort(distances)[::-1]
    selected = [int(all_by_dist[0])]
    for idx in all_by_dist[1:]:
        if len(selected) >= n_outliers:
            break
        min_sep = min(np.linalg.norm(train_2d[idx] - train_2d[s]) for s in selected)
        if min_sep > 0:
            selected.append(int(idx))
    outlier_indices = selected

    fig, axes = plt.subplots(n_outliers, 2, figsize=(10, 4 * n_outliers))
    fig.suptitle('Most Extreme UMAP Outliers', fontsize=14)

    other_color = 'orange'

    for row, idx in enumerate(outlier_indices):
        x, y = train_2d[idx]
        dist = distances[idx]

        # Left: galaxy image
        ax_img = axes[row, 0]
        ax_img.imshow(images[idx], cmap='viridis', origin='lower')
        if labels is not None and idx < len(labels):
            active = [ALL_CLASS_NAMES[j] for j in range(labels.shape[1]) if labels[idx, j] == 1]
            class_str = ', '.join(active) if active else '—'
        else:
            class_str = '—'
        ax_img.set_title(f'Outlier #{row + 1} | {class_str} | dist: {dist:.2f}')
        ax_img.axis('off')

        # Right: UMAP overview with all outliers marked
        ax_umap = axes[row, 1]
        ax_umap.scatter(train_2d[:, 0], train_2d[:, 1],
                        c='lightgrey', s=3, alpha=0.3, zorder=1)
        # Mark all other outliers in orange
        for other_row, other_idx in enumerate(outlier_indices):
            if other_idx == idx:
                continue
            ox, oy = train_2d[other_idx]
            ax_umap.scatter(ox, oy, c=other_color, s=60, marker='*',
                            zorder=2, label=f'Outlier #{other_row + 1}')
        # Highlight current outlier in red (on top)
        ax_umap.scatter(x, y, c='red', s=120, marker='*',
                        zorder=3, label=f'Outlier #{row + 1} (this)')
        ax_umap.set_xlabel('UMAP 1')
        ax_umap.set_ylabel('UMAP 2')
        ax_umap.legend(fontsize=8)
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
    loss_diff = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]

    monitor_raw = history.get('monitor_val_loss')
    monitor_valid = (monitor_raw is not None
                     and any(v is not None for v in monitor_raw))
    monitor = monitor_raw if monitor_valid else None

    sched = history.get('supervision_schedule')
    sched_label = 'Supervision Weight' if loss_mode == 'both' else 'Pairing Probability'

    def _add_best(ax):
        ax.axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training History - {model_type.upper()} Model', fontsize=FS_S)

    # Top-left: losses
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    if monitor:
        axes[0, 0].plot(epochs, monitor, color='orange', linestyle='--',
                        label='Val Monitor', linewidth=2, alpha=0.85)
    axes[0, 0].axhline(y=best_val_loss, color='g', linestyle='--',
                       label=f'Best Val ({best_val_loss:.4f})', alpha=0.7)
    _add_best(axes[0, 0])
    axes[0, 0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2,
                       label=f'Best Epoch ({best_epoch})', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=FS)
    axes[0, 0].set_ylabel('Loss', fontsize=FS)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=FS_T)
    axes[0, 0].legend(fontsize=FS)
    axes[0, 0].tick_params(labelsize=FS)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: overfitting indicator
    axes[0, 1].plot(epochs, loss_diff, 'purple', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    _add_best(axes[0, 1])
    axes[0, 1].set_xlabel('Epoch', fontsize=FS)
    axes[0, 1].set_ylabel('Val Loss - Train Loss', fontsize=FS)
    axes[0, 1].set_title('Overfitting Indicator (Val - Train)', fontsize=FS_T)
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

        axes2[0].plot(epochs_zoom, history['train_loss'][start_idx:], 'b-', label='Train Loss', linewidth=2)
        axes2[0].plot(epochs_zoom, history['val_loss'][start_idx:], 'r-', label='Val Loss', linewidth=2)
        if monitor:
            axes2[0].plot(epochs_zoom, monitor[start_idx:], color='orange', linestyle='--',
                          label='Val Monitor', linewidth=2, alpha=0.85)
        axes2[0].axhline(y=best_val_loss, color='g', linestyle='--',
                         label=f'Best Val ({best_val_loss:.4f})', alpha=0.7)
        if best_epoch >= start_idx:
            axes2[0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2,
                             label=f'Best Epoch ({best_epoch})', alpha=0.7)
        axes2[0].set_xlabel('Epoch', fontsize=FS)
        axes2[0].set_ylabel('Loss', fontsize=FS)
        axes2[0].set_title('Loss (Zoomed)', fontsize=FS_T)
        axes2[0].legend(fontsize=FS)
        axes2[0].tick_params(labelsize=FS)
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(epochs_zoom, loss_diff[start_idx:], 'purple', linewidth=2)
        axes2[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        if best_epoch >= start_idx:
            axes2[1].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)
        axes2[1].set_xlabel('Epoch', fontsize=FS)
        axes2[1].set_ylabel('Val Loss - Train Loss', fontsize=FS)
        axes2[1].set_title('Overfitting Indicator (Zoomed)', fontsize=FS_T)
        axes2[1].tick_params(labelsize=FS)
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'training_curves_zoomed{suffix}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Zoomed training curves saved to {output_dir / f'training_curves_zoomed{suffix}.png'}")

    plt.close('all')
