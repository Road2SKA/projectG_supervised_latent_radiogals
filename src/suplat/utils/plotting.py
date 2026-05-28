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

    class_names = CLASS_NAMES.get(label_type, [f"Class {i}" for i in range(n_classes)])[:n_classes]
    return colors, class_names


# =============================================================================
# HELPER: PIE-DOT FOR NON-PURE SAMPLES
# =============================================================================

def _draw_pie_dot(ax, x, y, active_colors, radius, zorder=3):
    """Equal-wedge pie of active_colors; faded grey if no active classes."""
    from matplotlib.patches import Circle, Wedge
    n = len(active_colors)
    if n == 0:
        ax.add_patch(Circle((x, y), radius, facecolor='grey',
                             edgecolor='none', alpha=0.4, zorder=zorder))
        return
    step = 360.0 / n
    for k, color in enumerate(active_colors):
        ax.add_patch(Wedge((x, y), radius,
                           k * step, (k + 1) * step,
                           facecolor=color, edgecolor='none',
                           alpha=0.7, zorder=zorder))


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
    title: str | None = None,
) -> None:
    from matplotlib.lines import Line2D

    # Non-pure samples: grey scatter
    mask_nonpure = colors == -1
    ax.scatter(embedding_2d[mask_nonpure, 0], embedding_2d[mask_nonpure, 1],
               c='grey', s=10, alpha=0.4, zorder=1)

    # Pure class samples: per-class scatter
    for class_idx in range(len(class_names)):
        mask = colors == class_idx
        if not np.any(mask):
            continue
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[FIXED_COLORS[class_idx]], s=10, alpha=0.7, zorder=2)

    # Legend: single-class / multi-class / unclassified counts
    n_pure = int(np.sum(colors >= 0))
    if type_labels is not None and np.any(mask_nonpure):
        n_active = type_labels[mask_nonpure].sum(axis=1)
        n_multi = int(np.sum(n_active > 1))
        n_unclassified = int(np.sum(n_active == 0))
    else:
        n_multi = 0
        n_unclassified = int(np.sum(mask_nonpure))
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=6, alpha=0.7, label=f'Single-class ({n_pure})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=6, alpha=0.7, label=f'Multi-class ({n_multi})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
               markersize=6, alpha=0.7, label=f'Unclassified ({n_unclassified})'),
    ]
    ax.legend(handles=legend_handles, loc='best', fontsize=12, framealpha=0.7)

    # Centroid labels over ALL samples with that class active (pure + non-pure)
    for class_idx, class_name in enumerate(class_names):
        if type_labels is not None:
            mask_all = type_labels[:, class_idx] == 1
        else:
            mask_all = colors == class_idx
        n_class = int(np.sum(mask_all))
        if n_class > 0:
            color = FIXED_COLORS[class_idx]
            cx = embedding_2d[mask_all, 0].mean()
            cy = embedding_2d[mask_all, 1].mean()
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
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    _title = title if title is not None else class_type.capitalize()
    ax.text(0.5, 0.97, _title, transform=ax.transAxes,
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.7, edgecolor='none'))


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
    n_total = len(embedding_2d)
    title_with_n = f'{title} ({n_total})'

    fig, ax = plt.subplots(figsize=(10, 8))

    if colouring in ('initial', 'morphology'):
        label_start, label_end = LABEL_RANGES[colouring]
        type_labels = labels_full[:, label_start:label_end]
        colors, class_names = get_pure_class_colors(type_labels, colouring, CLASS_NAMES)
        _plot_umap_ax(ax, embedding_2d, colors, class_names, colouring,
                      type_labels=type_labels, title=title_with_n)

    elif colouring == 'train_labelled':
        _split_colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
        _color_idx = 0
        if split_masks:
            for split_name, mask in split_masks.items():
                n = int(mask.sum())
                if 'unlabelled' in split_name.lower():
                    color = 'grey'
                    alpha = 0.4
                else:
                    color = _split_colors[_color_idx % len(_split_colors)]
                    alpha = 0.7
                    _color_idx += 1
                ax.scatter(
                    embedding_2d[mask, 0], embedding_2d[mask, 1],
                    c=color, s=10, alpha=alpha, label=f'{split_name} ({n})',
                )
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.text(0.5, 0.97, title_with_n, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.7, edgecolor='none'))
        ax.legend(fontsize=13)
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
        _plot_umap_ax(ax, embedding_2d, colors, class_names, class_type,
                      type_labels=type_labels)

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
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
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

    img_col_width = 3.5
    fig_height = img_col_width * n_regions
    # subplots_adjust margins — used both here and at the bottom
    _L, _R, _T, _B = 0.02, 0.99, 0.95, 0.04
    # umap_ratio derived so the UMAP panel is 1.5× as wide as it is tall after margins
    umap_ratio = (_T - _B) * n_regions / (_R - _L) * 1.0
    fig_width = img_col_width * (n_per_region + umap_ratio)

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [1] * n_per_region + [umap_ratio]},
    )
    fig.suptitle('UMAP Region Samples', fontsize=16)

    ax_umap = axd['umap']
    ax_umap.set_aspect('equal', adjustable='box')
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
            ax_img.text(0.03, 0.97, class_str, transform=ax_img.transAxes,
                        fontsize=11, va='top', ha='left', color='white',
                        clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='grey',
                                  alpha=0.6, edgecolor='none'))
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax_img.axis('off')

            ax_umap.scatter(train_2d[idx, 0], train_2d[idx, 1],
                            c=color, s=90, marker='*', zorder=3,
                            edgecolors='black', linewidths=0.4)

        label = region_labels[r] if r < len(region_labels) else f'Region {r + 1}'
        ax_umap.scatter([], [], c=color, s=90, marker='*', label=label)

    ax_umap.set_xlabel('UMAP 1', fontsize=13)
    ax_umap.set_ylabel('UMAP 2', fontsize=13)
    ax_umap.set_title('UMAP Regions', fontsize=14)
    ax_umap.tick_params(labelsize=12)
    ax_umap.legend(fontsize=12)
    ax_umap.grid(True, alpha=0.3)

    plt.subplots_adjust(left=_L, right=_R, top=_T, bottom=_B, hspace=0.02, wspace=0.0)
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
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    FS   = 13
    FS_T = 14
    FS_S = 18

    n_epochs = len(history['train_loss'])
    epochs   = range(1, n_epochs + 1)

    monitor_raw   = history.get('monitor_val_loss')
    monitor_valid = (monitor_raw is not None
                     and any(v is not None for v in monitor_raw))
    monitor = monitor_raw if monitor_valid else None

    sched       = history.get('supervision_schedule')
    sched_label = 'Supervision Weight' if loss_mode == 'both' else 'Pairing Probability'

    val_aug = history.get('val_aug_loss')
    val_fri = history.get('val_friend_loss')
    tr_aug  = history.get('train_aug_loss')
    tr_fri  = history.get('train_friend_loss')

    has_zoom = n_epochs > 10

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _add_best(ax, fs):
        ax.axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)

    def _draw_loss_ax(ax, ep, tr_l, va_l, tr_a, va_a, tr_f, va_f, mon,
                      title, best_in_view=True, show_legend=True, fs=FS, fs_t=FS_T):
        ax.plot(ep, tr_l, 'b-',  linewidth=2)
        ax.plot(ep, va_l, 'r-',  linewidth=2)
        if tr_a:
            ax.plot(ep, tr_a, color='blue', linestyle=':', linewidth=1.5, alpha=0.8)
        if va_a:
            ax.plot(ep, va_a, color='red',  linestyle=':', linewidth=1.5, alpha=0.8)
        if tr_f:
            ax.plot(ep, tr_f, 'b--', linewidth=1.5, alpha=0.8)
        if va_f:
            ax.plot(ep, va_f, 'r--', linewidth=1.5, alpha=0.8)
        if mon:
            ax.plot(ep, mon, color='orange', linestyle='--', linewidth=2, alpha=0.85,
                    label='Val Monitor')
        ax.axhline(y=best_val_loss, color='g', linestyle='--', alpha=0.7)
        if best_in_view:
            _add_best(ax, fs)
        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('Loss', fontsize=fs)
        ax.set_title(f'{title} (best val: {best_val_loss:.4f}, ep {best_epoch})', fontsize=fs_t)
        ax.tick_params(labelsize=fs)
        ax.grid(True, alpha=0.3)
        if show_legend:
            # Split legend: colours (upper right) + linestyles (lower left)
            color_handles = [
                Line2D([0], [0], color='blue', linewidth=2, label='Train'),
                Line2D([0], [0], color='red',  linewidth=2, label='Val'),
            ]
            style_handles = [
                Line2D([0], [0], color='grey', linestyle='-',  linewidth=2,   label='Total'),
            ]
            if tr_a or va_a:
                style_handles.append(
                    Line2D([0], [0], color='grey', linestyle=':', linewidth=1.5, label='L_aug'))
            if tr_f or va_f:
                style_handles.append(
                    Line2D([0], [0], color='grey', linestyle='--', linewidth=1.5, label='L_friend'))
            leg1 = ax.legend(handles=color_handles, loc='upper right', fontsize=fs)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, loc='lower left', fontsize=fs)

    def _draw_overfit_ax(ax, ep, tr_l, va_l, tr_f, va_f, tr_a, va_a,
                         title, best_in_view=True, show_legend=True, fs=FS, fs_t=FS_T):
        _tr = np.array(tr_l, dtype=float)
        _va = np.array(va_l, dtype=float)
        ax.plot(ep, np.where(_tr != 0, _va / _tr, np.nan),
                color='#1f77b4', label='Val/Train total', linewidth=2)
        if tr_f and va_f:
            _tf = np.array(tr_f, dtype=float)
            _vf = np.array(va_f, dtype=float)
            ax.plot(ep, np.where(_tf != 0, _vf / _tf, np.nan),
                    color='#d62728', linestyle='--', label='Val/Train L_friend',
                    linewidth=1.5, alpha=0.85)
        if tr_a and va_a:
            _ta  = np.array(tr_a, dtype=float)
            _va2 = np.array(va_a, dtype=float)
            ax.plot(ep, np.where(_ta != 0, _va2 / _ta, np.nan),
                    color='#2ca02c', linestyle=':', label='Val/Train L_aug',
                    linewidth=1.5, alpha=0.85)
        ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1.2, alpha=0.6,
                   label='Ratio = 1')
        if best_in_view:
            _add_best(ax, fs)
        ax.set_xlabel('Epoch', fontsize=fs)
        ax.set_ylabel('Val / Train loss ratio', fontsize=fs)
        ax.set_title(title, fontsize=fs_t)
        if show_legend:
            ax.legend(fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Main 2×2 figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Training History - {model_type.upper()} Model', fontsize=FS_S)
    outer_gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    ax00 = fig.add_subplot(outer_gs[0, 0])
    ax01 = fig.add_subplot(outer_gs[0, 1])
    ax11 = fig.add_subplot(outer_gs[1, 1])

    # [0,0] all loss components
    _draw_loss_ax(ax00, epochs,
                  history['train_loss'], history['val_loss'],
                  tr_aug, val_aug, tr_fri, val_fri, monitor,
                  'All Loss Components')

    # [0,1] overfitting indicator
    _draw_overfit_ax(ax01, epochs,
                     history['train_loss'], history['val_loss'],
                     tr_fri, val_fri, tr_aug, val_aug,
                     'Overfitting Indicator')

    # [1,1] learning rate + supervision schedule on dual y-axes
    _lr_color = 'darkorange'
    _sw_color = 'green'
    ax11.plot(epochs, history['lr'], color=_lr_color, linewidth=2, label='Learning Rate')
    ax11.set_yscale('log')
    ax11.set_xlabel('Epoch', fontsize=FS)
    ax11.set_ylabel('Learning Rate', fontsize=FS, color=_lr_color)
    ax11.tick_params(axis='y', labelcolor=_lr_color, labelsize=FS)
    ax11.tick_params(axis='x', labelsize=FS)
    ax11.set_title(f'LR & Supervision Schedule (ep {best_epoch})', fontsize=FS_T)
    ax11.grid(True, alpha=0.3)
    _add_best(ax11, FS)
    ax11r = ax11.twinx()
    ax11r.plot(epochs, sched if sched else [0] * n_epochs,
               color=_sw_color, linewidth=2, label=sched_label)
    ax11r.set_ylabel(sched_label, fontsize=FS, color=_sw_color)
    ax11r.tick_params(axis='y', labelcolor=_sw_color, labelsize=FS)
    _l1, _lb1 = ax11.get_legend_handles_labels()
    _l2, _lb2 = ax11r.get_legend_handles_labels()
    ax11.legend(_l1 + _l2, _lb1 + _lb2, fontsize=FS)

    # [1,0] zoomed panels embedded as nested gridspec
    if has_zoom:
        start_idx   = int(n_epochs * 0.8)
        epochs_zoom = list(epochs)[start_idx:]
        best_in_z   = best_epoch > start_idx

        def _sl(seq):
            return seq[start_idx:] if seq else None

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[1, 0], hspace=0.15)
        ax_z0 = fig.add_subplot(inner_gs[0])
        ax_z1 = fig.add_subplot(inner_gs[1], sharex=ax_z0)

        # Hide x-tick labels on top panel to save space
        plt.setp(ax_z0.get_xticklabels(), visible=False)

        _FZ   = FS - 2
        _FZ_T = FS_T - 2
        _draw_loss_ax(ax_z0, epochs_zoom,
                      _sl(history['train_loss']), _sl(history['val_loss']),
                      _sl(tr_aug), _sl(val_aug), _sl(tr_fri), _sl(val_fri),
                      _sl(monitor), 'Loss (Zoomed)',
                      best_in_view=best_in_z, show_legend=False, fs=_FZ, fs_t=_FZ_T)
        ax_z0.set_xlabel('')
        _draw_overfit_ax(ax_z1, epochs_zoom,
                         _sl(history['train_loss']), _sl(history['val_loss']),
                         _sl(tr_fri), _sl(val_fri), _sl(tr_aug), _sl(val_aug),
                         '',
                         best_in_view=best_in_z, show_legend=False, fs=_FZ, fs_t=_FZ_T)

    plt.savefig(output_dir / f'training_curves{suffix}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {output_dir / f'training_curves{suffix}.png'}")
    plt.close(fig)
