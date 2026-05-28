"""
embed_and_umap.py

Extracts BYOL projections for one or more datasets and plots a UMAP.

The two steps are combined intentionally:
  - Projections are cached to disk so UMAP can be re-plotted without
    re-running the encoder.
  - Use --force to re-extract even if cached projections exist.

Terminology:
  encoding   — output of the online encoder (e.g. 512-dim)
  projection — output of the projector MLP/PCA applied to the encoding
               (e.g. 256-dim); this is what BYOL directly optimises
  UMAP coordinates — 2-D layout produced by fitting UMAP on projections

Usage examples:
    # Extract + plot, colour by dataset origin
    python embed_and_umap.py --checkpoint runs/byol_best.pt

    # Only specific datasets
    python embed_and_umap.py --checkpoint runs/byol_best.pt \
        --datasets mgcls_20k mirabest first

    # Colour by morphology label (labelled datasets only)
    python embed_and_umap.py --checkpoint runs/byol_best.pt \
        --colour_by label

    # Force re-extraction even if cache exists
    python embed_and_umap.py --checkpoint runs/byol_best.pt --force

Output files:
    <checkpoint_dir>/projections/<dataset>_projections.npy
    <checkpoint_dir>/projections/<dataset>_labels.npy
    <checkpoint_dir>/figures/umap_<data_tag>_<colour_tag>.png
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

# Project imports — run from p3_SUPLAT root with the editable install active
from suplat.models.byol_models import (
    BYOLEfficient, BYOLEfficientNetB0, BYOLPretrainedBackbone, BYOLOriginal,
    create_resnet18_backbone, create_resnet50_backbone, create_convnext_tiny_backbone,
)
from suplat.data.eval_dataset import EvalDataset, DATASET_REGISTRY
from suplat.data.catalogue import Catalogue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE  = 128
NUM_WORKERS = 4
UMAP_SEED   = 42

# Mapping from model-type string (same as train_byol.py) to class + kwargs
MODEL_TYPE_MAP = {
    "convnet":        (BYOLEfficient,        {}),
    "efficientnet-b0":(BYOLEfficientNetB0,   {}),
    "original":       (BYOLOriginal,         {}),
    "resnet18":       (BYOLPretrainedBackbone, {"backbone": "resnet18"}),
    "resnet50":       (BYOLPretrainedBackbone, {"backbone": "resnet50"}),
    "convnext-tiny":  (BYOLPretrainedBackbone, {"backbone": "convnext-tiny"}),
}
MODEL_TYPE_CHOICES = list(MODEL_TYPE_MAP.keys())

# Colour palette for datasets (up to 6 datasets)
DATASET_COLOURS = [
    "#4C72B0",   # blue        — mgcls_20k
    "#55A868",   # green       — mgcls_5k
    "#C44E52",   # red         — mightee
    "#8172B2",   # purple      — mirabest
    "#CCB974",   # yellow      — first
    "#64B5CD",   # light blue  — mightee_fr
]

# Colour palette for morphology labels
LABEL_COLOURS = {
    0:  ("#4C72B0", "FRI"),
    1:  ("#C44E52", "FRII"),
    2:  ("#55A868", "Compact"),
    3:  ("#CCB974", "Bent"),
    -1: ("#CCCCCC", "Unlabelled"),
}

# ---------------------------------------------------------------------------
# Step 1: Load model
# ---------------------------------------------------------------------------

def _detect_model_type(checkpoint, checkpoint_path):
    """
    Determine model type from:
      1. checkpoint dict's 'config.model_type' key (saved by train_byol.py)
      2. checkpoint dict's top-level 'model_type' key (legacy format)
      3. heuristics on the checkpoint file path/name
    Returns a string matching MODEL_TYPE_MAP keys, or None.
    """
    if isinstance(checkpoint, dict):
        # Current format: nested under 'config'
        mt = checkpoint.get("config", {}).get("model_type")
        if mt and mt in MODEL_TYPE_MAP:
            return mt
        # Legacy format: top-level key
        mt = checkpoint.get("model_type")
        if mt:
            if mt in MODEL_TYPE_MAP:
                return mt
            print(f"  WARNING: checkpoint model_type='{mt}' not recognised; "
                  "falling back to path heuristics")

    # Path heuristics: look for substrings in the path
    path_lower = checkpoint_path.lower()
    for key in MODEL_TYPE_MAP:
        if key.replace("-", "_") in path_lower or key in path_lower:
            return key

    return None


def load_encoder(checkpoint_path, device, model_type=None):
    """
    Load a trained BYOL model and return it in eval mode.

    We extract embeddings from the ONLINE branch:
        online_encoder → online_projector
    The target network is not used here.

    Parameters
    ----------
    checkpoint_path : str
    device          : torch.device
    model_type      : str or None
        One of MODEL_TYPE_MAP keys.  If None, auto-detected from the
        checkpoint dict or path heuristics.  Falls back to 'convnet'.

    Returns
    -------
    encoder   : nn.Module  (online_encoder)
    projector : nn.Module  (online_projector)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both raw state_dict and wrapped checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Resolve model type
    if model_type is None:
        model_type = _detect_model_type(checkpoint, checkpoint_path)
    if model_type is None:
        print("  WARNING: could not detect model type; defaulting to 'convnet'")
        model_type = "convnet"
    print(f"  Model type: {model_type}")

    # Read feature_compression_mode from checkpoint; default 'pca' for back-compat
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    fcm         = cfg.get("feature_compression_mode", "pca")
    encoder_dim = cfg.get("encoder_dim", 512)
    proj_dim    = cfg.get("projection_dim", 256)
    hidden_dim  = cfg.get("hidden_dim", 4096)
    print(f"  Feature compression: {fcm}")

    _BACKBONE_CREATORS = {
        "resnet18":      create_resnet18_backbone,
        "resnet50":      create_resnet50_backbone,
        "convnext-tiny": create_convnext_tiny_backbone,
    }
    if model_type in _BACKBONE_CREATORS:
        backbone, _ = _BACKBONE_CREATORS[model_type]()
        model = BYOLPretrainedBackbone(
            backbone, encoder_dim=encoder_dim,
            projection_dim=proj_dim, hidden_dim=hidden_dim,
            feature_compression_mode=fcm,
        )
    else:
        model_cls, model_kwargs = MODEL_TYPE_MAP[model_type]
        if model_cls is BYOLEfficientNetB0:
            model_kwargs = {**model_kwargs, "feature_compression_mode": fcm}
        model = model_cls(**model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Return just the online encoder and projector
    encoder   = model.online_encoder
    projector = model.online_projector
    if hasattr(projector, 'net'):
        proj_out_dim = projector.net[-1].out_features
    else:
        try:
            proj_out_dim = projector.out_dim
        except (AttributeError, RuntimeError):
            proj_out_dim = '?'
    print(f"  Projector output dim: {proj_out_dim}")
    return encoder, projector


# ---------------------------------------------------------------------------
# Step 2: Extract projections for one dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_projections(encoder, projector, dataset, device,
                        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Pass all images in dataset through encoder → projector.

    Returns
    -------
    projections : np.ndarray, shape (N, proj_dim)
    labels      : np.ndarray, shape (N,), dtype int  (-1 if unlabelled)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_projections = []
    all_labels      = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        z    = projector(encoder(imgs))   # (B, proj_dim)
        all_projections.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_projections), np.concatenate(all_labels)


@torch.no_grad()
def extract_from_array(encoder, projector, images, device, batch_size=BATCH_SIZE):
    """
    Extract projections directly from a numpy array of images (N, H, W).

    Used by the catalogue path where images are already loaded into memory.
    """
    imgs_t = torch.from_numpy(images[:, None].astype(np.float32))
    all_projs = []
    for i in range(0, len(imgs_t), batch_size):
        batch = imgs_t[i:i + batch_size].to(device)
        all_projs.append(projector(encoder(batch)).cpu().numpy())
    return np.concatenate(all_projs)


# ---------------------------------------------------------------------------
# Step 3: Cache management
# ---------------------------------------------------------------------------

def projection_paths(proj_dir, name):
    """Return (projections_path, labels_path) for a dataset."""
    return (
        os.path.join(proj_dir, f"{name}_projections.npy"),
        os.path.join(proj_dir, f"{name}_labels.npy"),
    )


def load_or_extract(name, encoder, projector, device,
                    proj_dir, force, root=".",
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Load projections from cache if they exist, otherwise extract and save.

    Parameters
    ----------
    name      : str   dataset name
    force     : bool  if True, re-extract even if cache exists
    """
    proj_path, lbl_path = projection_paths(proj_dir, name)
    cached = os.path.exists(proj_path) and os.path.exists(lbl_path)

    if cached and not force:
        print(f"  {name}: loading from cache")
        return np.load(proj_path), np.load(lbl_path)

    if cached and force:
        print(f"  {name}: --force set, re-extracting")
    else:
        print(f"  {name}: extracting projections")

    dataset     = EvalDataset(name, root=root)
    projections, labels = extract_projections(
        encoder, projector, dataset, device,
        batch_size=batch_size, num_workers=num_workers,
    )

    np.save(proj_path, projections)
    np.save(lbl_path, labels)
    print(f"    {len(projections)} projections saved → {proj_dir}/")
    return projections, labels


# ---------------------------------------------------------------------------
# Step 4: UMAP
# ---------------------------------------------------------------------------

def run_umap(all_projections, seed=UMAP_SEED, n_neighbors=15, min_dist=0.1):
    """
    Fit UMAP on the concatenated projections from all datasets.

    All datasets are fitted together so relative positions are meaningful —
    i.e. you can see whether MGCLS and MIGHTEE crops cluster together or
    apart, and where labelled morphologies fall in that space.
    """
    print(f"\nFitting UMAP on {len(all_projections)} points "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
    )
    return reducer.fit_transform(all_projections)


# ---------------------------------------------------------------------------
# Step 5: Plot
# ---------------------------------------------------------------------------

def plot_umap(coords, dataset_names, projections_list, labels_list,
              colour_by, out_path):
    """
    Plot UMAP coordinates coloured by dataset origin or morphology label.

    Parameters
    ----------
    coords           : np.ndarray, shape (N, 2)
    dataset_names    : list of str
    projections_list : list of np.ndarray  (one per dataset, for size info)
    labels_list      : list of np.ndarray
    colour_by        : "dataset" or "label"
    out_path         : str
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)

    # Build per-point colour and legend arrays
    point_colours = []
    legend_patches = []
    offset = 0

    if colour_by == "dataset":
        for i, (name, embs) in enumerate(zip(dataset_names, projections_list)):
            n     = len(embs)
            colour = DATASET_COLOURS[i % len(DATASET_COLOURS)]
            point_colours.extend([colour] * n)
            legend_patches.append(
                mpatches.Patch(color=colour, label=f"{name} (n={n})")
            )
            offset += n

    elif colour_by == "label":
        for name, embs, labels in zip(dataset_names, projections_list,
                                      labels_list):
            for lbl in labels:
                colour, _ = LABEL_COLOURS.get(int(lbl), ("#CCCCCC", "?"))
                point_colours.append(colour)

        # Build legend from labels that actually appear
        seen = set()
        for labels in labels_list:
            seen.update(int(l) for l in labels)
        for lbl in sorted(seen):
            colour, name = LABEL_COLOURS.get(lbl, ("#CCCCCC", str(lbl)))
            legend_patches.append(mpatches.Patch(color=colour, label=name))

    # Scatter — unlabelled points smaller and more transparent
    sizes  = [4  if c != "#CCCCCC" else 2  for c in point_colours]
    alphas = [0.6 if c != "#CCCCCC" else 0.3 for c in point_colours]

    # Matplotlib scatter requires uniform alpha; plot in two passes
    mask_labelled   = np.array([c != "#CCCCCC" for c in point_colours])
    mask_unlabelled = ~mask_labelled

    if mask_unlabelled.any():
        ax.scatter(coords[mask_unlabelled, 0], coords[mask_unlabelled, 1],
                   c=[point_colours[i] for i in np.where(mask_unlabelled)[0]],
                   s=2, alpha=0.25, linewidths=0, rasterized=True)
    if mask_labelled.any():
        ax.scatter(coords[mask_labelled, 0], coords[mask_labelled, 1],
                   c=[point_colours[i] for i in np.where(mask_labelled)[0]],
                   s=5, alpha=0.7, linewidths=0, rasterized=True)

    ax.legend(handles=legend_patches, loc="best",
              fontsize=9, framealpha=0.8)
    title = ("UMAP coloured by " +
             ("dataset origin" if colour_by == "dataset" else "morphology label"))
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract BYOL embeddings and plot UMAP"
    )
    # --- Model ---
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to trained BYOL .pt checkpoint")
    parser.add_argument("--model-type",  dest="model_type",
                        choices=MODEL_TYPE_CHOICES, default=None,
                        help="BYOL architecture used during training. "
                             f"Choices: {MODEL_TYPE_CHOICES}. "
                             "Default: auto-detect from checkpoint or path.")
    # --- Data ---
    parser.add_argument("--catalogue",   default=None,
                        help="Path to a catalogue YAML. When provided, replaces --datasets.")
    parser.add_argument("--datasets",    nargs="+",
                        default=list(DATASET_REGISTRY.keys()),
                        help="Dataset names to include (ignored when --catalogue is used). "
                             f"Available: {list(DATASET_REGISTRY.keys())}. "
                             "Default: all.")
    parser.add_argument("--root",        default=".",
                        help="Project root directory (default: .)")
    parser.add_argument("--batch-size",  dest="batch_size", type=int,
                        default=BATCH_SIZE,
                        help=f"Batch size for embedding extraction "
                             f"(default: {BATCH_SIZE})")
    parser.add_argument("--num-workers", dest="num_workers", type=int,
                        default=NUM_WORKERS,
                        help=f"DataLoader worker processes "
                             f"(default: {NUM_WORKERS})")
    # --- Output ---
    parser.add_argument("--colour_by",   choices=["dataset", "label"],
                        default="dataset",
                        help="Colour UMAP points by dataset origin or "
                             "morphology label (default: dataset)")
    parser.add_argument("--proj_dir",    default=None,
                        help="Directory for cached projections. "
                             "Default: <checkpoint_dir>/projections/")
    parser.add_argument("--output_dir",  default=None,
                        help="Directory for UMAP plot images. "
                             "Default: <checkpoint_dir>/figures/")
    parser.add_argument("--force",       action="store_true",
                        help="Re-extract projections even if cached")
    parser.add_argument("--no_umap",     action="store_true",
                        help="Extract embeddings only, skip UMAP plot")
    # --- UMAP hyper-parameters ---
    parser.add_argument("--umap-n-neighbors", dest="umap_n_neighbors",
                        type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--umap-min-dist",    dest="umap_min_dist",
                        type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--umap-seed",        dest="umap_seed",
                        type=int, default=UMAP_SEED,
                        help=f"UMAP random seed (default: {UMAP_SEED})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Derive proj_dir from checkpoint directory unless overridden
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    proj_dir = args.proj_dir or os.path.join(checkpoint_dir, "projections")
    os.makedirs(proj_dir, exist_ok=True)

    # --- Load model ---
    encoder, projector = load_encoder(args.checkpoint, device,
                                      model_type=args.model_type)

    # --- Extract / load projections ---
    print("\nProjections:")
    dataset_names    = []
    projections_list = []
    labels_list      = []

    if args.catalogue:
        mat    = Catalogue.from_yaml(args.catalogue).materialise(root=args.root)
        splits = mat.get_split_datasets()

        # Set of dataset names that have labelled splits
        labelled_names = {sv.dataset for sv_list in splits.values() for sv in sv_list}

        for entry in mat._entries:
            name = entry.dataset

            if name in labelled_names:
                # Combine images/labels from all splits; record split membership
                split_code = {"train": 0, "val": 1, "test": 2}
                all_imgs, all_lbls_1d, all_split_ids = [], [], []
                for split_name, code in split_code.items():
                    for sv in splits[split_name]:
                        if sv.dataset != name:
                            continue
                        all_imgs.append(sv.images)
                        lbls = sv.labels
                        # Reduce multi-hot to 1-D for UMAP colouring
                        if lbls.ndim > 1:
                            lbls_1d = lbls.argmax(axis=1).astype(np.int64)
                        else:
                            lbls_1d = lbls.astype(np.int64)
                        all_lbls_1d.append(lbls_1d)
                        all_split_ids.append(
                            np.full(len(sv.images), code, dtype=np.int64)
                        )

                images_all = np.concatenate(all_imgs)
                lbls_all   = np.concatenate(all_lbls_1d)
                split_all  = np.concatenate(all_split_ids)

                proj_path = os.path.join(proj_dir, f"{name}_projections.npy")
                lbl_path  = os.path.join(proj_dir, f"{name}_labels.npy")
                cached = os.path.exists(proj_path) and os.path.exists(lbl_path)

                if cached and not args.force:
                    print(f"  {name}: loading from cache")
                    projs    = np.load(proj_path)
                    lbls_all = np.load(lbl_path)
                else:
                    if cached and args.force:
                        print(f"  {name}: --force set, re-extracting")
                    else:
                        print(f"  {name}: extracting projections")
                    projs = extract_from_array(
                        encoder, projector, images_all, device, args.batch_size
                    )
                    np.save(proj_path, projs)
                    np.save(lbl_path, lbls_all)
                    print(f"    {len(projs)} projections saved → {proj_dir}/")

                np.save(os.path.join(proj_dir, f"{name}_splits.npy"), split_all)

            else:
                # Unlabelled dataset: use existing EvalDataset path
                projs, lbls_all = load_or_extract(
                    name, encoder, projector, device,
                    proj_dir, args.force, root=args.root,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                )
                split_all = np.full(len(projs), -1, dtype=np.int64)
                np.save(os.path.join(proj_dir, f"{name}_splits.npy"), split_all)

            dataset_names.append(name)
            projections_list.append(projs)
            labels_list.append(lbls_all)

    else:
        for name in args.datasets:
            if name not in DATASET_REGISTRY:
                print(f"  WARNING: '{name}' not in registry — skipping")
                continue
            projs, lbls = load_or_extract(
                name, encoder, projector, device,
                proj_dir, args.force, root=args.root,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            dataset_names.append(name)
            projections_list.append(projs)
            labels_list.append(lbls)

    if not dataset_names:
        print("No valid datasets — exiting.")
        sys.exit(1)

    if args.no_umap:
        print("\nDone (--no_umap set, skipping plot).")
        return

    # --- UMAP ---
    all_projections = np.concatenate(projections_list)
    coords          = run_umap(all_projections,
                               seed=args.umap_seed,
                               n_neighbors=args.umap_n_neighbors,
                               min_dist=args.umap_min_dist)

    # --- Plot ---
    output_dir = args.output_dir or os.path.join(checkpoint_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    data_tag   = "multi" if len(dataset_names) > 1 else dataset_names[0]
    colour_tag = "datasets" if args.colour_by == "dataset" else "labels"
    plot_name  = f"umap_{data_tag}_{colour_tag}.png"
    out_path   = os.path.join(output_dir, plot_name)

    print(f"\nPlotting ({args.colour_by})...")
    plot_umap(coords, dataset_names, projections_list,
              labels_list, args.colour_by, out_path)


if __name__ == "__main__":
    main()