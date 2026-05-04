"""
embed_and_umap.py

Extracts BYOL embeddings for one or more datasets and plots a UMAP.

The two steps are combined intentionally:
  - Embeddings are cached to disk so UMAP can be re-plotted without
    re-running the encoder.
  - Use --force to re-extract even if cached embeddings exist.

What "embedding" means here:
  We pass each image through the ONLINE encoder + projector:
      image (1,89,89) → encoder → 512-dim → projector → 256-dim
  The 256-dim projector output is saved as the embedding.
  This is the representation that BYOL directly optimises.

Usage examples:
    # Extract + plot, colour by dataset origin
    python embed_and_umap.py --checkpoint runs/byol_best.pt

    # Only specific datasets
    python embed_and_umap.py --checkpoint runs/byol_best.pt \
        --datasets mgcls_20k mirabest radio_galaxy_dataset

    # Colour by morphology label (labelled datasets only)
    python embed_and_umap.py --checkpoint runs/byol_best.pt \
        --colour_by label

    # Force re-extraction even if cache exists
    python embed_and_umap.py --checkpoint runs/byol_best.pt --force

Output files:
    embeddings/<run_id>/<dataset>_embeddings.npy
    embeddings/<run_id>/<dataset>_labels.npy
    embeddings/<run_id>/umap_<colour_by>_<timestamp>.png
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
from suplat.models.byol_models import BYOLEfficient
from suplat.data.eval_dataset import EvalDataset, DATASET_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE  = 128
NUM_WORKERS = 4
UMAP_SEED   = 42

# Colour palette for datasets (up to 6 datasets)
DATASET_COLOURS = [
    "#4C72B0",   # blue        — mgcls_20k
    "#55A868",   # green       — mgcls_5k
    "#C44E52",   # red         — mightee
    "#8172B2",   # purple      — mirabest
    "#CCB974",   # yellow      — radio_galaxy_dataset
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

def load_encoder(checkpoint_path, device):
    """
    Load a trained BYOLEfficient model and return it in eval mode.

    We extract embeddings from the ONLINE branch:
        online_encoder → online_projector
    The target network is not used here.

    Parameters
    ----------
    checkpoint_path : str
    device          : torch.device

    Returns
    -------
    encoder   : nn.Module  (online_encoder, outputs 512-dim)
    projector : nn.Module  (online_projector, outputs 256-dim)
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

    model = BYOLEfficient()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Return just the online encoder and projector
    encoder   = model.online_encoder
    projector = model.online_projector
    print(f"  Encoder output dim : 512")
    print(f"  Projector output dim: "
          f"{model.online_projector.net[-1].out_features}")
    return encoder, projector


# ---------------------------------------------------------------------------
# Step 2: Extract embeddings for one dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(encoder, projector, dataset, device):
    """
    Pass all images in dataset through encoder → projector.

    Returns
    -------
    embeddings : np.ndarray, shape (N, embed_dim)
    labels     : np.ndarray, shape (N,), dtype int  (-1 if unlabelled)
    """
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    all_labels     = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        z    = projector(encoder(imgs))   # (B, embed_dim)
        all_embeddings.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Step 3: Cache management
# ---------------------------------------------------------------------------

def embedding_paths(embed_dir, name):
    """Return (embeddings_path, labels_path) for a dataset."""
    return (
        os.path.join(embed_dir, f"{name}_embeddings.npy"),
        os.path.join(embed_dir, f"{name}_labels.npy"),
    )


def load_or_extract(name, encoder, projector, device,
                    embed_dir, force, root="."):
    """
    Load embeddings from cache if they exist, otherwise extract and save.

    Parameters
    ----------
    name      : str   dataset name
    force     : bool  if True, re-extract even if cache exists
    """
    emb_path, lbl_path = embedding_paths(embed_dir, name)
    cached = os.path.exists(emb_path) and os.path.exists(lbl_path)

    if cached and not force:
        print(f"  {name}: loading from cache")
        return np.load(emb_path), np.load(lbl_path)

    if cached and force:
        print(f"  {name}: --force set, re-extracting")
    else:
        print(f"  {name}: extracting embeddings")

    dataset    = EvalDataset(name, root=root)
    embeddings, labels = extract_embeddings(encoder, projector,
                                            dataset, device)

    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)
    print(f"    {len(embeddings)} embeddings saved → {embed_dir}/")
    return embeddings, labels


# ---------------------------------------------------------------------------
# Step 4: UMAP
# ---------------------------------------------------------------------------

def run_umap(all_embeddings, seed=UMAP_SEED):
    """
    Fit UMAP on the concatenated embeddings from all datasets.

    All datasets are fitted together so relative positions are meaningful —
    i.e. you can see whether MGCLS and MIGHTEE crops cluster together or
    apart, and where labelled morphologies fall in that space.
    """
    print(f"\nFitting UMAP on {len(all_embeddings)} points...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=15,      # default; balance local vs global structure
        min_dist=0.1,        # default; controls compactness of clusters
        metric="euclidean",
    )
    return reducer.fit_transform(all_embeddings)


# ---------------------------------------------------------------------------
# Step 5: Plot
# ---------------------------------------------------------------------------

def plot_umap(coords, dataset_names, embeddings_list, labels_list,
              colour_by, out_path):
    """
    Plot UMAP coordinates coloured by dataset origin or morphology label.

    Parameters
    ----------
    coords          : np.ndarray, shape (N, 2)
    dataset_names   : list of str
    embeddings_list : list of np.ndarray  (one per dataset, for size info)
    labels_list     : list of np.ndarray
    colour_by       : "dataset" or "label"
    out_path        : str
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
        for i, (name, embs) in enumerate(zip(dataset_names, embeddings_list)):
            n     = len(embs)
            colour = DATASET_COLOURS[i % len(DATASET_COLOURS)]
            point_colours.extend([colour] * n)
            legend_patches.append(
                mpatches.Patch(color=colour, label=f"{name} (n={n})")
            )
            offset += n

    elif colour_by == "label":
        for name, embs, labels in zip(dataset_names, embeddings_list,
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
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to trained BYOL .pt checkpoint")
    parser.add_argument("--datasets",    nargs="+",
                        default=list(DATASET_REGISTRY.keys()),
                        help="Dataset names to include. "
                             f"Available: {list(DATASET_REGISTRY.keys())}. "
                             "Default: all.")
    parser.add_argument("--colour_by",  choices=["dataset", "label"],
                        default="dataset",
                        help="Colour UMAP points by dataset origin or "
                             "morphology label (default: dataset)")
    parser.add_argument("--embed_dir",  default="embeddings",
                        help="Directory for cached embeddings "
                             "(default: embeddings/)")
    parser.add_argument("--output_dir", default="outputs/umap",
                        help="Directory for UMAP plot images "
                             "(default: outputs/umap/)")
    parser.add_argument("--force",      action="store_true",
                        help="Re-extract embeddings even if cached")
    parser.add_argument("--root",       default=".",
                        help="Project root directory (default: .)")
    parser.add_argument("--no_umap",    action="store_true",
                        help="Extract embeddings only, skip UMAP plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Derive a run_id from the checkpoint filename for namespacing cache
    run_id    = os.path.splitext(os.path.basename(args.checkpoint))[0]
    embed_dir = os.path.join(args.embed_dir, run_id)
    os.makedirs(embed_dir,      exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    encoder, projector = load_encoder(args.checkpoint, device)

    # --- Extract / load embeddings ---
    print("\nEmbeddings:")
    dataset_names   = []
    embeddings_list = []
    labels_list     = []

    for name in args.datasets:
        if name not in DATASET_REGISTRY:
            print(f"  WARNING: '{name}' not in registry — skipping")
            continue
        embs, lbls = load_or_extract(
            name, encoder, projector, device,
            embed_dir, args.force, root=args.root
        )
        dataset_names.append(name)
        embeddings_list.append(embs)
        labels_list.append(lbls)

    if not dataset_names:
        print("No valid datasets — exiting.")
        sys.exit(1)

    if args.no_umap:
        print("\nDone (--no_umap set, skipping plot).")
        return

    # --- UMAP ---
    all_embeddings = np.concatenate(embeddings_list)
    coords         = run_umap(all_embeddings)

    # --- Plot ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_name = f"umap_{args.colour_by}_{run_id}_{timestamp}.png"
    out_path  = os.path.join(args.output_dir, plot_name)

    print(f"\nPlotting ({args.colour_by})...")
    plot_umap(coords, dataset_names, embeddings_list,
              labels_list, args.colour_by, out_path)


if __name__ == "__main__":
    main()