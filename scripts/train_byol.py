#!/usr/bin/env python3
"""
BYOL Implementation for Radio Galaxy Classification
Training script with configurable projector modes.
Supports convnet, efficientnet-b0, resnet18, resnet50, and convnext-tiny architectures.

Projector modes:
  mlp  — Trainable MLP projector, EMA-updated with target network
  pca  — PCA projector re-fitted after each epoch (--projection-dim D for fixed D components, or 95% variance threshold if omitted)
  none — No projector; encoder output (1280-dim) fed directly into predictor
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import copy
import os
import sys as _sys
from datetime import datetime
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

from suplat.data.data_samplers import BYOLSupDataset, UnlabelledBYOLDataset, weights_closest, weights_ponderate
from suplat.data.augmentations import get_augmentation
from suplat.models.byol_models import (
    BYOLEfficient, BYOLEfficientNetB0,
    BYOLPretrainedBackbone,
    PCAProjection, PredictionHead,
    create_resnet18_backbone,
    create_resnet50_backbone,
    create_convnext_tiny_backbone,
)
from suplat.trainer.trainer import byol_loss, get_warmup_lr, get_supervision_weight, extract_embeddings_from_loader
from suplat.utils.plotting import fit_umap, plot_umap_single, plot_umap_outliers, plot_training_curves, plot_umap_scalar

# Check device availability
if torch.cuda.is_available():
    print("using gpu")
else:
    print("using cpu")

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    """Parse command-line arguments for BYOL training configuration"""
    ap = argparse.ArgumentParser(description="BYOL training with configurable projector modes")

    # Random seed
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="Random seed for data split (if default (None), uses --seed)")

    # Data configuration
    ap.add_argument("--data-dir", type=Path,
                    default=Path('/users/mbredber/p3_SUPLAT/data/preprocessed/lotss'),
                    help="Root directory containing images.npy and labels.npy")
    ap.add_argument("--dataset", type=str, default="LOTSS",
                    choices=["LOTSS", "MOCK"],
                    help="Dataset to use: LOTSS (real data) or MOCK (synthetic data)")

    # Label configuration
    ap.add_argument("--label-type", type=str, default="full",
                    choices=["full", "all", "classical", "initial", "morphology", "environment", "derived"],
                    help="Label subset to use: 'full'/'all' (all 20), 'classical' (0-1: FRI, FRII), "
                        "'initial' (0-4: FRI, FRII, Hybrids, Spirals, Relaxed doubles), "
                        "'morphology' (5-14: C-curve, S-curve, Misalignment, Wings, X-shaped, Straight jets, Multiple hotspots, "
                        "Continuous jets, Banding, One-sided, Restarted), 'environment' (15-18: Cluster, Merger, Diffuse emission, Unknown), "
                        "'derived' (19-23: Compact+hybrids, Hybrid FRI/FRII, Curved FRIs, Curved FRIIs, Straight+multi hotspots)")

    # Dataset pairing strategy
    ap.add_argument("--weighting", type=str, default="closest",
                    choices=["closest", "ponderate"],
                    help="Weight function for sampling pairs: 'closest' or 'ponderate' (default: closest)")
    ap.add_argument("--f-label", type=float, default=1.0,
                    help="Fraction of training split that receives labels (default: 1.0 = fully supervised)")
    ap.add_argument("--supervision-weight", type=float, default=1.0,
                    help="Weight for supervised pairing loss (default: 1.0). Only applicable if loss_mode is 'both'.")
    ap.add_argument("--supervision-weight-schedule", type=str, default="constant",
                    choices=["constant", "linear", "cosine"],
                    help="Curriculum schedule for supervision weight (default: constant)")
    ap.add_argument("--supervision-weight-start", type=float, default=0.0,
                    help="Starting supervision weight for scheduled curriculum (default: 0.0)")
    ap.add_argument("--supervision-weight-end", type=float, default=1.0,
                    help="Ending supervision weight for scheduled curriculum (default: 1.0)")

    # Data subsampling
    ap.add_argument("--subsample", type=int, default=None,
                    help="Subsample dataset to N samples (for quick testing)")

    # Augmentation
    ap.add_argument("--augmentation", type=str, default="standard",
                    choices=["standard", "extended"],
                    help="Augmentation pipeline: 'standard' (flip+rotate) or 'extended' (+ gaussian noise + intensity scaling)")

    # Model selection
    ap.add_argument("--model-type", type=str, default="efficientnet-b0",
                    choices=["efficientnet-b0", "convnet", "resnet18", "resnet50", "convnext-tiny"],
                    help="Model architecture: 'efficientnet-b0' (EfficientNet-B0, 1280-dim), 'convnet' (custom plain CNN, 512-dim), 'resnet18' (ResNet-18, 512-dim), 'resnet50' (ResNet-50, 2048-dim), 'convnext-tiny' (ConvNeXt-Tiny, 768-dim)")

    # Training hyperparameters
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for training (default: 32)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 0.0003)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs (default: 100)")
    # Gradient and optimization
    ap.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "step", "cosine"],
                    help="Learning rate schedule: 'constant', 'step' (step decay at 70%% of epochs, gamma=0.2), or 'cosine' (cosine annealing to 0) (default: constant)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Gradient clipping max norm (default: None, no clipping)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="L2 weight decay for Adam optimizer (default: 0.0)")
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout rate applied after the encoder (default: 0.2)")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Number of learning rate warmup epochs (default: 0)")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile the model (EfficientNet-B0 only; ~30s overhead, then faster)")
    # Projector configuration
    ap.add_argument("--projector", type=str, default="none",
                    choices=["mlp", "pca", "none"],
                    help="Projector type: 'none' (no projector, default), 'mlp' (learned MLP head), 'pca' (PCA re-fitted each epoch)")
    ap.add_argument("--pca-verbose", action="store_true", default=False,
                    help="Print PCA variance-explained and drift diagnostics each epoch (pca mode only, off by default)")
    ap.add_argument("--projection-dim", type=int, default=None,
                    help="Projection head output dimension; for pca mode, sets fixed n_components. "
                         "If omitted, pca mode uses 95%% variance threshold (default: None)")
    ap.add_argument("--hidden-dim", type=int, default=4096,
                    help="Hidden layer dimension in MLP heads (default: 4096)")

    # Output configuration
    ap.add_argument("--output-dir", type=Path,
                    default=Path('./outputs'),
                    help="Base output directory for checkpoints and embeddings")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Custom run name (default: timestamp)")
    # Visualization
    ap.add_argument("--no-plot-history", action="store_true",
                    help="Disable training curve plots (enabled by default)")

    # UMAP visualization
    ap.add_argument("--no-plot-umap", action="store_true",
                    help="Disable UMAP plots (enabled by default)")
    ap.add_argument("--umap-n-neighbors", type=int, default=30,
                    help="UMAP n_neighbors parameter (default: 30)")
    ap.add_argument("--umap-min-dist", type=float, default=0.1,
                    help="UMAP min_dist parameter (default: 0.1)")
    # DataLoader workers
    ap.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="Number of DataLoader worker processes (default: min(4, cpu_count))")

    # METRICS
    ap.add_argument("--no-metrics", action="store_true",
                    help="Disable projection clustering metrics (enabled by default)")
    ap.add_argument("--run-protege", action="store_true",
                    help="Run Protege GP active learning after training (off by default)")
    ap.add_argument("--no-protege-pca", action="store_false", dest="protege_pca",
                    help="Disable PCA dimensionality reduction before Protege GP (on by default)")

    return ap.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================
args = parse_args()

# Model hyperparameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_EPOCHS = args.epochs
EMA_DECAY = 0.99
PROJECTION_DIM = args.projection_dim
HIDDEN_DIM = args.hidden_dim
MODEL_TYPE = args.model_type

# Optimization hyperparameters
GRAD_CLIP = args.grad_clip
WEIGHT_DECAY = args.weight_decay
DROPOUT = args.dropout
LR_SCHEDULE = args.lr_schedule
WARMUP_EPOCHS = args.warmup_epochs
NUM_WORKERS = args.num_workers
PROJECTOR = args.projector
USE_COMPILE = args.compile
# Dataset configuration
DATASET_NAME = args.dataset
F_LABEL = args.f_label

# Data subsampling
SUBSAMPLE_SIZE = args.subsample

# Random seed
SEED = args.seed
DATA_SEED = args.data_seed if args.data_seed is not None else SEED
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Force CUDA if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)

    print(f"Using device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.cuda.empty_cache()

else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")
    print("  This will be VERY slow and may crash with large batches")

use_cuda = torch.cuda.is_available()

# Set explicit output directory
OUTPUT_BASE = args.output_dir
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Create run directory — include projector type to avoid collisions with train_byol.py
_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
if args.run_name:
    RUN_ID = f"{args.run_name}_{_timestamp}"
else:
    RUN_ID = _timestamp
    if DATASET_NAME != "LOTSS":
        RUN_ID += f"_{DATASET_NAME}"
    RUN_ID += f"_{MODEL_TYPE}_proj{PROJECTOR}_w{args.weighting}_f{F_LABEL}"

# Truncate labels based on label type
LABEL_RANGES = {
    'full':        (0, 20),
    'all':         (0, 20),
    'classical':   (0, 2),
    'initial':     (0, 5),
    'morphology':  (5, 16),
    'environment': (16, 20),
    'derived':     (19, 24),
}

OUTPUT_DIR = OUTPUT_BASE / f'run_{RUN_ID}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subfolders
FIGURES_DIR = OUTPUT_DIR / 'figures'
UMAP_DIR    = FIGURES_DIR / 'umap'
LOGS_DIR    = OUTPUT_DIR / 'logs'
DATA_DIR    = OUTPUT_DIR / 'data'
for _d in [FIGURES_DIR, UMAP_DIR, LOGS_DIR, DATA_DIR]:
    _d.mkdir(exist_ok=True)

checkpoint_path = OUTPUT_DIR / 'byol_model_best.pt'

# Tee stdout to logs/byol.log (SLURM also keeps its own copy)
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
_log_file = open(LOGS_DIR / 'byol.log', 'w')
_sys.stdout = _Tee(_sys.__stdout__, _log_file)

# Write configuration log
with open(LOGS_DIR / 'configuration_log.txt', 'w') as _cfg:
    _cfg.write(f"Run: {RUN_ID}\n")
    _cfg.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _cfg.write("=" * 50 + "\n")
    for key, val in sorted(vars(args).items()):
        _cfg.write(f"{key}: {val}\n")
label_dims = LABEL_RANGES[args.label_type][1] - LABEL_RANGES[args.label_type][0]

print(f"\n{'='*70}")
print("CONFIGURATION")
print(f"Output directory: {OUTPUT_DIR}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"{'='*70}")
print(f"Model type:     {MODEL_TYPE}")
print(f"Projector:      {PROJECTOR}")
if PROJECTOR == 'pca':
    if PROJECTION_DIM is not None:
        print(f"PCA components: {PROJECTION_DIM} (fixed, re-fitted each epoch)")
    else:
        print(f"PCA components: 95% variance threshold (re-fitted each epoch)")
elif PROJECTOR == 'mlp':
    print(f"Projection dim: {PROJECTION_DIM}")
print(f"Hidden dim:     {HIDDEN_DIM}")
print(f"Dataset:        {DATASET_NAME}")
print(f"Data dir:       {args.data_dir}")
print(f"Label type:     {args.label_type} ({label_dims} dims)")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Learning rate:  {LEARNING_RATE}")
print(f"Epochs:         {NUM_EPOCHS}")
print(f"Warmup epochs:  {WARMUP_EPOCHS}")
print(f"Grad clip:      {GRAD_CLIP if GRAD_CLIP else 'None'}")
print(f"EMA decay:      {EMA_DECAY}")
print(f"Weighting:      {args.weighting}")
print(f"F_label:        {F_LABEL}")
print(f"Num workers:    {NUM_WORKERS}")
print(f"Compile:        {'enabled' if USE_COMPILE else 'disabled'}")
print(f"Device:         {device}")
if SUBSAMPLE_SIZE:
    print(f"Subsampling:    {SUBSAMPLE_SIZE} samples")
print(f"{'='*70}\n")

# =============================================================================
# DATASET LOADING
# =============================================================================

# Data paths
IMAGES_PATH = args.data_dir / 'images_filtered.npy'
LABELS_PATH = args.data_dir / 'labels_filtered.npy'

print(f"Attempting to load {DATASET_NAME} data...")
print(f"  Images: {IMAGES_PATH}")
print(f"  Labels: {LABELS_PATH}")

# Check if files exist
if not IMAGES_PATH.exists():
    raise FileNotFoundError(f"Images file not found: {IMAGES_PATH}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

# Load data
images = np.load(IMAGES_PATH).astype(np.float32)/255
labels = np.load(LABELS_PATH)
labels_full = labels  # preserve all 20 columns before any label-type slicing

label_start, label_end = LABEL_RANGES[args.label_type]
if args.label_type != 'full':
    labels = labels[:, label_start:label_end]
    n_labels = label_end - label_start
    print(f"\nUsing {args.label_type} labels only (indices {label_start}-{label_end-1}, {n_labels} dimensions)")

# Validate
assert len(images) == len(labels), f"Mismatch: {len(images)} images, {len(labels)} labels"
assert images.ndim == 3, f"Expected 3D images, got {images.ndim}D: {images.shape}"
assert images.shape[1] == images.shape[2] == 89, f"Expected 89x89, got {images.shape[1:3]}"

# Subsample if requested
if SUBSAMPLE_SIZE is not None and len(images) > SUBSAMPLE_SIZE:
    print(f"\nSubsampling {SUBSAMPLE_SIZE}/{len(images)} samples")
    indices = np.random.choice(len(images), SUBSAMPLE_SIZE, replace=False)
    images = images[indices]
    labels = labels[indices]
    labels_full = labels_full[indices]

print("\nData loaded")
print(f"  Images: {images.shape} ({images.dtype})")
print(f"  Labels: {labels.shape} ({labels.dtype})")
print(f"  Range: [{images.min():.2f}, {images.max():.2f}]")

# =============================================================================
# LOSS MODE CONSTANTS
# =============================================================================
SUPERVISION_WEIGHT = args.supervision_weight
SUPERVISION_WEIGHT_SCHEDULE = args.supervision_weight_schedule
SUPERVISION_WEIGHT_START = args.supervision_weight_start
SUPERVISION_WEIGHT_END = args.supervision_weight_end
USE_CURRICULUM = SUPERVISION_WEIGHT_SCHEDULE != "constant"

# =============================================================================
# WEIGHTING, AUGMENTATION, AND HELPER FUNCTIONS
# =============================================================================

WEIGHTING_FUNC = weights_closest if args.weighting == "closest" else weights_ponderate
byol_strong_aug = get_augmentation(args.augmentation)


def _make_dataset_loader(img_data, label_data, shuffle, drop_last=False):
    """Create a BYOLSupDataset and DataLoader for a single split."""
    df = pd.DataFrame(label_data)
    ds = BYOLSupDataset(
        tags_data=df, img_data=img_data,
        transform=byol_strong_aug, friend_transform=byol_strong_aug,
        weightfunc=WEIGHTING_FUNC, p_pair_from_class=0.5
    )
    _nw = NUM_WORKERS if use_cuda else 0
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                        num_workers=_nw, pin_memory=use_cuda, drop_last=drop_last)
    return ds, loader


def byol_collate_fn(batch):
    x1     = torch.stack([b[0] for b in batch])
    x1_aug = torch.stack([b[1] for b in batch])
    is_labelled = torch.tensor([b[2] is not None for b in batch], dtype=torch.bool)
    if is_labelled.any():
        dummy = torch.zeros_like(batch[0][0])
        x2_friend = torch.stack([b[2] if b[2] is not None else dummy for b in batch])
    else:
        x2_friend = None
    return x1, x1_aug, x2_friend, is_labelled


def _monitor_loss_batch(fold_model, x1, x1_trans, x2_friend):
    """Fixed monitoring loss for one batch: both mode, supervision weight=1, curriculum-independent."""
    pred1_f, pred2_f, proj1_f, proj2_f = fold_model(x1, x2_friend)
    loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
    pred1_t, pred2_t, proj1_t, proj2_t = fold_model(x1, x1_trans)
    loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
    return (loss_trans + loss_friend).item()


def refit_pca(fold_model, extract_loader, device, hidden_dim, verbose=False):
    """
    Re-fit PCA projector after an epoch using the current online encoder's outputs.
    Rebuilds the predictor if the PCA output dimension changes (variance-threshold mode).
    """
    fold_model.eval()
    enc_outputs = []
    with torch.no_grad():
        for x1, _, _, _ in extract_loader:
            enc_outputs.append(fold_model.online_encoder(x1.to(device)).float().cpu())
    embeddings = torch.cat(enc_outputs, dim=0)

    old_dim = fold_model.online_projector.out_dim if fold_model.online_projector._fitted else None
    old_components = fold_model.online_projector.components_.cpu().clone() if fold_model.online_projector._fitted else None

    fold_model.online_projector.fit(embeddings)
    fold_model.online_projector.to(device)
    fold_model.target_projector = copy.deepcopy(fold_model.online_projector).to(device)
    for param in fold_model.target_projector.parameters():
        param.requires_grad = False

    new_dim = fold_model.online_projector.out_dim
    if new_dim != old_dim:
        fold_model.online_predictor = PredictionHead(
            in_dim=new_dim, hidden_dim=hidden_dim,
            out_dim=new_dim, bn_momentum=0.1,
        ).to(device)

    if verbose:
        var_exp = fold_model.online_projector.variance_explained_
        drift_str = ""
        if old_components is not None and new_dim == old_dim:
            drift = (fold_model.online_projector.components_.cpu() - old_components).norm().item()
            drift_str = f"  drift: {drift:.4f}"
        dim_str = f"  (dim: {old_dim} → {new_dim})" if new_dim != old_dim else ""
        print(f"  [PCA] Re-fitted: {new_dim} components  "
              f"var_explained: {var_exp:.1%}{drift_str}{dim_str}")

    fold_model.train()


def train_fold(train_loader, test_loader, extract_loader=None):
    """
    Train one model fold from scratch.
    extract_loader: DataLoader used to fit/re-fit PCA when PROJECTOR='pca'.
    Returns: (model, history, best_val_loss, best_epoch)
    """
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # -------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    if MODEL_TYPE == "efficientnet-b0":
        fold_model = BYOLEfficientNetB0(
            projection_dim=PROJECTION_DIM,
            hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1,
            feature_compression_mode=PROJECTOR,
            dropout_rate=DROPOUT,
        )
    elif MODEL_TYPE == "convnet":
        fold_model = BYOLEfficient(
            encoder_dim=512,
            projection_dim=PROJECTION_DIM,
            hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1,
            feature_compression_mode=PROJECTOR,
            dropout_rate=DROPOUT,
        )
    elif MODEL_TYPE == "resnet18":
        backbone, enc_dim = create_resnet18_backbone(dropout_rate=DROPOUT)
        fold_model = BYOLPretrainedBackbone(
            backbone, encoder_dim=enc_dim,
            projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1, feature_compression_mode=PROJECTOR,
        )
    elif MODEL_TYPE == "resnet50":
        backbone, enc_dim = create_resnet50_backbone(dropout_rate=DROPOUT)
        fold_model = BYOLPretrainedBackbone(
            backbone, encoder_dim=enc_dim,
            projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1, feature_compression_mode=PROJECTOR,
        )
    elif MODEL_TYPE == "convnext-tiny":
        backbone, enc_dim = create_convnext_tiny_backbone(dropout_rate=DROPOUT)
        fold_model = BYOLPretrainedBackbone(
            backbone, encoder_dim=enc_dim,
            projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1, feature_compression_mode=PROJECTOR,
        )
    fold_model = fold_model.to(device)

    # -------------------------------------------------------------------------
    # PCA PROJECTOR SETUP (pca mode only)
    # Do an initial fit and build the predictor. If --projection-dim is given,
    # uses fixed n_components; otherwise falls back to 95% variance threshold.
    # -------------------------------------------------------------------------
    if PROJECTOR == 'pca':
        assert extract_loader is not None, "extract_loader required for pca projector"

        if PROJECTION_DIM is not None:
            fold_model.online_projector = PCAProjection(n_components=PROJECTION_DIM)
            fold_model.target_projector = PCAProjection(n_components=PROJECTION_DIM)
        else:
            fold_model.online_projector = PCAProjection(variance_threshold=0.95)
            fold_model.target_projector = PCAProjection(variance_threshold=0.95)
        fold_model = fold_model.to(device)

        # Collect encoder outputs for initial fit
        fold_model.eval()
        _enc_outputs = []
        with torch.no_grad():
            for _x1, _, _, _ in extract_loader:
                _enc_outputs.append(fold_model.online_encoder(_x1.to(device)).float().cpu())
        _embeddings = torch.cat(_enc_outputs, dim=0)

        # Fit online projector and copy to target
        fold_model.online_projector.fit(_embeddings)
        fold_model.online_projector.to(device)
        fold_model.target_projector = copy.deepcopy(fold_model.online_projector).to(device)
        for param in fold_model.target_projector.parameters():
            param.requires_grad = False

        # Build predictor now that PCA output dim is known
        pca_out_dim = fold_model.online_projector.out_dim
        fold_model.online_predictor = PredictionHead(
            in_dim=pca_out_dim, hidden_dim=HIDDEN_DIM,
            out_dim=pca_out_dim, bn_momentum=0.1,
        ).to(device)

        fold_model.train()
        if args.pca_verbose:
            var_exp = fold_model.online_projector.variance_explained_
            print(f"PCA fitted (initial): {pca_out_dim} components  var_explained: {var_exp:.1%}")
        else:
            print(f"PCA fitted (initial): {pca_out_dim} components")

    if USE_COMPILE and MODEL_TYPE in ("efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
        print("Compiling model with torch.compile() ...")
        fold_model = torch.compile(fold_model, backend="cudagraphs")

    total_params = sum(p.numel() for p in fold_model.parameters())
    trainable_params = sum(p.numel() for p in fold_model.parameters() if p.requires_grad)
    print("\nInitializing model...")
    print(f"{'='*70}")
    print(f"MODEL ARCHITECTURE ({MODEL_TYPE.upper()})")
    print(f"{'='*70}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    _enc_dim_map = {"efficientnet-b0": 1280, "resnet18": 512, "resnet50": 2048, "convnext-tiny": 768}
    _enc_dim = _enc_dim_map.get(MODEL_TYPE, 512)
    print(f"Encoder output:       {_enc_dim}-dim representation")
    if PROJECTOR == 'mlp':
        print(f"Projector:            MLP -> {PROJECTION_DIM}-dim projection")
        print(f"Predictor output:     {PROJECTION_DIM}-dim prediction")
    elif PROJECTOR == 'pca':
        if PROJECTION_DIM is not None:
            print(f"Projector:            PCA (fixed {PROJECTION_DIM} components, re-fitted each epoch)")
        else:
            print(f"Projector:            PCA (95% variance threshold, re-fitted each epoch)")
        print(f"Predictor output:     {pca_out_dim}-dim prediction (initial)")
    else:
        print("Projector:            none (encoder -> predictor directly)")
        print(f"Predictor output:     {_enc_dim}-dim prediction")
    print(f"{'='*70}\n")

    if use_cuda:
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.0f} MB")

    # -------------------------------------------------------------------------
    # OPTIMIZER AND SCHEDULER
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(fold_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    _sched_epochs = max(NUM_EPOCHS - WARMUP_EPOCHS, 1)
    if LR_SCHEDULE == "step":
        milestone = int(0.7 * _sched_epochs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=0.2)
    elif LR_SCHEDULE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_sched_epochs, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    if LR_SCHEDULE == "step":
        print(f"Scheduler: step (milestone={int(0.7*_sched_epochs)} epochs, gamma=0.2)")
    elif LR_SCHEDULE == "cosine":
        print(f"Scheduler: cosine (T_max={_sched_epochs} epochs, eta_min=0)")
    else:
        print(f"Scheduler: constant")
    if WARMUP_EPOCHS > 0:
        print(f"Warmup: {WARMUP_EPOCHS} epochs")
    if GRAD_CLIP:
        print(f"Gradient clipping: max_norm={GRAD_CLIP}")
    print("Loss: BYOL symmetric MSE")

    _compute_val = test_loader is not None and F_LABEL > 0 and SUPERVISION_WEIGHT > 0

    history = {
        'train_loss': [],
        'train_aug_loss': [],
        'train_friend_loss': [],
        'monitor_val_loss': [],
        'lr': [],
        'supervision_schedule': [],
    }
    if _compute_val:
        history['val_friend_loss'] = []

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    for epoch in range(NUM_EPOCHS):
        # -------------------------------------------------------------------
        # LEARNING RATE WARMUP
        # -------------------------------------------------------------------
        if epoch < WARMUP_EPOCHS:
            current_lr = get_warmup_lr(epoch, LEARNING_RATE, WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        current_ema_decay = EMA_DECAY

        # -------------------------------------------------------------------
        # SUPERVISION WEIGHT AND PAIRING PROB SCHEDULING
        # -------------------------------------------------------------------
        current_supervision_weight = get_supervision_weight(
            epoch, NUM_EPOCHS,
            schedule=SUPERVISION_WEIGHT_SCHEDULE,
            base_weight=SUPERVISION_WEIGHT,
            start_weight=SUPERVISION_WEIGHT_START,
            end_weight=SUPERVISION_WEIGHT_END,
        )
        # -------------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------------
        fold_model.train()
        train_loss = 0.0
        train_aug_loss = 0.0
        train_friend_loss = 0.0
        train_friend_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for x1, x1_aug, x2_friend, is_labelled in pbar:
            x1, x1_aug = x1.to(device), x1_aug.to(device)

            # L_aug: ALL samples
            pred1_t, pred2_t, proj1_t, proj2_t = fold_model(x1, x1_aug)
            loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
            loss = loss_trans

            # L_friend: labelled samples only
            if x2_friend is not None and is_labelled.sum() >= 8:
                x1_lab = x1[is_labelled.to(device)]
                x2_lab = x2_friend[is_labelled].to(device)
                pred1_f, pred2_f, proj1_f, proj2_f = fold_model(x1_lab, x2_lab)
                loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                loss = loss + current_supervision_weight * loss_friend
                train_friend_loss += loss_friend.item()
                train_friend_batches += 1

            loss = loss / (1 + current_supervision_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), GRAD_CLIP)
            optimizer.step()

            fold_model.update_target_network(momentum=current_ema_decay)

            train_loss += loss.item()
            train_aug_loss += loss_trans.item()
            pbar.set_postfix({'aug': f'{loss_trans.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_aug_loss = train_aug_loss / len(train_loader)
        avg_train_friend_loss = train_friend_loss / train_friend_batches if train_friend_batches > 0 else 0.0

        # -------------------------------------------------------------------
        # PCA RE-FIT (pca mode only): update projector with current encoder
        # -------------------------------------------------------------------
        if PROJECTOR == 'pca':
            refit_pca(fold_model, extract_loader, device, hidden_dim=HIDDEN_DIM, verbose=args.pca_verbose)

        # -------------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------------
        if _compute_val:
            fold_model.eval()
            val_friend_loss = 0.0
            monitor_loss = 0.0

            with torch.no_grad():
                for x1, _x1_trans, x2_friend, is_labelled in test_loader:
                    x1 = x1.to(device)
                    x2_friend = x2_friend.to(device)
                    pred1_f, pred2_f, proj1_f, proj2_f = fold_model(x1, x2_friend)
                    val_friend_loss += byol_loss(pred1_f, pred2_f, proj1_f, proj2_f).item()

                    if USE_CURRICULUM:
                        x1_trans = _x1_trans.to(device)
                        monitor_loss += _monitor_loss_batch(fold_model, x1, x1_trans, x2_friend)

            avg_val_friend_loss = val_friend_loss / len(test_loader)
            avg_monitor_loss = monitor_loss / len(test_loader) if USE_CURRICULUM else None
            best_val_loss = avg_val_friend_loss

        else:
            avg_val_friend_loss = 0.0
            avg_monitor_loss = None
            best_val_loss = avg_train_aug_loss

        # Test images are in the training pool — always keep last epoch as best
        best_model_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
        best_epoch = epoch + 1

        # -------------------------------------------------------------------
        # LOGGING
        # -------------------------------------------------------------------
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['train_aug_loss'].append(avg_train_aug_loss)
        history['train_friend_loss'].append(avg_train_friend_loss)
        history['monitor_val_loss'].append(avg_monitor_loss)
        history['lr'].append(current_lr)
        history['supervision_schedule'].append(current_supervision_weight)
        if _compute_val:
            history['val_friend_loss'].append(avg_val_friend_loss)

        sup_str = f" | sup: {current_supervision_weight:.3f}"
        mon_str = f" | mon: {avg_monitor_loss:.4f}" if USE_CURRICULUM else ""
        _loss_str = f"t_aug: {avg_train_aug_loss:.4f}"
        if avg_train_friend_loss > 0 and _compute_val:
            _loss_str += f"  t_fri: {avg_train_friend_loss:.4f}  v_fri: {avg_val_friend_loss:.4f}"
        print(f"Epoch {epoch+1:>4}/{NUM_EPOCHS} | {_loss_str}{mon_str} | lr: {current_lr:.2e}{sup_str}")

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

    print(f"{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    loss_label = "Best monitor loss" if USE_CURRICULUM else "Best self-supervised val loss"
    print(f"{loss_label}: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    fold_model.load_state_dict(best_model_state)
    return fold_model, history, best_val_loss, best_epoch


def evaluate_test(eval_model, test_loader_ref):
    """Evaluate eval_model on held-out test set. Returns avg test loss."""
    final_sup_weight = get_supervision_weight(
        NUM_EPOCHS - 1, NUM_EPOCHS,
        schedule=SUPERVISION_WEIGHT_SCHEDULE,
        base_weight=SUPERVISION_WEIGHT,
        start_weight=SUPERVISION_WEIGHT_START,
        end_weight=SUPERVISION_WEIGHT_END,
    )
    eval_model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for x1, x1_trans, x2_friend, _ in tqdm(test_loader_ref, desc="Test"):
            x1, x1_trans, x2_friend = x1.to(device), x1_trans.to(device), x2_friend.to(device)
            pred1_f, pred2_f, proj1_f, proj2_f = eval_model(x1, x2_friend)
            loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
            pred1_t, pred2_t, proj1_t, proj2_t = eval_model(x1, x1_trans)
            loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
            test_loss_total += ((loss_trans + final_sup_weight * loss_friend)
                                / (1 + final_sup_weight)).item()
    return test_loss_total / len(test_loader_ref)


# =============================================================================
# DATA SPLIT, DATASETS, LOADERS, AND TRAINING
# =============================================================================

TRAIN_RATIO, TEST_RATIO = 0.70, 0.30

np.random.seed(DATA_SEED)

all_idx = np.arange(len(images))
print(f"\nSplitting data ({TRAIN_RATIO:.0%}/{TEST_RATIO:.0%})...")
train_idx, test_idx = train_test_split(all_idx, test_size=TEST_RATIO, random_state=DATA_SEED)
train_images, train_labels = images[train_idx], labels[train_idx]
test_images  = images[test_idx]
test_labels  = labels[test_idx]

# Labelled subset via stratified sampling
if F_LABEL == 0.0:
    labelled_mask = np.zeros(len(train_idx), dtype=bool)
elif F_LABEL >= 1.0:
    labelled_mask = np.ones(len(train_idx), dtype=bool)
else:
    strat_key = np.argmax(train_labels[:, :min(5, train_labels.shape[1])], axis=1)
    n_lab = max(2, int(round(F_LABEL * len(train_idx))))
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_lab, random_state=DATA_SEED)
        lab_rel, _ = next(sss.split(train_images, strat_key))
    except ValueError:
        print("Stratification failed, falling back to random selection")
        lab_rel = np.random.choice(len(train_idx), n_lab, replace=False)
    labelled_mask = np.zeros(len(train_idx), dtype=bool)
    labelled_mask[lab_rel] = True

labelled_images = train_images[labelled_mask]
labelled_labels = train_labels[labelled_mask]
unlabelled_images = train_images[~labelled_mask]
labelled_train_idx = train_idx[labelled_mask]

print(f"  Train BYOL total: {len(train_idx) + len(test_idx)} "
      f"({len(labelled_images)} labelled, {len(unlabelled_images)} unlabelled train, "
      f"{len(test_images)} test as unlabelled)")
print(f"  Test:  {len(test_images)}")

# Datasets
print("\nCreating datasets...")
print(f"  Augmentation: {args.augmentation}")
unlab_ds = UnlabelledBYOLDataset(unlabelled_images, transform=byol_strong_aug)

_nw = NUM_WORKERS if use_cuda else 0

if len(labelled_images) > 0:
    lab_df = pd.DataFrame(labelled_labels)
    lab_ds = BYOLSupDataset(tags_data=lab_df, img_data=labelled_images,
                             transform=byol_strong_aug, friend_transform=byol_strong_aug,
                             weightfunc=WEIGHTING_FUNC, p_pair_from_class=0.5)
    _train_combined = ConcatDataset([lab_ds, unlab_ds])
    train_extract_loader = DataLoader(lab_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=_nw, pin_memory=use_cuda)
else:
    lab_ds = None
    _train_combined = unlab_ds
    train_extract_loader = DataLoader(unlab_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=_nw, pin_memory=use_cuda,
                                       collate_fn=byol_collate_fn)

# Test images always enter the BYOL training pool regardless of F_LABEL or supervision_weight.
test_unlab_ds = UnlabelledBYOLDataset(test_images, transform=byol_strong_aug)
_train_combined = ConcatDataset([_train_combined, test_unlab_ds])

train_loader = DataLoader(_train_combined, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=_nw, pin_memory=use_cuda,
                          collate_fn=byol_collate_fn)
unlab_extract_loader = DataLoader(unlab_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=_nw, pin_memory=use_cuda,
                                   collate_fn=byol_collate_fn)
pca_fit_loader = DataLoader(_train_combined, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=_nw, pin_memory=use_cuda,
                             collate_fn=byol_collate_fn)

_, test_loader = _make_dataset_loader(test_images, test_labels, shuffle=False, drop_last=USE_COMPILE)

np.save(DATA_DIR / 'train_idx.npy',          train_idx)
np.save(DATA_DIR / 'labelled_train_idx.npy', labelled_train_idx)
np.save(DATA_DIR / 'test_idx.npy', test_idx)
unlabelled_train_idx = train_idx[~labelled_mask]
np.save(DATA_DIR / 'unlabelled_train_idx.npy', unlabelled_train_idx)

# Set train_labels / train_images for downstream to labelled-only
if len(labelled_images) > 0:
    train_labels = labelled_labels
    train_images = labelled_images
    train_idx    = labelled_train_idx

print(f"\n{'='*70}")
print("DATA LOADED")
print(f"{'='*70}")
print(f"Train: {len(train_loader)} batches x {BATCH_SIZE}")
print(f"Test:  {len(test_loader)} batches x {BATCH_SIZE}")
print(f"{'='*70}\n")

x1, x1_aug, x2_friend, is_labelled = next(iter(train_loader))
print(f"Test batch: {x1.shape}, {x1_aug.shape}")
print(f"  Labelled fraction: {is_labelled.float().mean():.2f}")

model, history, best_val_loss, best_epoch = train_fold(
    train_loader, test_loader, extract_loader=pca_fit_loader)

print("\nEvaluating on TEST set...")
avg_test_loss = evaluate_test(model, test_loader)
print(f"Test Loss: {avg_test_loss:.4f}  Best Val: {best_val_loss:.4f}")

# =============================================================================
# PROTEGE TIER CONSTANTS
# =============================================================================
LABEL_COLS = [
    "fri", "frii", "hybrid", "spiral", "relaxed",
    "cshaped", "sshaped", "misaligned", "wings", "xshaped",
    "straight", "multihotspots", "continuous", "banding", "onesided",
    "restarted", "cluster", "merger", "diffuse", "unknown",
]
SCORE_4 = ["xshaped", "unknown", "cluster", "merger"]
SCORE_3 = ["diffuse", "sshaped", "spiral"]
SCORE_2 = ["restarted", "onesided", "banding", "cshaped", "wings", "misaligned", "multihotspots", "relaxed"]
SCORE_1 = ["fri", "frii", "hybrid", "straight", "continuous"]
TIERS   = [(4, SCORE_4), (3, SCORE_3), (2, SCORE_2), (1, SCORE_1)]
POSITIVE_THRESHOLD   = 3
PROTEGE_INITIAL_STEPS = 10

# =============================================================================
# DOWNSTREAM: per-model loop
# =============================================================================

_items = [{'fold_idx': None, 'model': model, 'history': history,
           'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
           'avg_test_loss': avg_test_loss,
           'train_extract_loader': train_extract_loader,
           'train_labels': train_labels, 'train_idx': train_idx,
           'train_images': train_images}]

for _item in _items:
    _fi      = _item['fold_idx']
    _suffix  = "" if _fi is None else f"_fold{_fi + 1}"
    _label   = ""
    model                = _item['model']
    history              = _item['history']
    best_val_loss        = _item['best_val_loss']
    best_epoch           = _item['best_epoch']
    avg_test_loss        = _item['avg_test_loss']
    train_extract_loader = _item['train_extract_loader']
    train_labels         = _item['train_labels']
    train_idx            = _item['train_idx']
    train_images         = _item['train_images']

    history['test_loss'] = avg_test_loss

    # =========================================================================
    # SAVE MODEL AND HISTORY
    # =========================================================================
    _chk_path = OUTPUT_DIR / f'byol_model_best{_suffix}.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': None,
        'epoch': NUM_EPOCHS,
        'best_val_loss': best_val_loss,
        'test_loss': avg_test_loss,
        'history': history,
        'config': {
            'model_type': MODEL_TYPE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'grad_clip': GRAD_CLIP,
            'ema_decay': EMA_DECAY,
            'projection_dim': PROJECTION_DIM,
            'hidden_dim': HIDDEN_DIM,
            'encoder_dim': {"efficientnet-b0": 1280, "resnet18": 512, "resnet50": 2048, "convnext-tiny": 768}.get(MODEL_TYPE, 512),
            'weighting': args.weighting,
            'f_label': F_LABEL,
            'data_seed': DATA_SEED,
            'supervision_weight': SUPERVISION_WEIGHT,
            'supervision_weight_schedule': SUPERVISION_WEIGHT_SCHEDULE,
            'projector': PROJECTOR,
            'label_type': args.label_type,
        }
    }, _chk_path)

    print(f"Model checkpoint saved to {_chk_path}")

    np.save(DATA_DIR / f'training_history{_suffix}.npy', history)
    print(f"Training history saved to {DATA_DIR / f'training_history{_suffix}.npy'}")

    if not args.no_plot_history:
        print(f"\nGenerating training curve plots{_label}...")
        plot_training_curves(history, best_val_loss, best_epoch, MODEL_TYPE, FIGURES_DIR,
                             suffix=_suffix, loss_mode="both")

    # =========================================================================
    # EXTRACT PROJECTIONS
    # =========================================================================

    print(f"\nExtracting projections{_label}...")
    train_projections = extract_embeddings_from_loader(
        model, train_extract_loader, MODEL_TYPE, device, max_batches=None
    )
    print(f"   Train set projections: {train_projections.shape}")
    test_projections = extract_embeddings_from_loader(
        model, test_loader, MODEL_TYPE, device, max_batches=None
    )
    print(f"   Test set projections: {test_projections.shape}")

    np.save(DATA_DIR / f'labelled_train_projections{_suffix}.npy', train_projections)
    np.save(DATA_DIR / f'train_labels{_suffix}.npy', train_labels[:len(train_projections)])
    np.save(DATA_DIR / f'test_projections{_suffix}.npy', test_projections)
    np.save(DATA_DIR / f'test_labels{_suffix}.npy', test_labels[:len(test_projections)])

    if len(unlabelled_images) > 0 and len(labelled_images) > 0:
        unlab_projections = extract_embeddings_from_loader(
            model, unlab_extract_loader, MODEL_TYPE, device, max_batches=None
        )
        print(f"   Unlabelled train set projections: {unlab_projections.shape}")
        np.save(DATA_DIR / f'unlabelled_train_projections{_suffix}.npy', unlab_projections)
    else:
        unlab_projections = None

    print(f"\nProjections saved to {DATA_DIR}/")

    # =========================================================================
    # UMAP VISUALIZATIONS
    # =========================================================================
    if not args.no_plot_umap:
        print(f"\nGenerating UMAP visualizations{_label}...")

        CLASS_NAMES = {
            'initial':     ['FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles'],
            'morphology':  ['C-curvature', 'S-curvature', 'Misalignment', 'Wings', 'X-shaped',
                            'Straight jets', 'Multiple hotspots', 'Continuous jets', 'Banding',
                            'One-sided', 'Restarted'],
            'environment': ['Cluster', 'Merger', 'Diffuse emission', 'Unknown'],
            'derived':     ['Compact+hybrids', 'Hybrid FRI/FRII', 'Curved FRIs',
                            'Curved FRIIs', 'Straight+multi hotspots'],
        }

        _n_tr = len(train_projections)
        _n_te = len(test_projections)

        _lf_train = labels_full[train_idx][:_n_tr]
        _lf_test  = labels_full[test_idx][:_n_te]

        if unlab_projections is not None:
            _n_ul = len(unlab_projections)
            _lf_unlab = np.zeros((_n_ul, _lf_train.shape[1]), dtype=_lf_train.dtype)
            _parts = [unlab_projections, train_projections]
            _lf_parts = [_lf_unlab, _lf_train]
        else:
            _n_ul = 0
            _parts = [train_projections]
            _lf_parts = [_lf_train]
        _parts.append(test_projections)
        _lf_parts.append(_lf_test)
        _all_proj = np.concatenate(_parts)
        _all_lf   = np.concatenate(_lf_parts)

        _n_all = _n_ul + _n_tr + _n_te
        _mask_ul = np.zeros(_n_all, dtype=bool); _mask_ul[:_n_ul] = True
        _mask_tr = np.zeros(_n_all, dtype=bool); _mask_tr[_n_ul:_n_ul+_n_tr] = True
        _mask_te = np.zeros(_n_all, dtype=bool); _mask_te[_n_ul+_n_tr:] = True

        _, _all_2d = fit_umap(_all_proj, args.umap_n_neighbors, args.umap_min_dist, SEED)
        np.save(DATA_DIR / f'umap_all_coords{_suffix}.npy', _all_2d)

        _split_masks_all = {}
        if _n_ul > 0:
            _split_masks_all['Unlabelled train'] = _mask_ul
        _tr_key = 'Labelled train' if len(labelled_images) > 0 else 'Unlabelled train'
        _split_masks_all[_tr_key] = _mask_tr
        _split_masks_all['Test'] = _mask_te
        for _col in ('initial', 'morphology', 'train_labelled'):
            plot_umap_single(
                _all_2d, _all_lf, _col, CLASS_NAMES, LABEL_RANGES,
                title=f'All — {_col}',
                save_path=UMAP_DIR / f'umap_all_{_col}{_suffix}.png',
                split_masks=_split_masks_all,
            )

        _img_parts = []
        if unlab_projections is not None:
            _img_parts.append(unlabelled_images[:_n_ul])
        _img_parts.append(train_images[:_n_tr])
        _img_parts.append(test_images[:_n_te])
        _all_images_arr = np.concatenate(_img_parts, axis=0)

        _all_pixel_sum = _all_images_arr.sum(axis=(1, 2))
        plot_umap_scalar(
            _all_2d, _all_pixel_sum,
            title='All — brightness',
            cbar_label='Total pixel sum',
            save_path=UMAP_DIR / f'umap_all_brightness{_suffix}.png',
            cmap='plasma',
        )

        _lf_b    = _all_lf.astype(bool)
        _interest = np.ones(len(_all_lf), dtype=float)
        for _score, _cols in TIERS:
            _col_idx = [LABEL_COLS.index(c) for c in _cols]
            _interest[_lf_b[:, _col_idx].any(axis=1)] = _score
        plot_umap_scalar(
            _all_2d, _interest,
            title='All — interest score',
            cbar_label='Interest score',
            save_path=UMAP_DIR / f'umap_all_interest{_suffix}.png',
            cmap='plasma',
            vmin=1, vmax=4,
            cbar_ticks=[1, 2, 3, 4],
        )

        _, _test_2d = fit_umap(test_projections, args.umap_n_neighbors, args.umap_min_dist, SEED)
        np.save(DATA_DIR / f'umap_test_coords{_suffix}.npy', _test_2d)

        plot_umap_outliers(
            _all_2d[:_n_tr],
            train_images[:_n_tr],
            OUTPUT_DIR=UMAP_DIR,
            labels=train_labels[:_n_tr],
            save_prefix=f"umap_outliers{_suffix}",
        )

        print(f"\nUMAP plots saved to {UMAP_DIR}/")

    # =========================================================================
    # CLUSTERING METRICS
    # =========================================================================
    if not args.no_metrics:
        metrics = {}
        _metric_splits = [('train', train_projections, train_labels[:len(train_projections)]),
                          ('test', test_projections, test_labels[:len(test_projections)])]
        _all_m_projs  = []
        _all_m_labels = []
        if unlab_projections is not None:
            _all_m_projs.append(unlab_projections)
            _all_m_labels.append(np.zeros((len(unlab_projections), train_labels.shape[1]),
                                          dtype=train_labels.dtype))
        _all_m_projs.append(train_projections)
        _all_m_labels.append(train_labels[:len(train_projections)])
        _all_m_projs.append(test_projections)
        _all_m_labels.append(test_labels[:len(test_projections)])
        _metric_splits.append(('all', np.concatenate(_all_m_projs), np.concatenate(_all_m_labels)))
        for split, projections, split_labels in _metric_splits:
            metrics[split] = {'fri_vs_frii': {}, 'base_classes': {}}
            fri_only = (split_labels[:, 0] == 1) & (split_labels[:, 1] == 0)
            frii_only = (split_labels[:, 1] == 1) & (split_labels[:, 0] == 0)
            combined_fri_frii = fri_only | frii_only
            labels_hot = np.argmax(split_labels[combined_fri_frii][:, :2], axis=1)
            metrics[split]['fri_vs_frii'] = {
                'silhouette': float(silhouette_score(projections[combined_fri_frii], labels_hot)),
                'davies_bouldin': float(davies_bouldin_score(projections[combined_fri_frii], labels_hot)),
                'calinski_harabasz': float(calinski_harabasz_score(projections[combined_fri_frii], labels_hot))
            }
            fri_only = (split_labels[:, 0] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            frii_only = (split_labels[:, 1] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            all_hybrids = (split_labels[:, 2] == 1)
            spirals_only = (split_labels[:, 3] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            relaxed_doubles_only = (split_labels[:, 4] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            combined = fri_only | frii_only | all_hybrids | spirals_only | relaxed_doubles_only

            labels_hot = np.argmax(split_labels[combined][:, :5], axis=1)
            labels_hot[all_hybrids[combined]] = 2
            metrics[split]['base_classes'] = {
                'silhouette': float(silhouette_score(projections[combined], labels_hot)),
                'davies_bouldin': float(davies_bouldin_score(projections[combined], labels_hot)),
                'calinski_harabasz': float(calinski_harabasz_score(projections[combined], labels_hot))
            }

        with open(OUTPUT_DIR / f'projection_metrics{_suffix}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Projection clustering metrics saved to {OUTPUT_DIR / f'projection_metrics{_suffix}.json'}")

print(f"\n{'='*70}")
print("SCRIPT COMPLETE")
print(f"{'='*70}")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"{'='*70}\n")

# =============================================================================
# PROTEGE GP ACTIVE LEARNING  (--run-protege)
# =============================================================================
if args.run_protege:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from astronomaly.anomaly_detection import human_loop_learning, protege

    print(f"\n{'='*70}")
    print("PROTEGE GP ACTIVE LEARNING")
    print(f"{'='*70}\n")

    PROTEGE_DIR = OUTPUT_DIR / "protege"
    PROTEGE_DIR.mkdir(parents=True, exist_ok=True)

    lab_proj  = np.load(DATA_DIR / "labelled_train_projections.npy")
    lab_idx   = np.load(DATA_DIR / "labelled_train_idx.npy")
    unlab_proj_path = DATA_DIR / "unlabelled_train_projections.npy"
    if unlab_proj_path.exists():
        unlab_proj = np.load(unlab_proj_path)
        unlab_idx  = np.load(DATA_DIR / "unlabelled_train_idx.npy")
        cat_proj   = np.concatenate([lab_proj, unlab_proj], axis=0)
        cat_idx    = np.concatenate([lab_idx, unlab_idx])
    else:
        cat_proj = lab_proj
        cat_idx  = lab_idx
    test_proj_prot   = np.load(DATA_DIR / "test_projections.npy")
    test_idx_prot    = np.load(DATA_DIR / "test_idx.npy")

    _csv_df    = pd.read_csv("/users/mbredber/p3_SUPLAT/data/metadata/lotss_classifications_horton_et_al_2025_filtered.csv")
    _labels_all = np.load("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss/labels_filtered.npy")

    train_source_names = _csv_df.iloc[cat_idx]["Source_Name"].values
    test_source_names  = _csv_df.iloc[test_idx_prot]["Source_Name"].values

    _labels_train_df = pd.DataFrame(_labels_all[cat_idx].astype(bool), columns=LABEL_COLS)
    _human_train = np.ones(len(_labels_train_df), dtype=int)
    for _sv, _sc in TIERS:
        _human_train[_labels_train_df[_sc].any(axis=1).values] = _sv
    labels_df = pd.DataFrame({"human_label": _human_train}, index=train_source_names)

    _labels_test_df = pd.DataFrame(_labels_all[test_idx_prot].astype(bool), columns=LABEL_COLS)
    _human_test = np.ones(len(_labels_test_df), dtype=int)
    for _sv, _sc in TIERS:
        _human_test[_labels_test_df[_sc].any(axis=1).values] = _sv
    test_labels_df = pd.DataFrame({"human_label": _human_test}, index=test_source_names)

    np.random.seed(DATA_SEED)
    scaler      = StandardScaler()
    cat_scaled  = scaler.fit_transform(cat_proj)
    test_scaled = scaler.transform(test_proj_prot)
    if args.protege_pca:
        pca     = PCA(n_components=0.95, svd_solver="full", random_state=DATA_SEED)
        cat_pca = pca.fit_transform(cat_scaled)
        test_pca = pca.transform(test_scaled)
        print(f"  Protege PCA: {cat_proj.shape[1]}D -> {cat_pca.shape[1]}D")
    else:
        cat_pca  = cat_scaled
        test_pca = test_scaled
        print(f"  Protege features: {cat_pca.shape[1]}D (no PCA)")

    features_train = pd.DataFrame(cat_pca, index=train_source_names)
    features_test  = pd.DataFrame(test_pca, index=test_source_names)
    n_train = len(features_train)

    _sorted_order   = np.argsort(cat_pca[:, 0])
    _seed_positions = np.linspace(0, n_train - 1, PROTEGE_INITIAL_STEPS, dtype=int)
    _seed_rows      = _sorted_order[_seed_positions]
    seed_names      = features_train.index[_seed_rows]

    anomaly_scores = pd.DataFrame({"score": np.zeros(n_train)}, index=features_train.index)
    anomaly_scores.loc[seed_names, "score"] = labels_df.loc[seed_names, "human_label"].values.astype(float)

    score_converter = human_loop_learning.ScoreConverter(force_rerun=True, output_dir=str(PROTEGE_DIR))
    anomaly_scores  = score_converter.run(anomaly_scores)

    def _run_GP_exhaustive(features, labels, input_anomaly_scores, output_dir, steps=10, initial_steps=None, epsilon=0.5):
        _as = input_anomaly_scores.copy()
        _as['human_label'] = [-1] * len(_as)
        _gp = protege.GaussianProcess(features, force_rerun=True, output_dir=output_dir, ei_tradeoff=epsilon)
        if initial_steps is not None:
            _as.sort_values('score', ascending=False, inplace=True)
            _inds = _as[_as.human_label == -1].index[:initial_steps]
            _as.loc[_inds, 'human_label'] = labels.loc[_inds, 'human_label']
            _fwl = _gp.combine_data_frames(features, _as)
            _out = _gp.run(_fwl)
            _as['trained_score'] = _out['trained_score']
            _as['acquisition']   = _out['acquisition']
        while True:
            if _as[_as.human_label == -1].empty:
                break
            _as.sort_values('acquisition', ascending=False, inplace=True)
            _inds = _as[_as.human_label == -1].index[:steps]
            _as.loc[_inds, 'human_label'] = labels.loc[_inds, 'human_label']
            _fwl = _gp.combine_data_frames(features, _as)
            _out = _gp.run(_fwl)
            _as['trained_score'] = _out['trained_score']
            _as['acquisition']   = _out['acquisition']
        return _as, _gp

    print(f"  Running GP (initial_steps={PROTEGE_INITIAL_STEPS}, exhausting all {n_train} train sources)...")
    active_output, gp_pipeline = _run_GP_exhaustive(
        features_train, labels_df, anomaly_scores,
        output_dir=str(PROTEGE_DIR),
        steps=10,
        initial_steps=PROTEGE_INITIAL_STEPS,
        epsilon=0.0,
    )

    test_scores = gp_pipeline.estimator.predict(features_test.values)
    test_output = pd.DataFrame({"trained_score": test_scores, "human_label": -2}, index=features_test.index)

    true_labels = test_labels_df["human_label"]
    true_pos    = (true_labels >= POSITIVE_THRESHOLD).astype(int)
    n_pos  = int(true_pos.sum())
    n_eval = len(test_output)

    sorted_idx = test_output["trained_score"].sort_values(ascending=False).index
    sorted_pos = true_pos.loc[sorted_idx].values
    cum_found  = np.cumsum(sorted_pos)
    x = np.arange(1, n_eval + 1)
    auc = float(np.trapz(cum_found, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0
    print(f"  Test-set AUC={auc:.4f}  ({n_pos} positives in {n_eval} eval sources)")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, cum_found, label=f"Protege (AUC={auc:.4f})", linewidth=2)
    ax.plot(x, x * (n_pos / n_eval), "k--", label="Random baseline", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Sources inspected (ranked by Protege score)")
    ax.set_ylabel(f"Interesting sources found (label >= {POSITIVE_THRESHOLD})")
    ax.set_title(f"{OUTPUT_DIR.name}  (AUC={auc:.4f}, {n_pos} positives / {n_eval} eval)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PROTEGE_DIR / "recall_curve.png", dpi=120)
    plt.close(fig)

    combined_output = pd.concat([active_output, test_output])
    combined_output.to_parquet(PROTEGE_DIR / "protege_scores.parquet")

    summary = {
        "run_dir":          str(OUTPUT_DIR),
        "data_seed":        DATA_SEED,
        "n_labelled_seed":  PROTEGE_INITIAL_STEPS,
        "pca_seeded":       True,
        "steps":            10,
        "epsilon":          0.0,
        "n_eval":           n_eval,
        "n_eval_positives": n_pos,
        "auc":              auc,
        "pca_components":   int(cat_pca.shape[1]),
    }
    with open(PROTEGE_DIR / "protege_summary.json", "w") as _fh:
        json.dump(summary, _fh, indent=2)

    import logging as _logging, shutil as _shutil
    for _h in list(_logging.getLogger().handlers):
        if isinstance(_h, _logging.FileHandler) and 'astronomaly.log' in _h.baseFilename:
            _h.close()
            _logging.getLogger().removeHandler(_h)
            _shutil.move(_h.baseFilename, LOGS_DIR / 'protege.log')
            break

    print(f"\nProtege outputs saved to {PROTEGE_DIR}/")
    print(f"  recall_curve.png, protege_scores.parquet, protege_summary.json")
