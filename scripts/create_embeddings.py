#!/usr/bin/env python3
"""
BYOL Implementation for Radio Galaxy Classification
Training script for SLURM GPU submission
Supports both convnet, efficientnet-b0, and original (snippet-style) architectures
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
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

from suplat.data.data_samplers import BYOLSupDataset, weights_closest, weights_ponderate
from suplat.data.augmentations import get_augmentation
from suplat.models.byol_models import (
    BYOLEfficient, BYOLEfficientNetB0, BYOLOriginal, BYOLEncoder,
    BYOLPretrainedBackbone,
    create_resnet18_backbone,
    create_resnet50_backbone,
    create_convnext_tiny_backbone,
)
from suplat.trainer.trainer import byol_loss, get_warmup_lr, get_supervision_weight, extract_embeddings_from_loader
from suplat.utils.plotting import plot_umap_pure_classes, plot_umap_outliers, plot_training_curves

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
    ap = argparse.ArgumentParser(description="BYOL training for radio galaxy classification")
    
    # Random seed
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="Random seed for data split (if default (None), uses --seed)")
    
    # Data configuration
    ap.add_argument("--data-dir", type=Path, 
                    default=Path('./data/preprocessed/lotss/'),
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
    ap.add_argument("--loss-mode", type=str, default="both", choices=["either", "both"],
                    help="Whether to use 'either' (randomly choose friend or transformed) or 'both' (compute loss for both pairs) in BYOL loss")
    ap.add_argument("--prob", type=float, default=0.5,
                    help="Probability of pairing from same class (default: 0.5). Only applicable if loss_mode is 'either'.")
    ap.add_argument("--prob-schedule", type=str, default="constant",
                    choices=["constant", "linear", "cosine"],
                    help="Curriculum schedule for pairing probability (default: constant)")
    ap.add_argument("--prob-start", type=float, default=0.0,
                    help="Starting probability for scheduled curriculum (default: 0.0)")
    ap.add_argument("--prob-end", type=float, default=0.5,
                    help="Ending probability for scheduled curriculum (default: 0.5)")
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
                    choices=["efficientnet-b0", "convnet", "original", "resnet18", "resnet50", "convnext-tiny"],
                    help="Model architecture: 'efficientnet-b0' (EfficientNet-B0, 1280-dim), 'convnet' (custom plain CNN, 512-dim), 'original' (snippet-style NetWrapper), 'resnet18' (ResNet-18, 512-dim), 'resnet50' (ResNet-50, 2048-dim), 'convnext-tiny' (ConvNeXt-Tiny, 768-dim)")
    
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
    # Model architecture
    ap.add_argument("--feature-compression-mode", type=str, default="pca",
                    choices=["pca", "mlp", "none"],
                    help="Projector type: 'pca' (PCA keeping 95%% variance, default), 'mlp' (learned MLP head), 'none' (no projector)")
    ap.add_argument("--projection-dim", type=int, default=256,
                    help="Projection head output dimension (default: 256)")
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

    # Cross-validation
    ap.add_argument("--cv-folds", type=int, default=1,
                    help="Number of cross-validation folds (default: 1 = single train/val/test split)")

    # METRICS
    ap.add_argument("--no-metrics", action="store_true",
                    help="Disable projection clustering metrics (enabled by default)")

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
CV_FOLDS = args.cv_folds
FEATURE_COMPRESSION_MODE = args.feature_compression_mode
USE_COMPILE = args.compile
# Dataset configuration
DATASET_NAME = args.dataset
PROB_PAIR_FROM_CLASS = args.prob

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

    print(f"✓ Using device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.cuda.empty_cache()

else:
    device = torch.device('cpu')
    print("⚠ CUDA not available, using CPU")
    print("  This will be VERY slow and may crash with large batches")

use_cuda = torch.cuda.is_available()

# Set explicit output directory
OUTPUT_BASE = args.output_dir
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Create run directory
_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
if args.run_name:
    RUN_ID = f"{args.run_name}_{_timestamp}"
else:
    RUN_ID = _timestamp
    if DATASET_NAME != "LOTSS":
        RUN_ID += f"_{DATASET_NAME}"
    RUN_ID += f"_{MODEL_TYPE}_w{args.weighting}_p{PROB_PAIR_FROM_CLASS}"
    
# Truncate labels based on label type
LABEL_RANGES = {
    'full':        (0, 20),   # All labels
    'all':         (0, 20),   # Alias for full
    'classical':   (0, 2),    # FRI, FRII only
    'initial':     (0, 5),    # FRI, FRII, Hybrids, Spirals, Relaxed doubles
    'morphology':  (5, 15),   # C-curve through Restarted
    'environment': (15, 19),  # Cluster, Merger, Diffuse emission, Unknown
    'derived':     (19, 24),  # Compact+hybrids through Straight+multi hotspots
}

OUTPUT_DIR = OUTPUT_BASE / f'run_{RUN_ID}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subfolders
FIGURES_DIR    = OUTPUT_DIR / 'figures'
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'
LOGS_DIR       = OUTPUT_DIR / 'logs'
DATA_DIR       = OUTPUT_DIR / 'data'
for _d in [FIGURES_DIR, EMBEDDINGS_DIR, LOGS_DIR, DATA_DIR]:
    _d.mkdir(exist_ok=True)

checkpoint_path = OUTPUT_DIR / 'byol_model_best.pt'

# Tee stdout to logs/run.log (SLURM also keeps its own copy)
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
_log_file = open(LOGS_DIR / 'run.log', 'w')
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
print(f"Feature compression mode: {FEATURE_COMPRESSION_MODE}")
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
print(f"Pair prob:      {PROB_PAIR_FROM_CLASS}")
print(f"Num workers:    {NUM_WORKERS}")
print(f"CV folds:       {CV_FOLDS}")
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
    print(f"\n✓ Using {args.label_type} labels only (indices {label_start}-{label_end-1}, {n_labels} dimensions)")

# Validate
assert len(images) == len(labels), f"Mismatch: {len(images)} images, {len(labels)} labels"
assert images.ndim == 3, f"Expected 3D images, got {images.ndim}D: {images.shape}"
assert images.shape[1] == images.shape[2] == 89, f"Expected 89×89, got {images.shape[1:3]}"

# Subsample if requested
if SUBSAMPLE_SIZE is not None and len(images) > SUBSAMPLE_SIZE:
    print(f"\n⚠ Subsampling {SUBSAMPLE_SIZE}/{len(images)} samples")
    indices = np.random.choice(len(images), SUBSAMPLE_SIZE, replace=False)
    images = images[indices]
    labels = labels[indices]
    labels_full = labels_full[indices]

print("\n✓ Data loaded")
print(f"  Images: {images.shape} ({images.dtype})")
print(f"  Labels: {labels.shape} ({labels.dtype})")
print(f"  Range: [{images.min():.2f}, {images.max():.2f}]")

# =============================================================================
# LOSS MODE CONSTANTS
# =============================================================================
LOSS_MODE = args.loss_mode
SUPERVISION_WEIGHT = args.supervision_weight
SUPERVISION_WEIGHT_SCHEDULE = args.supervision_weight_schedule
SUPERVISION_WEIGHT_START = args.supervision_weight_start
SUPERVISION_WEIGHT_END = args.supervision_weight_end
PROB_SCHEDULE = args.prob_schedule
PROB_START = args.prob_start
PROB_END = args.prob_end
USE_CURRICULUM = SUPERVISION_WEIGHT_SCHEDULE != "constant" or PROB_SCHEDULE != "constant"

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
        weightfunc=WEIGHTING_FUNC, p_pair_from_class=PROB_PAIR_FROM_CLASS
    )
    _nw = NUM_WORKERS if use_cuda else 0
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                        num_workers=_nw, pin_memory=use_cuda, drop_last=drop_last)
    return ds, loader


def _monitor_loss_batch(fold_model, x1, x1_trans, x2_friend):
    """Fixed monitoring loss for one batch: both mode, supervision weight=1, curriculum-independent."""
    if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
        pred1_f, pred2_f, proj1_f, proj2_f = fold_model(x1, x2_friend)
        loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
        pred1_t, pred2_t, proj1_t, proj2_t = fold_model(x1, x1_trans)
        loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
    else:
        loss_friend = fold_model(torch.cat((x1, x2_friend), dim=0))
        loss_trans  = fold_model(torch.cat((x1, x1_trans),  dim=0))
    return (loss_trans + loss_friend).item()


def train_fold(train_loader, val_loader, extract_loader=None):
    """
    Train one model fold from scratch.
    When curriculum scheduling is active, model selection uses the monitoring val loss
    (both mode, supervision_weight=1, curriculum-independent). Otherwise, the regular
    val loss is used for model selection.
    extract_loader: DataLoader used to fit PCA when FEATURE_COMPRESSION_MODE='pca'.
    Returns: (model, history, best_val_loss, best_epoch)
    """
    # -------------------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    if MODEL_TYPE == "efficientnet-b0":
        fold_model = BYOLEfficientNetB0(
            projection_dim=PROJECTION_DIM,
            hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1,
            feature_compression_mode=FEATURE_COMPRESSION_MODE,
            dropout_rate=DROPOUT,
        )
    elif MODEL_TYPE == "convnet":
        fold_model = BYOLEfficient(
            encoder_dim=512,
            projection_dim=PROJECTION_DIM,
            hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1,
            feature_compression_mode=FEATURE_COMPRESSION_MODE,
            dropout_rate=DROPOUT,
        )
    elif MODEL_TYPE == "resnet18":
        backbone, enc_dim = create_resnet18_backbone(dropout_rate=DROPOUT)
        fold_model = BYOLPretrainedBackbone(
            backbone, encoder_dim=enc_dim,
            projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1, feature_compression_mode=FEATURE_COMPRESSION_MODE,
        )
    elif MODEL_TYPE == "resnet50":
        backbone, enc_dim = create_resnet50_backbone(dropout_rate=DROPOUT)
        fold_model = BYOLPretrainedBackbone(
            backbone, encoder_dim=enc_dim,
            projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1, feature_compression_mode=FEATURE_COMPRESSION_MODE,
        )
    elif MODEL_TYPE == "convnext-tiny":
        backbone, enc_dim = create_convnext_tiny_backbone(dropout_rate=DROPOUT)
        fold_model = BYOLPretrainedBackbone(
            backbone, encoder_dim=enc_dim,
            projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM,
            bn_momentum=0.1, feature_compression_mode=FEATURE_COMPRESSION_MODE,
        )
    else:
        enc = BYOLEncoder(bn_momentum=0.1)
        fold_model = BYOLOriginal(
            enc,
            image_size=89,
            projection_size=PROJECTION_DIM,
            projection_hidden_size=HIDDEN_DIM,
            moving_average_decay=EMA_DECAY,
            use_momentum=True,
            bn_momentum=0.1
        )
    fold_model = fold_model.to(device)

    # Fit PCA projector before training (requires one pass through data)
    if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny") and FEATURE_COMPRESSION_MODE == 'pca':
        assert extract_loader is not None, "extract_loader required for PCA fitting"
        fold_model.eval()
        _enc_outputs = []
        with torch.no_grad():
            for _x1, _, _, _ in extract_loader:
                _enc_outputs.append(fold_model.online_encoder(_x1.to(device)).float().cpu())
        fold_model.fit_pca(torch.cat(_enc_outputs, dim=0))
        fold_model = fold_model.to(device)
        print(f"✓ PCA fitted: {fold_model.online_projector.out_dim} components")

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
    if FEATURE_COMPRESSION_MODE == 'mlp':
        print(f"Projector:            MLP → {PROJECTION_DIM}-dim projection")
        print(f"Predictor output:     {PROJECTION_DIM}-dim prediction")
    elif FEATURE_COMPRESSION_MODE == 'pca':
        print("Projector:            PCA (fitted, 95% variance)")
        print("Predictor output:     PCA-dim prediction (set after fit_pca)")
    else:
        print("Projector:            none (encoder → predictor directly)")
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

    print(f"✓ Optimizer: Adam (lr={LEARNING_RATE})")
    if LR_SCHEDULE == "step":
        print(f"✓ Scheduler: step (milestone={int(0.7*_sched_epochs)} epochs, gamma=0.2)")
    elif LR_SCHEDULE == "cosine":
        print(f"✓ Scheduler: cosine (T_max={_sched_epochs} epochs, eta_min=0)")
    else:
        print(f"✓ Scheduler: constant")
    if WARMUP_EPOCHS > 0:
        print(f"✓ Warmup: {WARMUP_EPOCHS} epochs")
    if GRAD_CLIP:
        print(f"✓ Gradient clipping: max_norm={GRAD_CLIP}")
    print("✓ Loss: BYOL symmetric MSE")

    history = {
        'train_loss': [],
        'val_loss': [],
        'monitor_val_loss': [],
        'lr': [],
        'supervision_schedule': [],
    }

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
        current_prob = get_supervision_weight(
            epoch, NUM_EPOCHS,
            schedule=PROB_SCHEDULE,
            base_weight=PROB_PAIR_FROM_CLASS,
            start_weight=PROB_START,
            end_weight=PROB_END,
        )

        if MODEL_TYPE == "original":
            fold_model.target_ema_updater.beta = current_ema_decay

        # -------------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------------
        fold_model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for x1, x1_trans, x2_friend, _ in pbar:
            x1, x1_trans, x2_friend = x1.to(device), x1_trans.to(device), x2_friend.to(device)
            if LOSS_MODE == "either":
                u = (torch.rand(x1.size(0), device=device)
                     .unsqueeze(1).unsqueeze(2).unsqueeze(3)
                     .expand_as(x1))
                x2 = torch.where(u < current_prob, x2_friend, x1_trans)
                if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                    pred1, pred2, proj1, proj2 = fold_model(x1, x2)
                    loss = byol_loss(pred1, pred2, proj1, proj2)
                else:  # original
                    loss = fold_model(torch.cat((x1, x2), dim=0))
            else:  # "both"
                if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                    pred1_f, pred2_f, proj1_f, proj2_f = fold_model(x1, x2_friend)
                    loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                    pred1_t, pred2_t, proj1_t, proj2_t = fold_model(x1, x1_trans)
                    loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
                else:  # original
                    loss_friend = fold_model(torch.cat((x1, x2_friend), dim=0))
                    loss_trans  = fold_model(torch.cat((x1, x1_trans),  dim=0))
                loss = loss_trans + current_supervision_weight * loss_friend

            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), GRAD_CLIP)
            optimizer.step()

            if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                fold_model.update_target_network(momentum=current_ema_decay)
            else:  # original
                fold_model.update_moving_average()

            train_loss += loss.item()
            pbar.set_postfix({'train': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # -------------------------------------------------------------------
        # VALIDATION: scheduled loss + (if curriculum) monitoring loss
        # -------------------------------------------------------------------
        fold_model.eval()
        val_loss = 0.0
        monitor_loss = 0.0

        with torch.no_grad():
            for x1, x1_trans, x2_friend, _ in val_loader:
                x1, x1_trans, x2_friend = x1.to(device), x1_trans.to(device), x2_friend.to(device)

                # Scheduled val loss (mirrors training loss)
                if LOSS_MODE == "either":
                    u = (torch.rand(x1.size(0), device=device)
                         .unsqueeze(1).unsqueeze(2).unsqueeze(3)
                         .expand_as(x1))
                    x2 = torch.where(u < current_prob, x2_friend, x1_trans)
                    if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                        pred1, pred2, proj1, proj2 = fold_model(x1, x2)
                        val_loss += byol_loss(pred1, pred2, proj1, proj2).item()
                    else:  # original
                        val_loss += fold_model(torch.cat((x1, x2), dim=0)).item()
                else:  # "both"
                    if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                        pred1_f, pred2_f, proj1_f, proj2_f = fold_model(x1, x2_friend)
                        loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                        pred1_t, pred2_t, proj1_t, proj2_t = fold_model(x1, x1_trans)
                        loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
                    else:  # original
                        loss_friend = fold_model(torch.cat((x1, x2_friend), dim=0))
                        loss_trans  = fold_model(torch.cat((x1, x1_trans),  dim=0))
                    val_loss += (loss_trans + current_supervision_weight * loss_friend).item()

                # Monitoring loss: only computed when curriculum scheduling is active
                if USE_CURRICULUM:
                    monitor_loss += _monitor_loss_batch(fold_model, x1, x1_trans, x2_friend)

        avg_val_loss = val_loss / len(val_loader)
        avg_monitor_loss = monitor_loss / len(val_loader) if USE_CURRICULUM else None

        # -------------------------------------------------------------------
        # LOGGING
        # -------------------------------------------------------------------
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['monitor_val_loss'].append(avg_monitor_loss)
        history['lr'].append(current_lr)
        sched_val = current_supervision_weight if LOSS_MODE == "both" else current_prob
        history['supervision_schedule'].append(sched_val)

        # Model selection: use monitor loss when curriculum is active, otherwise val loss
        selection_loss = avg_monitor_loss if USE_CURRICULUM else avg_val_loss
        is_best = selection_loss < best_val_loss
        if is_best:
            best_val_loss = selection_loss
            best_model_state = copy.deepcopy(fold_model.state_dict())
            best_epoch = epoch + 1

        best_marker = ' ★' if is_best else ''
        sup_str = (f" | sup: {current_supervision_weight:.3f}" if LOSS_MODE == "both"
                   else f" | prob: {current_prob:.3f}")
        mon_str = f" | mon: {avg_monitor_loss:.4f}" if USE_CURRICULUM else ""
        print(f"Epoch {epoch+1:>4}/{NUM_EPOCHS} | train: {avg_train_loss:.4f}"
              f" | val: {avg_val_loss:.4f}{mon_str}"
              f" | lr: {current_lr:.2e}{sup_str}{best_marker}")

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

    print(f"{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    loss_label = "Best monitor loss" if USE_CURRICULUM else "Best val loss"
    print(f"{loss_label}: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    fold_model.load_state_dict(best_model_state)
    return fold_model, history, best_val_loss, best_epoch


def evaluate_test(eval_model, test_loader_ref):
    """Evaluate eval_model on held-out test set. Returns avg test loss."""
    # Use fully-ramped supervision weight (as at the final training epoch)
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
            if LOSS_MODE == "either":
                u = (torch.rand(x1.size(0), device=device)
                     .unsqueeze(1).unsqueeze(2).unsqueeze(3)
                     .expand_as(x1))
                x2 = torch.where(u < PROB_PAIR_FROM_CLASS, x2_friend, x1_trans)
                if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                    pred1, pred2, proj1, proj2 = eval_model(x1, x2)
                    test_loss_total += byol_loss(pred1, pred2, proj1, proj2).item()
                else:  # original
                    test_loss_total += eval_model(torch.cat((x1, x2), dim=0)).item()
            else:  # "both"
                if MODEL_TYPE in ("convnet", "efficientnet-b0", "resnet18", "resnet50", "convnext-tiny"):
                    pred1_f, pred2_f, proj1_f, proj2_f = eval_model(x1, x2_friend)
                    loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                    pred1_t, pred2_t, proj1_t, proj2_t = eval_model(x1, x1_trans)
                    loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
                else:  # original
                    loss_friend = eval_model(torch.cat((x1, x2_friend), dim=0))
                    loss_trans  = eval_model(torch.cat((x1, x1_trans),  dim=0))
                test_loss_total += (loss_trans + final_sup_weight * loss_friend).item()
    return test_loss_total / len(test_loader_ref)


# =============================================================================
# DATA SPLIT, DATASETS, LOADERS, AND TRAINING
# =============================================================================

if CV_FOLDS == 1:
    # -------------------------------------------------------------------------
    # SINGLE TRAIN/VAL/TEST SPLIT (default, backward compatible)
    # -------------------------------------------------------------------------
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    print(f"\nSplitting data ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")
    all_idx = np.arange(len(images))
    train_idx, temp_idx = train_test_split(all_idx, test_size=(VAL_RATIO + TEST_RATIO), random_state=DATA_SEED)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=DATA_SEED)

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    val_images   = images[val_idx]
    val_labels   = labels[val_idx]
    test_images  = images[test_idx]
    test_labels  = labels[test_idx]

    np.save(DATA_DIR / 'train_idx.npy', train_idx)
    np.save(DATA_DIR / 'val_idx.npy',   val_idx)
    np.save(DATA_DIR / 'test_idx.npy',  test_idx)

    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")

    print("\nCreating datasets...")
    print("  Converted labels to DataFrames")
    print(f"  Augmentation: {args.augmentation}")

    _, train_loader         = _make_dataset_loader(train_images, train_labels, shuffle=True,  drop_last=True)
    _, train_extract_loader = _make_dataset_loader(train_images, train_labels, shuffle=False)
    _, val_loader           = _make_dataset_loader(val_images,   val_labels,   shuffle=False, drop_last=USE_COMPILE)
    _, test_loader          = _make_dataset_loader(test_images,  test_labels,  shuffle=False, drop_last=USE_COMPILE)

    print(f"\n{'='*70}")
    print("✓ DATA LOADED")
    print(f"{'='*70}")
    print(f"Train: {len(train_loader)} batches × {BATCH_SIZE}")
    print(f"Val:   {len(val_loader)} batches × {BATCH_SIZE}")
    print(f"Test:  {len(test_loader)} batches × {BATCH_SIZE}")
    print(f"{'='*70}\n")

    x1, x1_trans, x2_friend, _ = next(iter(train_loader))
    print(f"✓ Test batch: {x1.shape}, {x1_trans.shape}, {x2_friend.shape}")
    print(f"  Different: {not torch.allclose(x1, x1_trans)}")

    model, history, best_val_loss, best_epoch = train_fold(train_loader, val_loader, extract_loader=train_extract_loader)

    print("\nEvaluating on TEST set (held-out)...")
    avg_test_loss = evaluate_test(model, test_loader)
    print(f"\n{'='*70}")
    print("TEST SET RESULTS (Best Model)")
    print(f"{'='*70}")
    print(f"Test Loss:  {avg_test_loss:.4f}")
    print(f"Best Val:   {best_val_loss:.4f}")
    print(f"Difference: {abs(avg_test_loss - best_val_loss):.4f}")
    print(f"{'='*70}\n")

else:
    # -------------------------------------------------------------------------
    # K-FOLD CROSS-VALIDATION
    # -------------------------------------------------------------------------
    print(f"\n{CV_FOLDS}-fold cross-validation")
    all_idx = np.arange(len(images))
    trainval_idx, test_idx = train_test_split(all_idx, test_size=0.15, random_state=DATA_SEED)

    test_images = images[test_idx]
    test_labels = labels[test_idx]

    np.save(DATA_DIR / 'test_idx.npy',      test_idx)
    np.save(DATA_DIR / 'trainval_idx.npy',  trainval_idx)

    print(f"  Test set: {len(test_images)} samples (constant across all folds)")
    print(f"  TrainVal: {len(trainval_idx)} samples (split {CV_FOLDS} ways)")
    print(f"  Augmentation: {args.augmentation}")

    _, test_loader = _make_dataset_loader(test_images, test_labels, shuffle=False)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=DATA_SEED) #unsure of data_seed vs seed here
    fold_results = []

    for fold_idx, (rel_train, rel_val) in enumerate(kf.split(trainval_idx)):
        actual_train_idx = trainval_idx[rel_train]
        actual_val_idx   = trainval_idx[rel_val]

        fold_train_images = images[actual_train_idx]
        fold_train_labels = labels[actual_train_idx]
        fold_val_images   = images[actual_val_idx]
        fold_val_labels   = labels[actual_val_idx]

        _, fold_train_loader         = _make_dataset_loader(fold_train_images, fold_train_labels, shuffle=True,  drop_last=True)
        _, fold_train_extract_loader = _make_dataset_loader(fold_train_images, fold_train_labels, shuffle=False)
        _, fold_val_loader           = _make_dataset_loader(fold_val_images,   fold_val_labels,   shuffle=False, drop_last=USE_COMPILE)

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{CV_FOLDS}  |  train={len(fold_train_images)}, val={len(fold_val_images)}")
        print(f"{'='*70}")
        print(f"Train: {len(fold_train_loader)} batches × {BATCH_SIZE}")
        print(f"Val:   {len(fold_val_loader)} batches × {BATCH_SIZE}\n")

        fold_model, fold_history, fold_best_val, fold_best_epoch = train_fold(
            fold_train_loader, fold_val_loader, extract_loader=fold_train_extract_loader
        )

        print(f"\nEvaluating fold {fold_idx+1} on test set...")
        fold_test_loss = evaluate_test(fold_model, test_loader)
        print(f"  Fold {fold_idx+1} test loss: {fold_test_loss:.4f}")

        fold_results.append({
            'model':                fold_model,
            'history':              fold_history,
            'best_val_loss':        fold_best_val,
            'test_loss':            fold_test_loss,
            'best_epoch':           fold_best_epoch,
            'train_idx':            actual_train_idx,
            'val_idx':              actual_val_idx,
            'train_extract_loader': fold_train_extract_loader,
            'val_loader':           fold_val_loader,
            'train_labels':         fold_train_labels,
            'val_labels':           fold_val_labels,
            'train_images':         fold_train_images,
        })

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Fold':>6} | {'Best Mon Loss':>13} | {'Test Loss':>9}")
    print("-" * 37)
    for i, r in enumerate(fold_results):
        print(f"{i+1:>6} | {r['best_val_loss']:>13.4f} | {r['test_loss']:>9.4f}")
    _cv_val_losses  = [r['best_val_loss'] for r in fold_results]
    _cv_test_losses = [r['test_loss']     for r in fold_results]
    print("-" * 37)
    print(f"{'Mean':>6} | {np.mean(_cv_val_losses):>13.4f} | {np.mean(_cv_test_losses):>9.4f}")
    print(f"{'Std':>6} | {np.std(_cv_val_losses):>13.4f} | {np.std(_cv_test_losses):>9.4f}")
    print(f"{'='*70}\n")

    _fold_results_meta = [
        {
            'fold_idx':      i,
            'best_val_loss': r['best_val_loss'],
            'test_loss':     r['test_loss'],
            'best_epoch':    r['best_epoch'],
            'train_idx':     r['train_idx'],
            'val_idx':       r['val_idx'],
        }
        for i, r in enumerate(fold_results)
    ]
    np.save(DATA_DIR / 'fold_results.npy', _fold_results_meta, allow_pickle=True)
    print(f"✓ Fold metadata saved to {DATA_DIR / 'fold_results.npy'}")

# =============================================================================
# DOWNSTREAM: per-model loop (single model for CV_FOLDS==1, N models for CV_FOLDS>1)
# =============================================================================

if CV_FOLDS == 1:
    _items = [{'fold_idx': None, 'model': model, 'history': history,
               'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
               'avg_test_loss': avg_test_loss,
               'train_extract_loader': train_extract_loader,
               'val_loader': val_loader, 'train_labels': train_labels,
               'val_labels': val_labels, 'train_idx': train_idx,
               'train_images': train_images}]
else:
    _items = [{'fold_idx': i, 'model': r['model'], 'history': r['history'],
               'best_val_loss': r['best_val_loss'], 'best_epoch': r['best_epoch'],
               'avg_test_loss': r['test_loss'],
               'train_extract_loader': r['train_extract_loader'],
               'val_loader': r['val_loader'], 'train_labels': r['train_labels'],
               'val_labels': r['val_labels'], 'train_idx': r['train_idx'],
               'train_images': r['train_images']}
              for i, r in enumerate(fold_results)]

for _item in _items:
    _fi      = _item['fold_idx']
    _suffix  = "" if _fi is None else f"_fold{_fi + 1}"
    _label   = "" if _fi is None else f" [Fold {_fi + 1}/{CV_FOLDS}]"
    model                = _item['model']
    history              = _item['history']
    best_val_loss        = _item['best_val_loss']
    best_epoch           = _item['best_epoch']
    avg_test_loss        = _item['avg_test_loss']
    train_extract_loader = _item['train_extract_loader']
    val_loader           = _item['val_loader']
    train_labels         = _item['train_labels']
    val_labels           = _item['val_labels']
    train_idx            = _item['train_idx']
    train_images         = _item['train_images']

    # Add to history
    history['test_loss'] = avg_test_loss

    # =========================================================================
    # SAVE MODEL AND HISTORY
    # =========================================================================
    _chk_path = OUTPUT_DIR / f'byol_model_best{_suffix}.pt'

    # Save model checkpoint
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
            'p_pair_from_class': PROB_PAIR_FROM_CLASS,
            'prob_schedule': PROB_SCHEDULE,
            'supervision_weight': SUPERVISION_WEIGHT,
            'supervision_weight_schedule': SUPERVISION_WEIGHT_SCHEDULE,
            'dataset': DATASET_NAME,
            'label_type': args.label_type,
        }
    }, _chk_path)

    print(f"✓ Model checkpoint saved to {_chk_path}")

    # Save training history
    np.save(DATA_DIR / f'training_history{_suffix}.npy', history)
    print(f"✓ Training history saved to {DATA_DIR / f'training_history{_suffix}.npy'}")

    # Plot training history (default behavior unless disabled)
    if not args.no_plot_history:
        print(f"\nGenerating training curve plots{_label}...")
        plot_training_curves(history, best_val_loss, best_epoch, MODEL_TYPE, FIGURES_DIR,
                             suffix=_suffix, loss_mode=LOSS_MODE)

    # =========================================================================
    # EXTRACT EMBEDDINGS
    # =========================================================================

    print(f"\nExtracting embeddings{_label}...")

    # Extract from train loader (no-shuffle for ordered alignment with images)
    print("\n  Train set:")
    train_projections = extract_embeddings_from_loader(
        model, train_extract_loader, MODEL_TYPE, device, max_batches=None
    )
    print(f"    Projections: {train_projections.shape}")

    # Extract from val loader
    print("\n  Val set:")
    val_projections = extract_embeddings_from_loader(
        model, val_loader, MODEL_TYPE, device, max_batches=None
    )
    print(f"    Projections: {val_projections.shape}")

    # Extract from test loader
    print("\n  Test set:")
    test_projections = extract_embeddings_from_loader(
        model, test_loader, MODEL_TYPE, device, max_batches=None
    )
    print(f"    Projections: {test_projections.shape}")

    # Save embeddings
    np.save(EMBEDDINGS_DIR / f'train_projections{_suffix}.npy', train_projections)
    np.save(EMBEDDINGS_DIR / f'val_projections{_suffix}.npy', val_projections)
    np.save(EMBEDDINGS_DIR / f'test_projections{_suffix}.npy', test_projections)

    # Save corresponding labels
    np.save(EMBEDDINGS_DIR / f'train_labels{_suffix}.npy', train_labels[:len(train_projections)])
    np.save(EMBEDDINGS_DIR / f'val_labels{_suffix}.npy', val_labels[:len(val_projections)])
    np.save(EMBEDDINGS_DIR / f'test_labels{_suffix}.npy', test_labels[:len(test_projections)])

    print(f"\n✓ Embeddings saved to {EMBEDDINGS_DIR}/")

    # Generate UMAP plots (default behavior unless disabled)
    if not args.no_plot_umap:
        print(f"\nGenerating UMAP visualizations{_label}...")

        # Define class names for each label type
        CLASS_NAMES = {
            'initial': ['FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles'],
            'morphology': ['C-curvature', 'S-curvature', 'Misalignment', 'Wings', 'X-shaped',
                          'Straight jets', 'Multiple hotspots', 'Continuous jets', 'Banding',
                          'One-sided', 'Restarted'],
            'environment': ['Cluster', 'Merger', 'Diffuse emission', 'Unknown'],
            'derived': ['Compact+hybrids', 'Hybrid FRI/FRII', 'Curved FRIs',
                       'Curved FRIIs', 'Straight+multi hotspots']
        }

        train_labels_full = labels_full[train_idx]
        test_labels_full = labels_full[test_idx]

        # Train UMAP: fit and save
        train_reducer, train_2d = plot_umap_pure_classes(
            train_projections,
            train_labels[:len(train_projections)],
            "Train (256-dim)",
            f"umap_train{_suffix}",
            "train",
            args=args,
            SEED=SEED,
            LABEL_RANGES=LABEL_RANGES,
            CLASS_NAMES=CLASS_NAMES,
            OUTPUT_DIR=FIGURES_DIR,
            train_labels_full=train_labels_full,
            test_labels_full=test_labels_full,
        )
        np.save(DATA_DIR / f'umap_train_coords{_suffix}.npy', train_2d)

        # Test UMAP: independent fit
        _, _test_2d = plot_umap_pure_classes(
            test_projections,
            test_labels[:len(test_projections)],
            "Test (256-dim)",
            f"umap_test{_suffix}",
            "test",
            args=args,
            SEED=SEED,
            LABEL_RANGES=LABEL_RANGES,
            CLASS_NAMES=CLASS_NAMES,
            OUTPUT_DIR=FIGURES_DIR,
            train_labels_full=train_labels_full,
            test_labels_full=test_labels_full,
        )
        np.save(DATA_DIR / f'umap_test_coords{_suffix}.npy', _test_2d)

        # Test UMAP: transformed into train space (fair comparison)
        _, _test_transformed_2d = plot_umap_pure_classes(
            test_projections,
            test_labels[:len(test_projections)],
            "Test in Train UMAP Space (256-dim)",
            f"umap_test_transformed{_suffix}",
            "test",
            args=args,
            SEED=SEED,
            LABEL_RANGES=LABEL_RANGES,
            CLASS_NAMES=CLASS_NAMES,
            OUTPUT_DIR=FIGURES_DIR,
            train_labels_full=train_labels_full,
            test_labels_full=test_labels_full,
            reducer=train_reducer,
        )
        np.save(DATA_DIR / f'umap_test_transformed_coords{_suffix}.npy', _test_transformed_2d)

        # Outlier plot: 4 most extreme points in train UMAP space
        plot_umap_outliers(
            train_2d,
            train_images[:len(train_2d)],
            OUTPUT_DIR=FIGURES_DIR,
            labels=train_labels[:len(train_2d)],
            save_prefix=f"umap_outliers{_suffix}",
        )

        print(f"\n✓ UMAP plots saved to {FIGURES_DIR}/")

    # Compute metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) for test and train projections
    if not args.no_metrics:
        # take the following cases :
        # - only FRI vs only FRII;
        # - all the base classes (FRI only, FRII only, all Hybrids, Spirals only, Relaxed doubles only) together

        metrics = {}
        for split, projections, split_labels in zip(
            ['train', 'test'],
            [train_projections, test_projections],
            [train_labels[:len(train_projections)], test_labels[:len(test_projections)]]
        ):
            metrics[split] = {'fri_vs_frii': {}, 'base_classes': {}}
            fri_only = (split_labels[:, 0] == 1) & (split_labels[:, 1] == 0)
            frii_only = (split_labels[:, 1] == 1) & (split_labels[:, 0] == 0)
            combined_fri_frii = fri_only | frii_only
            #the labels are in the format of a one-hot encoding, so we need to convert them to a single label for each class
            labels_hot = np.argmax(split_labels[combined_fri_frii][:, :2], axis=1)
            metrics[split]['fri_vs_frii'] = {
                'silhouette': silhouette_score(projections[combined_fri_frii], labels_hot).item(),
                'davies_bouldin': davies_bouldin_score(projections[combined_fri_frii], labels_hot).item(),
                'calinski_harabasz': calinski_harabasz_score(projections[combined_fri_frii], labels_hot).item()
            }
            fri_only = (split_labels[:, 0] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            frii_only = (split_labels[:, 1] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            all_hybrids = (split_labels[:, 2] == 1)
            spirals_only = (split_labels[:, 3] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            relaxed_doubles_only = (split_labels[:, 4] == 1) & (split_labels[:, :5].sum(axis=1) == 1)
            combined = fri_only | frii_only | all_hybrids | spirals_only | relaxed_doubles_only
            print("DEBUG", split_labels.shape, fri_only.shape, all_hybrids.shape, combined.shape)
            labels_hot = np.argmax(split_labels[combined][:, :5], axis=1)
            labels_hot[all_hybrids[combined]] = 2  # assign hybrid label (index 2) to all hybrids, even if they also have spiral or relaxed double labels
            metrics[split]['base_classes'] = {
                'silhouette': silhouette_score(projections[combined], labels_hot).item(),
                'davies_bouldin': davies_bouldin_score(projections[combined], labels_hot).item(),
                'calinski_harabasz': calinski_harabasz_score(projections[combined], labels_hot).item()
            }

        # and save to a json file
        with open(OUTPUT_DIR / f'projection_metrics{_suffix}.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Projection clustering metrics saved to {OUTPUT_DIR / f'projection_metrics{_suffix}.json'}")

print(f"\n{'='*70}")
print("SCRIPT COMPLETE")
print(f"{'='*70}")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"{'='*70}\n")