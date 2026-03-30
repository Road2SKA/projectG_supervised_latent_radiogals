#!/usr/bin/env python3
"""
BYOL Implementation for Radio Galaxy Classification
Training script for SLURM GPU submission
Supports both efficient and original (snippet-style) architectures
"""

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader #, Dataset
import torchvision.transforms as T
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import copy
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import umap

from suplat.data.data_samplers import BYOLSupDataset, weights_closest, weights_ponderate
from suplat.data.augmentations import get_augmentation
from suplat.models.byol_models import BYOLEfficient, BYOLOriginal, BYOLEncoder
from suplat.trainer.trainer import byol_loss, get_ema_decay, get_warmup_lr, get_supervision_weight, extract_embeddings_from_loader
from suplat.utils.plotting import plot_umap_pure_classes, plot_umap_overlay, plot_umap_outliers, plot_training_curves

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
    
    # Data configuration
    ap.add_argument("--data-dir", type=Path, 
                    default=Path('/users/mbredber/supervised_latent/'),
                    help="Root directory containing images.npy and labels.npy")
    ap.add_argument("--dataset", type=str, default="LOTSS",
                    choices=["LOTSS", "MOCK"],
                    help="Dataset to use: LOTSS (real data) or MOCK (synthetic data)")
    
    # Label configuration
    ap.add_argument("--label-type", type=str, default="full",
                    choices=["full", "initial", "morphology", "environment", "derived"],
                    help="Label subset to use: 'full' (all 20), 'initial' (0-4: FRI, FRII, Hybrids, Spirals, Relaxed doubles), "
                        "'morphology' (5-14: C-curve, S-curve, Misalignment, Wings, X-shaped, Straight jets, Multiple hotspots, "
                        "Continuous jets, Banding, One-sided, Restarted), 'environment' (15-18: Cluster, Merger, Diffuse emission, Unknown), "
                        "'derived' (19-23: Compact+hybrids, Hybrid FRI/FRII, Curved FRIs, Curved FRIIs, Straight+multi hotspots)")
    
    # Dataset pairing strategy
    ap.add_argument("--weighting", type=str, default="closest",
                    choices=["closest", "ponderate"],
                    help="Weight function for sampling pairs: 'closest' or 'ponderate' (default: closest)")
    ap.add_argument("--loss-mode", type=str, default="either", choices=["either", "both"],
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
    ap.add_argument("--model-type", type=str, default="efficient",
                    choices=["efficient", "original"],
                    help="Model architecture: 'efficient' (simple forward) or 'original' (snippet-style NetWrapper)")
    
    # Training hyperparameters
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for training (default: 32)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 0.0003)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs (default: 100)")
    ap.add_argument("--ema-decay", type=float, default=0.996,
                    help="EMA decay rate for target network (default: 0.996)")
    
    # Gradient and optimization
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="Gradient clipping max norm (default: None, no clipping)")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Number of learning rate warmup epochs (default: 0)")
    
    # Batch normalization
    ap.add_argument("--bn-momentum", type=float, default=0.1,
                    help="BatchNorm momentum (default: 0.1, PyTorch default)")
    
    # EMA decay scheduling
    ap.add_argument("--ema-decay-schedule", type=str, default="constant",
                    choices=["constant", "cosine"],
                    help="EMA decay scheduling strategy (default: constant)")
    ap.add_argument("--ema-decay-start", type=float, default=0.996,
                    help="Starting EMA decay for scheduled decay (default: 0.996)")
    ap.add_argument("--ema-decay-end", type=float, default=0.9999,
                    help="Ending EMA decay for scheduled decay (default: 0.9999)")
    
    # Model architecture
    ap.add_argument("--projection-dim", type=int, default=256,
                    help="Projection head output dimension (default: 256)")
    ap.add_argument("--hidden-dim", type=int, default=4096,
                    help="Hidden layer dimension in MLP heads (default: 4096)")
    
    # Output configuration
    ap.add_argument("--output-dir", type=Path,
                    default=Path('/users/mbredber/supervised_latent/outputs'),
                    help="Base output directory for checkpoints and embeddings")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Custom run name (default: timestamp)")
    # Visualization
    ap.add_argument("--no-plot-history", action="store_true",
                    help="Disable training curve plots (enabled by default)")
    
    # UMAP visualization
    ap.add_argument("--no-plot-umap", action="store_true",
                    help="Disable UMAP plots (enabled by default)")
    ap.add_argument("--umap-n-neighbors", type=int, default=15,
                    help="UMAP n_neighbors parameter (default: 15)")
    ap.add_argument("--umap-min-dist", type=float, default=0.1,
                    help="UMAP min_dist parameter (default: 0.1)")
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
EMA_DECAY = args.ema_decay
PROJECTION_DIM = args.projection_dim
HIDDEN_DIM = args.hidden_dim
BN_MOMENTUM = args.bn_momentum
MODEL_TYPE = args.model_type

# Optimization hyperparameters
GRAD_CLIP = args.grad_clip
WARMUP_EPOCHS = args.warmup_epochs

# EMA decay scheduling
EMA_DECAY_SCHEDULE = args.ema_decay_schedule
EMA_DECAY_START = args.ema_decay_start
EMA_DECAY_END = args.ema_decay_end

# Dataset configuration
DATASET_NAME = args.dataset
P_PAIR_FROM_CLASS = args.prob

# Data subsampling
MOCK_DATA_SIZE = args.subsample

# Random seed
SEED = args.seed
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
    print(f"⚠ CUDA not available, using CPU")
    print(f"  This will be VERY slow and may crash with large batches")

use_cuda = torch.cuda.is_available()

# Set explicit output directory
OUTPUT_BASE = args.output_dir
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Create run directory
from datetime import datetime
_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
if args.run_name:
    RUN_ID = f"{args.run_name}_{_timestamp}"
else:
    RUN_ID = _timestamp
    if DATASET_NAME != "LOTSS":
        RUN_ID += f"_{DATASET_NAME}"
    RUN_ID += f"_{MODEL_TYPE}_w{args.weighting}_p{P_PAIR_FROM_CLASS}"
    
# Truncate labels based on label type
LABEL_RANGES = {
    'full': (0, 20),          # All labels
    'initial': (0, 5),        # FRI, FRII, Hybrids, Spirals, Relaxed doubles
    'morphology': (5, 15),    # C-curve through Restarted
    'environment': (15, 19),  # Cluster, Merger, Diffuse emission, Unknown
    'derived': (19, 24)       # Compact+hybrids through Straight+multi hotspots (note: may only have 19-23, adjust if needed)
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
import sys as _sys
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
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
print(f"CONFIGURATION")
print(f"Output directory: {OUTPUT_DIR}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"{'='*70}")
print(f"Model type:     {MODEL_TYPE}")
print(f"Dataset:        {DATASET_NAME}")
print(f"Data dir:       {args.data_dir}")
print(f"Label type:     {args.label_type} ({label_dims} dims)")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Learning rate:  {LEARNING_RATE}")
print(f"Epochs:         {NUM_EPOCHS}")
print(f"Warmup epochs:  {WARMUP_EPOCHS}")
print(f"Grad clip:      {GRAD_CLIP if GRAD_CLIP else 'None'}")
print(f"BN momentum:    {BN_MOMENTUM}")
print(f"EMA decay:      {EMA_DECAY_SCHEDULE}")
if EMA_DECAY_SCHEDULE == "cosine":
    print(f"  Start:        {EMA_DECAY_START}")
    print(f"  End:          {EMA_DECAY_END}")
else:
    print(f"  Value:        {EMA_DECAY}")
print(f"Weighting:      {args.weighting}")
print(f"Pair prob:      {P_PAIR_FROM_CLASS}")
print(f"Device:         {device}")
if MOCK_DATA_SIZE:
    print(f"Subsampling:    {MOCK_DATA_SIZE} samples")
print(f"{'='*70}\n")

# =============================================================================
# DATASET LOADING
# =============================================================================

# Data paths
IMAGES_PATH = args.data_dir / 'data/images.npy'
LABELS_PATH = args.data_dir / 'data/labels.npy'

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
if MOCK_DATA_SIZE is not None and len(images) > MOCK_DATA_SIZE:
    print(f"\n⚠ Subsampling {MOCK_DATA_SIZE}/{len(images)} samples")
    indices = np.random.choice(len(images), MOCK_DATA_SIZE, replace=False)
    images = images[indices]
    labels = labels[indices]

print(f"\n✓ Data loaded")
print(f"  Images: {images.shape} ({images.dtype})")
print(f"  Labels: {labels.shape} ({labels.dtype})")
print(f"  Range: [{images.min():.2f}, {images.max():.2f}]")

# =============================================================================
# TRAIN/VAL/TEST SPLIT
# =============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print(f"\nSplitting data ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

indices = np.arange(len(images))

# Split
train_idx, temp_idx = train_test_split(
    indices, test_size=(VAL_RATIO + TEST_RATIO), random_state=SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=SEED
)

train_images = images[train_idx]
train_labels = labels[train_idx]
val_images = images[val_idx]
val_labels = labels[val_idx]
test_images = images[test_idx]
test_labels = labels[test_idx]

print(f"  Train: {len(train_images)}")
print(f"  Val:   {len(val_images)}")
print(f"  Test:  {len(test_images)}")

# =============================================================================
# CREATE DATASETS
# =============================================================================

WEIGHTING_FUNC = weights_closest if args.weighting == "closest" else weights_ponderate
print("\nCreating datasets...")

# Convert numpy arrays to DataFrames
train_labels_df = pd.DataFrame(train_labels)
val_labels_df = pd.DataFrame(val_labels)
test_labels_df = pd.DataFrame(test_labels)

print(f"  Converted labels to DataFrames")

# Transforms
byol_strong_aug = get_augmentation(args.augmentation)
print(f"  Augmentation: {args.augmentation}")

train_dataset = BYOLSupDataset(
    tags_data=train_labels_df,
    img_data=train_images,
    transform=byol_strong_aug,
    friend_transform=byol_strong_aug,
    weightfunc=WEIGHTING_FUNC,
    p_pair_from_class=P_PAIR_FROM_CLASS
)

val_dataset = BYOLSupDataset(
    tags_data=val_labels_df,
    img_data=val_images,
    transform=byol_strong_aug,
    friend_transform=byol_strong_aug,
    weightfunc=WEIGHTING_FUNC,  
    p_pair_from_class=P_PAIR_FROM_CLASS
)

test_dataset = BYOLSupDataset(
    tags_data=test_labels_df,
    img_data=test_images,
    transform=byol_strong_aug,
    friend_transform=byol_strong_aug,
    weightfunc=WEIGHTING_FUNC,
    p_pair_from_class=P_PAIR_FROM_CLASS
)

# DATA LOADERS 
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4 if use_cuda else 0,
    pin_memory=use_cuda, drop_last=True
)

# No-shuffle loader for ordered embedding extraction (UMAP + outlier plots)
train_extract_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4 if use_cuda else 0,
    pin_memory=use_cuda, drop_last=False
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4 if use_cuda else 0,
    pin_memory=use_cuda
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4 if use_cuda else 0,
    pin_memory=use_cuda
)

print(f"\n{'='*70}")
print(f"✓ DATA LOADED")
print(f"{'='*70}")
print(f"Train: {len(train_loader)} batches × {BATCH_SIZE}")
print(f"Val:   {len(val_loader)} batches × {BATCH_SIZE}")
print(f"Test:  {len(test_loader)} batches × {BATCH_SIZE}")
print(f"{'='*70}\n")

# Test sampling
x1, x1_trans, x2_friend, _ = next(iter(train_loader))
print(f"✓ Test batch: {x1.shape}, {x1_trans.shape}, {x2_friend.shape}")
print(f"  Different: {not torch.allclose(x1, x1_trans)}")

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

LOSS_MODE = args.loss_mode
SUPERVISION_WEIGHT = args.supervision_weight
SUPERVISION_WEIGHT_SCHEDULE = args.supervision_weight_schedule
SUPERVISION_WEIGHT_START = args.supervision_weight_start
SUPERVISION_WEIGHT_END = args.supervision_weight_end
PROB_PAIR_FROM_CLASS = args.prob
PROB_SCHEDULE = args.prob_schedule
PROB_START = args.prob_start
PROB_END = args.prob_end

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print("\nInitializing model...")

if MODEL_TYPE == "efficient":
    # Efficient model (Document 2)
    model = BYOLEfficient(
        encoder_dim=512,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        bn_momentum=BN_MOMENTUM
    )
else:
    # Original model (Document 3)
    encoder = BYOLEncoder(bn_momentum=BN_MOMENTUM)
    model = BYOLOriginal(
        encoder, 
        image_size=89,
        projection_size=PROJECTION_DIM, 
        projection_hidden_size=HIDDEN_DIM,
        moving_average_decay=EMA_DECAY,
        use_momentum=True,
        bn_momentum=BN_MOMENTUM
    )

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"{'='*70}")
print(f"MODEL ARCHITECTURE ({MODEL_TYPE.upper()})")
print(f"{'='*70}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Encoder output:       512-dim representation")
print(f"Projector output:     {PROJECTION_DIM}-dim projection")
print(f"Predictor output:     {PROJECTION_DIM}-dim prediction")
print(f"{'='*70}\n")

if use_cuda:
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.0f} MB")

# =============================================================================
# TRAINING SETUP
# =============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler: warmup + cosine annealing
if WARMUP_EPOCHS > 0:
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[int(0.7*(NUM_EPOCHS - WARMUP_EPOCHS))], 
        gamma=0.2
    )
else:
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[int(0.7*NUM_EPOCHS)], 
        gamma=0.2
    )

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': [],
    'ema_decay': []
}

best_val_loss = float('inf')
best_model_state = None
best_epoch = 0

print(f"✓ Optimizer: Adam (lr={LEARNING_RATE})")
print(f"✓ Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
if WARMUP_EPOCHS > 0:
    print(f"✓ Warmup: {WARMUP_EPOCHS} epochs")
if GRAD_CLIP:
    print(f"✓ Gradient clipping: max_norm={GRAD_CLIP}")
print(f"✓ Loss: BYOL symmetric MSE")

# =============================================================================
# TRAINING LOOP
# =============================================================================
print(f"\n{'='*70}")
print(f"STARTING TRAINING")
print(f"{'='*70}\n")

for epoch in range(NUM_EPOCHS):
    # -------------------------------------------------------------------------
    # LEARNING RATE WARMUP
    # -------------------------------------------------------------------------
    if epoch < WARMUP_EPOCHS:
        current_lr = get_warmup_lr(epoch, LEARNING_RATE, WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    
    # -------------------------------------------------------------------------
    # EMA DECAY SCHEDULING
    # -------------------------------------------------------------------------
    current_ema_decay = get_ema_decay(
        epoch, NUM_EPOCHS, 
        schedule=EMA_DECAY_SCHEDULE,
        base_decay=EMA_DECAY,
        start_decay=EMA_DECAY_START,
        end_decay=EMA_DECAY_END
    )
    
    # -------------------------------------------------------------------------
    # SUPERVISION WEIGHT SCHEDULING
    # -------------------------------------------------------------------------
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

    # Update EMA decay for original model
    if MODEL_TYPE == "original":
        model.target_ema_updater.beta = current_ema_decay
    
    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for x1, x1_trans, x2_friend, _ in pbar:
        x1, x1_trans, x2_friend = x1.to(device), x1_trans.to(device), x2_friend.to(device)
        if LOSS_MODE == "either":
            u = torch.rand(x1.size(0), device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)*torch.ones(x1.size(), device=device)
            x2 = torch.where(u < current_prob, x2_friend, x1_trans)
            if MODEL_TYPE == "efficient":
                pred1, pred2, proj1, proj2 = model(x1, x2)
                loss = byol_loss(pred1, pred2, proj1, proj2)
            else:  # original
                images = torch.cat((x1, x2), dim=0)
                loss = model(images)
        else:  # "both"
            if MODEL_TYPE == "efficient":
                pred1_f, pred2_f, proj1_f, proj2_f = model(x1, x2_friend)
                loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                pred1_t, pred2_t, proj1_t, proj2_t = model(x1, x1_trans)
                loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
            else:  # original
                images_friend = torch.cat((x1, x2_friend), dim=0)
                loss_friend = model(images_friend)
                images_trans = torch.cat((x1, x1_trans), dim=0)
                loss_trans = model(images_trans)
            loss = loss_trans + current_supervision_weight * loss_friend

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if MODEL_TYPE == "efficient":
            model.update_target_network(momentum=current_ema_decay)
        else:  # original
            model.update_moving_average()

        train_loss += loss.item()
        pbar.set_postfix({'train': f'{loss.item():.4f}'})

    avg_train_loss = train_loss / len(train_loader)

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x1, x1_trans, x2_friend, _ in val_loader:
            x1, x1_trans, x2_friend = x1.to(device), x1_trans.to(device), x2_friend.to(device)
            if LOSS_MODE == "either":
                u = torch.rand(x1.size(0), device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)*torch.ones(x1.size(), device=device)
                x2 = torch.where(u < current_prob, x2_friend, x1_trans)
                if MODEL_TYPE == "efficient":
                    pred1, pred2, proj1, proj2 = model(x1, x2)
                    val_loss += byol_loss(pred1, pred2, proj1, proj2).item()
                else:  # original
                    images = torch.cat((x1, x2), dim=0)
                    val_loss += model(images).item()
            else:  # "both"
                if MODEL_TYPE == "efficient":
                    pred1_f, pred2_f, proj1_f, proj2_f = model(x1, x2_friend)
                    loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                    pred1_t, pred2_t, proj1_t, proj2_t = model(x1, x1_trans)
                    loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
                else:  # original
                    images_friend = torch.cat((x1, x2_friend), dim=0)
                    loss_friend = model(images_friend)
                    images_trans = torch.cat((x1, x1_trans), dim=0)
                    loss_trans = model(images_trans)
                val_loss += (loss_trans + current_supervision_weight * loss_friend).item()

    avg_val_loss = val_loss / len(val_loader)

    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['lr'].append(current_lr)
    history['ema_decay'].append(current_ema_decay)

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1

    best_marker = ' ★' if is_best else ''
    sup_str = f" | sup: {current_supervision_weight:.3f}" if LOSS_MODE == "both" else f" | prob: {current_prob:.3f}"
    print(f"Epoch {epoch+1:>4}/{NUM_EPOCHS} | train: {avg_train_loss:.4f} | val: {avg_val_loss:.4f} | lr: {current_lr:.2e} | ema: {current_ema_decay:.4f}{sup_str}{best_marker}")
    
    # Step scheduler (after warmup phase)
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

print(f"{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"{'='*70}\n")

# Load best model
model.load_state_dict(best_model_state)

# =============================================================================
# TEST SET EVALUATION
# =============================================================================
print("\nEvaluating on TEST set (held-out)...")

model.eval()
test_loss = 0.0

with torch.no_grad():
    for x1, x1_trans, x2_friend, _ in tqdm(test_loader, desc="Test"):
        x1, x1_trans, x2_friend = x1.to(device), x1_trans.to(device), x2_friend.to(device)
        if LOSS_MODE == "either":
                u = torch.rand(x1.size(0), device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)*torch.ones(x1.size(), device=device)
                x2 = torch.where(u < PROB_PAIR_FROM_CLASS, x2_friend, x1_trans)
                if MODEL_TYPE == "efficient":
                    pred1, pred2, proj1, proj2 = model(x1, x2)
                    test_loss += byol_loss(pred1, pred2, proj1, proj2).item()
                else:  # original
                    images = torch.cat((x1, x2), dim=0)
                    test_loss += model(images).item()
        else:  # "both"
            if MODEL_TYPE == "efficient":
                pred1_f, pred2_f, proj1_f, proj2_f = model(x1, x2_friend)
                loss_friend = byol_loss(pred1_f, pred2_f, proj1_f, proj2_f)
                pred1_t, pred2_t, proj1_t, proj2_t = model(x1, x1_trans)
                loss_trans = byol_loss(pred1_t, pred2_t, proj1_t, proj2_t)
            else:  # original
                images_friend = torch.cat((x1, x2_friend), dim=0)
                loss_friend = model(images_friend)
                images_trans = torch.cat((x1, x1_trans), dim=0)
                loss_trans = model(images_trans)
            test_loss += (loss_trans + current_supervision_weight * loss_friend).item()

avg_test_loss = test_loss / len(test_loader)

print(f"\n{'='*70}")
print(f"TEST SET RESULTS (Best Model)")
print(f"{'='*70}")
print(f"Test Loss:  {avg_test_loss:.4f}")
print(f"Best Val:   {best_val_loss:.4f}")
print(f"Difference: {abs(avg_test_loss - best_val_loss):.4f}")
print(f"{'='*70}\n")

# Add to history
history['test_loss'] = avg_test_loss

# =============================================================================
# SAVE MODEL AND HISTORY
# =============================================================================

# Save model checkpoint
torch.save({
    'model_state_dict': best_model_state,
    'optimizer_state_dict': optimizer.state_dict(),
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
        'bn_momentum': BN_MOMENTUM,
        'ema_decay': EMA_DECAY,
        'ema_decay_schedule': EMA_DECAY_SCHEDULE,
        'ema_decay_start': EMA_DECAY_START,
        'ema_decay_end': EMA_DECAY_END,
        'projection_dim': PROJECTION_DIM,
        'hidden_dim': HIDDEN_DIM,
        'encoder_dim': 512,
        'weighting': args.weighting,
        'p_pair_from_class': P_PAIR_FROM_CLASS,
            'prob_schedule': PROB_SCHEDULE,
            'supervision_weight': SUPERVISION_WEIGHT,
            'supervision_weight_schedule': SUPERVISION_WEIGHT_SCHEDULE,
        'dataset': DATASET_NAME,
        'label_type': args.label_type,
    }
}, checkpoint_path)

print(f"✓ Model checkpoint saved to {checkpoint_path}")

# Save training history
np.save(DATA_DIR / 'training_history.npy', history)
print(f"✓ Training history saved to {DATA_DIR / 'training_history.npy'}")

# Plot training history (default behavior unless disabled)
if not args.no_plot_history:
    print("\nGenerating training curve plots...")
    plot_training_curves(history, best_val_loss, best_epoch, MODEL_TYPE, FIGURES_DIR)

# =============================================================================
# EXTRACT EMBEDDINGS
# =============================================================================

print("\nExtracting embeddings from DataLoaders...")

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
np.save(EMBEDDINGS_DIR / 'train_projections.npy', train_projections)
np.save(EMBEDDINGS_DIR / 'val_projections.npy', val_projections)
np.save(EMBEDDINGS_DIR / 'test_projections.npy', test_projections)

# Save corresponding labels
np.save(EMBEDDINGS_DIR / 'train_labels.npy', train_labels[:len(train_projections)])
np.save(EMBEDDINGS_DIR / 'val_labels.npy', val_labels[:len(val_projections)])
np.save(EMBEDDINGS_DIR / 'test_labels.npy', test_labels[:len(test_projections)])

print(f"\n✓ Embeddings saved to {EMBEDDINGS_DIR}/")

# Generate UMAP plots (default behavior unless disabled)
if not args.no_plot_umap:
    print("\nGenerating UMAP visualizations...")
    
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
    
    # Store full labels before any filtering (needed for all classification types)
    # Reload full labels
    labels_full = np.load(LABELS_PATH)
    train_labels_full = labels_full[train_idx]
    test_labels_full = labels_full[test_idx]
    
    # Train UMAP: fit and save
    train_reducer, train_2d = plot_umap_pure_classes(
        train_projections,
        train_labels[:len(train_projections)],
        "Train Projections (256-dim)",
        "umap_train_projections",
        "train",
        args=args,
        SEED=SEED,
        LABEL_RANGES=LABEL_RANGES,
        CLASS_NAMES=CLASS_NAMES,
        OUTPUT_DIR=FIGURES_DIR,
        train_labels_full=train_labels_full,
        test_labels_full=test_labels_full,
    )
    np.save(EMBEDDINGS_DIR / 'umap_train_coords.npy', train_2d)

    # Test UMAP: independent fit with centroid annotations
    _, _test_2d = plot_umap_pure_classes(
        test_projections,
        test_labels[:len(test_projections)],
        "Test Projections (256-dim)",
        "umap_test_projections",
        "test",
        args=args,
        SEED=SEED,
        LABEL_RANGES=LABEL_RANGES,
        CLASS_NAMES=CLASS_NAMES,
        OUTPUT_DIR=FIGURES_DIR,
        train_labels_full=train_labels_full,
        test_labels_full=test_labels_full,
        annotate_centroids=True,
    )
    np.save(EMBEDDINGS_DIR / 'umap_test_coords.npy', _test_2d)

    # Test UMAP: transformed into train space (fair comparison)
    plot_umap_pure_classes(
        test_projections,
        test_labels[:len(test_projections)],
        "Test Projections in Train UMAP Space (256-dim)",
        "umap_test_projections_transformed",
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
    
    # Outlier plot: 4 most extreme points in train UMAP space
    plot_umap_outliers(
        train_2d,
        train_images[:len(train_2d)],
        OUTPUT_DIR=FIGURES_DIR,
    )

    print(f"\n✓ UMAP plots saved to {FIGURES_DIR}/")

# Compute metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) for test and train projections
if not args.no_metrics:
    # take the following cases : 
    # - only FRI vs only FRII; 
    # - all the base classes (FRI only, FRII only, all Hybrids, Spirals only, Relaxed doubles only) together

    metrics = {}
    for split, projections, labels in zip(
        ['train', 'test'], 
        [train_projections, test_projections], 
        [train_labels[:len(train_projections)], test_labels[:len(test_projections)]]
    ):
        metrics[split] = {'fri_vs_frii':{}, 'base_classes':{}}
        fri_only = (labels[:, 0] == 1) & (labels[:,1] == 0)
        frii_only = (labels[:, 1] == 1) & (labels[:,0] == 0)
        combined_fri_frii = fri_only | frii_only
        #the labels are in the format of a one-hot encoding, so we need to convert them to a single label for each class
        labels_hot = np.argmax(labels[combined_fri_frii][:,:2], axis=1)
        metrics[split]['fri_vs_frii'] = {
            'silhouette': silhouette_score(projections[combined_fri_frii], labels_hot).item(),
            'davies_bouldin': davies_bouldin_score(projections[combined_fri_frii], labels_hot).item(),
            'calinski_harabasz': calinski_harabasz_score(projections[combined_fri_frii], labels_hot).item()
        }
        fri_only = (labels[:, 0] == 1) & (labels[:,:5].sum(axis=1) == 1)
        frii_only = (labels[:, 1] == 1) & (labels[:,:5].sum(axis=1) == 1)
        all_hybrids = (labels[:, 2] == 1)
        spirals_only = (labels[:, 3] == 1) & (labels[:,:5].sum(axis=1) == 1)
        relaxed_doubles_only = (labels[:, 4] == 1) & (labels[:,:5].sum(axis=1) == 1)
        combined = fri_only | frii_only | all_hybrids | spirals_only | relaxed_doubles_only
        print("DEBUG", labels.shape, fri_only.shape, all_hybrids.shape, combined.shape)
        labels_hot = np.argmax(labels[combined][:,:5], axis=1)
        labels_hot[all_hybrids[combined]] = 2  # assign hybrid label (index 2) to all hybrids, even if they also have spiral or relaxed double labels
        metrics[split]['base_classes'] = {
            'silhouette': silhouette_score(projections[combined], labels_hot).item(),
            'davies_bouldin': davies_bouldin_score(projections[combined], labels_hot).item(),
            'calinski_harabasz': calinski_harabasz_score(projections[combined], labels_hot).item()
        }

    # and save to a json file
    with open(OUTPUT_DIR / 'projection_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Projection clustering metrics saved to {OUTPUT_DIR / 'projection_metrics.json'}")

print(f"\n{'='*70}")
print(f"SCRIPT COMPLETE")
print(f"{'='*70}")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"{'='*70}\n")