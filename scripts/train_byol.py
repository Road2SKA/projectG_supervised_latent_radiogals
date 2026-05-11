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
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

from suplat.data.data_samplers import BYOLSupDataset, UnlabelledBYOLDataset, weights_closest, weights_ponderate
from suplat.data.catalogue import Catalogue, LOTSS_LABEL_NAMES
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

    # Note: --data-dir, --dataset, --label-type, --subsample, --data-seed, and
    # --cv-folds have been removed. All of this is now encoded in the catalogue
    # YAML: which datasets to load, which label alias to expose, what fraction
    # of train labels to use (f_labels), and the split seed (dataset_seed).
    # Pass a catalogue YAML via --catalogue to configure data loading.

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
                    help="Base output directory for checkpoints and projections")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Custom run name (default: timestamp)")
    # Visualization
    ap.add_argument("--no-plot-history", action="store_true",
                    help="Disable training curve plots (enabled by default)")
    
    # UMAP visualization
    ap.add_argument("--no-plot-umap", action="store_true",
                    help="Disable UMAP plots (enabled by default)")
    ap.add_argument("--simple-multilabel", action="store_true",
                    help="Draw multi-label sources as open white circles instead of "
                         "pie-sector wedges (faster, less cluttered)")
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

    # Catalogue
    ap.add_argument("--catalogue", type=Path, required=True,
                    help="Path to a catalogue YAML file.")

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
FEATURE_COMPRESSION_MODE = args.feature_compression_mode
USE_COMPILE = args.compile
PROB_PAIR_FROM_CLASS = args.prob

# Random seed
SEED = args.seed

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

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
    RUN_ID = f"{_timestamp}_{MODEL_TYPE}_w{args.weighting}_p{PROB_PAIR_FROM_CLASS}"
    

OUTPUT_DIR = OUTPUT_BASE / f'run_{RUN_ID}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subfolders
FIGURES_DIR = OUTPUT_DIR / 'figures'
LOGS_DIR    = OUTPUT_DIR / 'logs'
DATA_DIR    = OUTPUT_DIR / 'data'
for _d in [FIGURES_DIR, LOGS_DIR, DATA_DIR]:
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
with open(LOGS_DIR / 'train_byol_configuration_log.txt', 'w') as _cfg:
    _cfg.write(f"Run: {RUN_ID}\n")
    _cfg.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _cfg.write("=" * 50 + "\n")
    for key, val in sorted(vars(args).items()):
        _cfg.write(f"{key}: {val}\n")

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
print(f"Catalogue:      {args.catalogue}")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Learning rate:  {LEARNING_RATE}")
print(f"Epochs:         {NUM_EPOCHS}")
print(f"Warmup epochs:  {WARMUP_EPOCHS}")
print(f"Grad clip:      {GRAD_CLIP if GRAD_CLIP else 'None'}")
print(f"EMA decay:      {EMA_DECAY}")
print(f"Weighting:      {args.weighting}")
print(f"Pair prob:      {PROB_PAIR_FROM_CLASS}")
print(f"Num workers:    {NUM_WORKERS}")
print(f"Compile:        {'enabled' if USE_COMPILE else 'disabled'}")
print(f"Device:         {device}")
print(f"{'='*70}\n")


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

# -------------------------------------------------------------------------
# CATALOGUE-BASED DATA LOADING
# -------------------------------------------------------------------------
mat = Catalogue.from_yaml(str(args.catalogue)).materialise(root='.')

print(f"\n{'='*70}")
print(f"CATALOGUE: {args.catalogue}")
print(f"  n_labelled_train: {mat.n_labelled_train}")
print(f"  label_names: {mat.label_names}")
print(f"{'='*70}\n")

all_images, labelled_df = mat.get_byol_data()
splits = mat.get_split_datasets()

# Build labelled train dataset
lab_idx = labelled_df['image_idx'].values
lab_images = all_images[lab_idx]
# Normalise labels to uniform 2-D shape before building tags_df.
# Multi-dataset catalogues can mix scalar integer labels (e.g. MiraBest native)
# with multi-hot arrays (e.g. LoTSS morphology); np.stack would fail on the
# shape mismatch.  Convert each label to a 1-D float array then zero-pad all
# to the same length so cdist inside BYOLSupDataset always gets a 2-D matrix.
label_arrs = [np.atleast_1d(np.asarray(l, dtype=np.float32))
              for l in labelled_df['label'].values]
max_len = max(a.shape[0] for a in label_arrs)
if max_len > 1:
    label_arrs = [np.pad(a, (0, max_len - len(a))) for a in label_arrs]
raw_labels = np.stack(label_arrs)
tags_df = pd.DataFrame(raw_labels).reset_index(drop=True)

byol_sup_ds = BYOLSupDataset(
    tags_data=tags_df, img_data=lab_images,
    transform=byol_strong_aug, friend_transform=byol_strong_aug,
    weightfunc=WEIGHTING_FUNC, p_pair_from_class=PROB_PAIR_FROM_CLASS,
)

# Build unlabelled dataset (all images not in the labelled train set)
unlabelled_mask = np.ones(len(all_images), dtype=bool)
unlabelled_mask[lab_idx] = False
unlabelled_images = all_images[unlabelled_mask]
unlab_ds = UnlabelledBYOLDataset(unlabelled_images, transform=byol_strong_aug)

# Combined train loader
train_ds = ConcatDataset([byol_sup_ds, unlab_ds])
_nw = NUM_WORKERS if use_cuda else 0
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=_nw, pin_memory=use_cuda, drop_last=True)
# Extract loader for PCA fitting (labelled split only, ordered)
train_extract_loader = DataLoader(byol_sup_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=_nw, pin_memory=use_cuda)

# Val / test loaders from labelled splits
def _concat_views(views):
    imgs = np.concatenate([v.images for v in views])
    # Normalise label arrays to uniform shape (same issue as for labelled_df
    # above: multi-dataset catalogues can mix 2-D multi-hot with 1-D scalars).
    label_blocks = []
    for v in views:
        lbl = v.labels
        if lbl.ndim == 1:
            lbl = lbl.reshape(-1, 1)
        label_blocks.append(lbl)
    max_cols = max(b.shape[1] for b in label_blocks)
    label_blocks = [np.pad(b, ((0, 0), (0, max_cols - b.shape[1])))
                    if b.shape[1] < max_cols else b
                    for b in label_blocks]
    lbls = np.concatenate(label_blocks)
    return imgs, lbls

val_images, val_labels = _concat_views(splits['val'])
test_images, test_labels = _concat_views(splits['test'])
_, val_loader  = _make_dataset_loader(val_images,  val_labels,  shuffle=False, drop_last=USE_COMPILE)
_, test_loader = _make_dataset_loader(test_images, test_labels, shuffle=False, drop_last=USE_COMPILE)

train_images = lab_images
train_labels = raw_labels
labels_full  = None   # full 20-dim LoTSS labels not available in catalogue mode
train_idx    = lab_idx

np.save(DATA_DIR / 'train_idx.npy', train_idx)

# Save val/test split indices so train_generative.py can pair images with projections.
# Indices are into the per-dataset image array (== images_filtered.npy for LoTSS-only).
# For multi-dataset catalogues these become global offsets into the concatenated all_images.
_img_off = 0
_val_parts, _test_parts = [], []
for _ds_name, _ds_data in mat._labelled.items():
    _val_parts.append(mat._splits[_ds_name]['val']  + _img_off)
    _test_parts.append(mat._splits[_ds_name]['test'] + _img_off)
    _img_off += len(_ds_data['images'])
np.save(DATA_DIR / 'val_idx.npy',  np.concatenate(_val_parts)  if _val_parts  else np.array([], dtype=np.int64))
np.save(DATA_DIR / 'test_idx.npy', np.concatenate(_test_parts) if _test_parts else np.array([], dtype=np.int64))

print(f"\n{'='*70}")
print("✓ CATALOGUE DATA LOADED")
print(f"{'='*70}")
print(f"  Labelled train: {len(lab_images)}")
print(f"  Unlabelled:     {len(unlabelled_images)}")
print(f"  Val:            {len(val_images)}")
print(f"  Test:           {len(test_images)}")
print(f"  Train batches:  {len(train_loader)}")
print(f"{'='*70}\n")

model, history, best_val_loss, best_epoch = train_fold(
    train_loader, val_loader, extract_loader=train_extract_loader
)

print("\nEvaluating on TEST set (held-out)...")
avg_test_loss = evaluate_test(model, test_loader)
print(f"\n{'='*70}")
print("TEST SET RESULTS (Best Model)")
print(f"{'='*70}")
print(f"Test Loss:  {avg_test_loss:.4f}")
print(f"Best Val:   {best_val_loss:.4f}")
print(f"Difference: {abs(avg_test_loss - best_val_loss):.4f}")
print(f"{'='*70}\n")

_items = [{'fold_idx': None, 'model': model, 'history': history,
            'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
            'avg_test_loss': avg_test_loss,
            'train_extract_loader': train_extract_loader,
            'val_loader': val_loader, 'train_labels': train_labels,
            'val_labels': val_labels, 'train_idx': train_idx,
            'train_images': train_images}]


# =============================================================================
# DOWNSTREAM: save, plot, extract embeddings
# =============================================================================

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
            'feature_compression_mode': FEATURE_COMPRESSION_MODE,
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
            'catalogue': str(args.catalogue),
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

    print(f"\nExtracting projections{_label}...")

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

    # Save projections and labels
    np.save(DATA_DIR / f'train_projections{_suffix}.npy', train_projections)
    np.save(DATA_DIR / f'val_projections{_suffix}.npy', val_projections)
    np.save(DATA_DIR / f'test_projections{_suffix}.npy', test_projections)
    np.save(DATA_DIR / f'train_labels{_suffix}.npy', train_labels[:len(train_projections)])
    np.save(DATA_DIR / f'val_labels{_suffix}.npy', val_labels[:len(val_projections)])
    np.save(DATA_DIR / f'test_labels{_suffix}.npy', test_labels[:len(test_projections)])

    print(f"\n✓ Projections saved to {DATA_DIR}/")

    # =========================================================================
    # UMAP VISUALISATION
    # =========================================================================
    # The catalogue carries the label alias (e.g. 'initial', 'morphology',
    # 'full'). We build a LABEL_RANGES dict so that only the panel matching
    # the catalogue's label type has a non-zero column range; the other three
    # panels get (0, 0) and are hidden automatically by plot_umap_pure_classes.
    # For 'pure' aliases the catalogue returns 1-D integer class indices, so
    # we convert them to one-hot before calling the plotting function.
    if not args.no_plot_umap:
        import types as _types

        print(f"\nGenerating UMAP visualizations{_label}...")

        _cat_label_type = mat._entries[0].labels

        # Map each catalogue label alias to per-panel column ranges
        # Column ranges within the catalogue's label array per UMAP panel.
        # initial: cols 0-4 (5), morphology: cols 5-15 (11), environment: cols 16-19 (4).
        # Derived labels are computed from combinations of other labels and are not
        # stored as columns, so the derived panel is always hidden (0, 0).
        # For partial-label catalogues only the matching panel is non-zero; the rest
        # return shape (N, 0) which plot_umap_pure_classes hides automatically.
        _UMAP_RANGES_MAP = {
            'full':             {'initial': (0, 5),  'morphology': (5, 16), 'environment': (16, 20), 'derived': (0, 0)},
            'native':           {'initial': (0, 5),  'morphology': (5, 16), 'environment': (16, 20), 'derived': (0, 0)},
            'initial':          {'initial': (0, 5),  'morphology': (0, 0),  'environment': (0, 0),   'derived': (0, 0)},
            'initial_pure':     {'initial': (0, 5),  'morphology': (0, 0),  'environment': (0, 0),   'derived': (0, 0)},
            'classical':        {'initial': (0, 2),  'morphology': (0, 0),  'environment': (0, 0),   'derived': (0, 0)},
            'classical_pure':   {'initial': (0, 2),  'morphology': (0, 0),  'environment': (0, 0),   'derived': (0, 0)},
            'morphology':       {'initial': (0, 0),  'morphology': (0, 11), 'environment': (0, 0),   'derived': (0, 0)},
            'morphology_pure':  {'initial': (0, 0),  'morphology': (0, 11), 'environment': (0, 0),   'derived': (0, 0)},
            'environment':      {'initial': (0, 0),  'morphology': (0, 0),  'environment': (0, 4),   'derived': (0, 0)},
            'environment_pure': {'initial': (0, 0),  'morphology': (0, 0),  'environment': (0, 4),   'derived': (0, 0)},
        }
        _empty = {'initial': (0, 0), 'morphology': (0, 0), 'environment': (0, 0), 'derived': (0, 0)}
        umap_label_ranges = _UMAP_RANGES_MAP.get(_cat_label_type, _empty)

        # Convert 1-D integer labels (pure aliases) to one-hot (N, n_classes)
        def _to_onehot(lbl, n_cls):
            if lbl.ndim == 1:
                oh = np.zeros((len(lbl), n_cls), dtype=np.int64)
                oh[np.arange(len(lbl)), lbl] = 1
                return oh
            return lbl

        _n_cls = len(mat.label_names)

        # In multi-dataset catalogues, non-primary datasets have scalar class
        # indices stored in column 0 (e.g. [2, 0, ..., 0]) rather than proper
        # multi-hot binary labels.  The plotting code misreads value 2.0 as
        # "2 active labels" and crashes.  Zero out non-primary-dataset rows so
        # they appear as unlabelled in the UMAP instead.
        _primary_ds = mat._entries[0].dataset
        _tr_labels_umap = train_labels[:len(train_projections)].copy()
        if len(mat._labelled) > 1:
            _non_primary_train = labelled_df['dataset'].values[:len(train_projections)] != _primary_ds
            _tr_labels_umap[_non_primary_train] = 0
            # Build the same mask for test: views are ordered by mat._labelled insertion order
            _test_ds_tags = []
            for v in splits['test']:
                _test_ds_tags.extend([v.dataset] * len(v.images))
            _test_ds_tags = np.array(_test_ds_tags)
            _te_labels_umap = test_labels[:len(test_projections)].copy()
            _te_labels_umap[_test_ds_tags[:len(test_projections)] != _primary_ds] = 0
        else:
            _te_labels_umap = test_labels[:len(test_projections)]

        _tr_lbl = _to_onehot(_tr_labels_umap, _n_cls)
        _te_lbl = _to_onehot(_te_labels_umap, _n_cls)

        # CLASS_NAMES keyed by panel name.  Each panel's list must have exactly
        # as many entries as the columns it receives from LABEL_RANGES (otherwise
        # the enumerate loop in _plot_umap_ax goes out of bounds).
        # For 'full'/'native', each panel gets its own subset of label names.
        # For single-alias catalogues only one panel is active and mat.label_names
        # has exactly the right length for that panel.
        if _cat_label_type in ('full', 'native', 'none'):
            CLASS_NAMES = {
                'initial':     LOTSS_LABEL_NAMES['initial'],
                'morphology':  LOTSS_LABEL_NAMES['morphology'],
                'environment': LOTSS_LABEL_NAMES['environment'],
                'derived':     [],
            }
        else:
            CLASS_NAMES = {k: mat.label_names for k in ('initial', 'morphology', 'environment', 'derived')}

        # plot_umap_pure_classes checks args.label_type; pass 'full' so it
        # reads labels directly from the array rather than looking up a fallback
        _umap_args = _types.SimpleNamespace(
            label_type='full',
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

        _emb_dim = train_projections.shape[1]

        _simple_ml = args.simple_multilabel

        # Train UMAP: fit and save
        train_reducer, train_2d = plot_umap_pure_classes(
            train_projections, _tr_lbl,
            f"Train ({_emb_dim}-dim)", f"umap_train{_suffix}", "train",
            args=_umap_args, SEED=SEED,
            LABEL_RANGES=umap_label_ranges, CLASS_NAMES=CLASS_NAMES,
            OUTPUT_DIR=FIGURES_DIR, simple_multilabel=_simple_ml,
        )
        np.save(DATA_DIR / f'umap_train_coords{_suffix}.npy', train_2d)

        # Test UMAP: independent fit
        _, _test_2d = plot_umap_pure_classes(
            test_projections, _te_lbl,
            f"Test ({_emb_dim}-dim)", f"umap_test{_suffix}", "test",
            args=_umap_args, SEED=SEED,
            LABEL_RANGES=umap_label_ranges, CLASS_NAMES=CLASS_NAMES,
            OUTPUT_DIR=FIGURES_DIR, simple_multilabel=_simple_ml,
        )
        np.save(DATA_DIR / f'umap_test_coords{_suffix}.npy', _test_2d)

        # Test UMAP: transformed into train UMAP space
        _, _test_transformed_2d = plot_umap_pure_classes(
            test_projections, _te_lbl,
            f"Test in Train UMAP Space ({_emb_dim}-dim)",
            f"umap_test_transformed{_suffix}", "test",
            args=_umap_args, SEED=SEED,
            LABEL_RANGES=umap_label_ranges, CLASS_NAMES=CLASS_NAMES,
            OUTPUT_DIR=FIGURES_DIR, reducer=train_reducer,
            simple_multilabel=_simple_ml,
        )
        np.save(DATA_DIR / f'umap_test_transformed_coords{_suffix}.npy', _test_transformed_2d)

        # Outlier plot: 4 most extreme points in train UMAP space
        plot_umap_outliers(
            train_2d, train_images[:len(train_2d)],
            OUTPUT_DIR=FIGURES_DIR,
            labels=_tr_lbl,
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