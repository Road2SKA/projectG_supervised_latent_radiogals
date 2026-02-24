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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import copy
import random
from functools import wraps
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import umap

# Check device availability
if torch.cuda.is_available(): 
    print("using gpu")
else: 
    print("using cpu")

# =============================================================================
# HELPER FUNCTIONS (for original model)
# =============================================================================

def default(val, def_val):
    """Return default value if val is None"""
    return def_val if val is None else val

def flatten(t):
    """Flatten tensor to (batch_size, features)"""
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    """Decorator for singleton pattern - creates instance once and caches it"""
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    """Get device of first parameter in module"""
    return next(module.parameters()).device

def set_requires_grad(model, val):
    """Set requires_grad for all parameters in model"""
    for p in model.parameters():
        p.requires_grad = val

# =============================================================================
# AUGMENTATION UTILS
# =============================================================================

class RandomApply(nn.Module):
    """Apply augmentation with probability p"""
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# =============================================================================
# EMA CLASS (for original model)
# =============================================================================

class EMA():
    """Exponential moving average updater"""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    
    def update_average(self, old, new):
        """Update EMA: new_avg = beta * old + (1 - beta) * new"""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    """Update all parameters in ma_model using EMA from current_model"""
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# =============================================================================
# MLP HEADS
# =============================================================================

def MLP(dim, projection_size, hidden_size=4096, bn_momentum=0.1):
    """Standard MLP projection head for BYOL"""
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size, momentum=bn_momentum),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# =============================================================================
# LOSS FUNCTION (for original model)
# =============================================================================

def loss_fn(x, y):
    """
    BYOL loss: normalized MSE between prediction and target
    Returns per-sample loss (not reduced)
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

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
    ap.add_argument("--prob", type=float, default=0.5,
                    help="Probability of pairing from same class (default: 0.5)")
    
    # Data subsampling
    ap.add_argument("--subsample", type=int, default=None,
                    help="Subsample dataset to N samples (for quick testing)")
    
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
DATA_DIR = args.data_dir
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
if args.run_name:
    RUN_ID = args.run_name
else:
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
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
checkpoint_path = OUTPUT_DIR / 'byol_model_best.pt'
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
print(f"Data dir:       {DATA_DIR}")
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
IMAGES_PATH = DATA_DIR / 'data/images-ashley.npy'
LABELS_PATH = DATA_DIR / 'data/labels.npy'

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
    indices, test_size=(VAL_RATIO + TEST_RATIO), random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=42
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
# DATASET CLASS
# =============================================================================
from scipy.spatial.distance import cdist

def weights_closest(pi):
    """Weight function: closest samples get weight 1, others get 0"""
    weights = ((pi-pi.min())==0).squeeze()
    return weights/weights.sum()

def weights_ponderate(pi):
    """Weight function: inverse square distance weighting"""
    weights = 1/(pi-pi.min()+1).squeeze()**2
    return weights/weights.sum()

WEIGHTING_FUNC = weights_closest if args.weighting == "closest" else weights_ponderate

class BYOLSupDataset(Dataset):
    """
    Dataset for BYOL with supervised pairing based on label similarity.
    Returns augmented pairs (x1, x2) and their label distance.
    """
    def __init__(self, 
                 tags_data, 
                 img_data, 
                 transform=None, 
                 friend_transform=None,
                 weightfunc=weights_closest,
                 p_pair_from_class=0.5):
        self.all_labels = tags_data
        self.img_data = img_data
        self.transform = transform
        self.friend_transform = friend_transform
        self.weightfunc = weightfunc
        self.p_pair_from_class = p_pair_from_class
    
    def __len__(self):
        return self.all_labels.shape[0]
    
    def __getitem__(self, idx):
        # Fetch numpy array from storage
        img = self.img_data[idx]
        label_vec = self.all_labels.iloc[idx, :].values.reshape(1, -1)
        
        u = np.random.rand()
        if u < self.p_pair_from_class:
            # Sample a friend image from similar class
            all_tags_nofid = self.all_labels.drop(index=idx)
            pi = cdist(label_vec, all_tags_nofid.values, metric="cityblock")
            weights = self.weightfunc(pi)
            sample = np.random.choice(all_tags_nofid.shape[0], p=weights)
            idx_friend = all_tags_nofid.index[sample]
            mdist = pi[0, sample]
            img_friend = self.img_data[idx_friend]
        else:
            # Use same image (will be augmented differently)
            img_friend = img.copy()
            mdist = 0.0
        
        # Convert numpy arrays to tensors BEFORE transforms
        img = torch.from_numpy(img).unsqueeze(0).float()  # Shape: (1, H, W)
        img_friend = torch.from_numpy(img_friend).unsqueeze(0).float()
        
        # Apply transforms to tensors
        if self.transform:
            img = self.transform(img)
        if self.friend_transform:
            img_friend = self.friend_transform(img_friend)
        
        return img, img_friend, mdist

# =============================================================================
# CREATE DATASETS
# =============================================================================
print("\nCreating datasets...")

# Convert numpy arrays to DataFrames
train_labels_df = pd.DataFrame(train_labels)
val_labels_df = pd.DataFrame(val_labels)
test_labels_df = pd.DataFrame(test_labels)

print(f"  Converted labels to DataFrames")

# Transforms
base_transform = T.Compose([
    # Empty - tensors already created in __getitem__
])

byol_strong_aug = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(180),
])

train_dataset = BYOLSupDataset(
    tags_data=train_labels_df,
    img_data=train_images,
    transform=base_transform,
    friend_transform=byol_strong_aug,
    weightfunc=WEIGHTING_FUNC,
    p_pair_from_class=P_PAIR_FROM_CLASS
)

val_dataset = BYOLSupDataset(
    tags_data=val_labels_df,
    img_data=val_images,
    transform=base_transform,
    friend_transform=byol_strong_aug,
    weightfunc=WEIGHTING_FUNC,  
    p_pair_from_class=P_PAIR_FROM_CLASS
)

test_dataset = BYOLSupDataset(
    tags_data=test_labels_df,
    img_data=test_images,
    transform=base_transform,
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
x1, x2, _ = next(iter(train_loader))
print(f"✓ Test batch: {x1.shape}, {x2.shape}")
print(f"  Different: {not torch.allclose(x1, x2)}")

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Shared encoder architecture
class BYOLEncoder(nn.Module):
    """
    ResNet-style encoder (f_θ in BYOL paper) for 89x89 greyscale images.
    Outputs 512-dimensional representation (y_θ in BYOL vocabulary).
    """
    def __init__(self, bn_momentum=0.1):
        super().__init__()
        
        # Initial conv: 89x89 -> 45x45
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        
        # Residual blocks: 45x45 -> 23x23 -> 12x12 -> 6x6
        self.layer1 = self._make_layer(64, 128, stride=2, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(128, 256, stride=2, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(256, 512, stride=2, bn_momentum=bn_momentum)
        
        # Global pooling: 6x6 -> 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, in_channels, out_channels, stride=1, bn_momentum=0.1):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling - output: representation y (batch, 512)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


# Efficient model components
class ProjectionHead(nn.Module):
    """
    MLP projection head (g_θ in BYOL paper).
    Projects representation y to projection z.
    """
    def __init__(self, in_dim=512, hidden_dim=4096, out_dim=256, bn_momentum=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    """
    MLP prediction head (q_θ in BYOL paper).
    Predicts target projection from online projection.
    Only exists in online network (asymmetry prevents collapse).
    """
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256, bn_momentum=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class BYOLEfficient(nn.Module):
    """
    Efficient BYOL model (Document 2 style).
    
    Architecture:
    - Encoder f_θ: input → representation y (512-dim)
    - Projector g_θ: representation y → projection z (256-dim)
    - Predictor q_θ: projection z → prediction (256-dim) [online only]
    
    Loss compares online prediction with target projection.
    """
    def __init__(self, encoder_dim=512, projection_dim=256, hidden_dim=4096, bn_momentum=0.1):
        super().__init__()
        
        # Online network: encoder → projector → predictor
        self.online_encoder = BYOLEncoder(bn_momentum=bn_momentum)
        self.online_projector = ProjectionHead(
            in_dim=encoder_dim, 
            hidden_dim=hidden_dim, 
            out_dim=projection_dim,
            bn_momentum=bn_momentum
        )
        self.online_predictor = PredictionHead(
            in_dim=projection_dim, 
            hidden_dim=hidden_dim, 
            out_dim=projection_dim,
            bn_momentum=bn_momentum
        )
        
        # Target network: encoder → projector (no predictor!)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target network parameters (updated via EMA only)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def forward(self, x1, x2):
        """
        Forward pass for two augmented views.
        
        Returns:
            online_pred_1: prediction from view 1 (256-dim)
            online_pred_2: prediction from view 2 (256-dim)
            target_proj_1: projection from view 1 (256-dim, detached)
            target_proj_2: projection from view 2 (256-dim, detached)
        """
        # Online network forward pass
        online_repr_1 = self.online_encoder(x1)           # (batch, 512)
        online_repr_2 = self.online_encoder(x2)           # (batch, 512)
        
        online_proj_1 = self.online_projector(online_repr_1)  # (batch, 256)
        online_proj_2 = self.online_projector(online_repr_2)  # (batch, 256)
        
        online_pred_1 = self.online_predictor(online_proj_1)  # (batch, 256)
        online_pred_2 = self.online_predictor(online_proj_2)  # (batch, 256)
        
        # Target network forward pass (no gradients)
        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)       # (batch, 512)
            target_repr_2 = self.target_encoder(x2)       # (batch, 512)
            
            target_proj_1 = self.target_projector(target_repr_1)  # (batch, 256)
            target_proj_2 = self.target_projector(target_repr_2)  # (batch, 256)
        
        return online_pred_1, online_pred_2, target_proj_1, target_proj_2
    
    @torch.no_grad()
    def update_target_network(self, momentum=0.996):
        """
        Exponential moving average (EMA) update of target network.
        
        Target parameters: θ_target = m * θ_target + (1 - m) * θ_online
        """
        # Update target encoder
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + (1 - momentum) * online_params.data
            )
        
        # Update target projector
        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + (1 - momentum) * online_params.data
            )


# Original (snippet-style) model components
class NetWrapper(nn.Module):
    """
    Wrapper for base network following snippet pattern.
    Manages projection head and representation extraction.
    """
    def __init__(self, net, projection_size, projection_hidden_size, bn_momentum=0.1):
        super().__init__()
        self.net = net
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.bn_momentum = bn_momentum
    
    @singleton('projector')
    def _get_projector(self, hidden):
        """Create projector on first forward pass (singleton pattern)"""
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size, self.bn_momentum)
        return projector.to(hidden)
    
    def get_representation(self, x):
        """Extract features from encoder"""
        return self.net(x)
    
    def forward(self, x, return_projection=True):
        """
        Forward pass through encoder and optionally projector.
        
        Args:
            x: Input tensor
            return_projection: If True, return (projection, representation)
                             If False, return representation only
        """
        representation = self.get_representation(x)
        
        if not return_projection:
            return representation
        
        # Get or create projector (singleton pattern ensures single creation)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOLOriginal(nn.Module):
    """
    Original BYOL model (Document 3 style, snippet-based).
    Uses NetWrapper with singleton pattern and EMA class.
    """
    def __init__(
        self,
        net,
        image_size,
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        use_momentum=True,
        bn_momentum=0.1
    ):
        super().__init__()
        self.net = net
        
        # Online encoder with projection head
        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            bn_momentum=bn_momentum
        )
        
        # Target encoder (EMA of online)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        
        # Predictor (only for online network)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, bn_momentum)
        
        # Move to correct device
        device = get_module_device(net)
        self.to(device)
        
        # Initialize singleton parameters with mock forward pass
        self.forward(torch.randn(2, 1, image_size, image_size, device=device))
    
    @singleton('target_encoder')
    def _get_target_encoder(self):
        """Create target encoder as deepcopy of online encoder (singleton)"""
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    
    def reset_moving_average(self):
        """Reset target encoder (forces recreation on next forward)"""
        del self.target_encoder
        self.target_encoder = None
    
    def update_moving_average(self):
        """Update target network using EMA"""
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
    
    def forward(self, x, return_embedding=False, return_projection=True):
        """
        Forward pass for BYOL.
        
        Args:
            x: Input batch (concatenated pair [view1; view2] for training)
            return_embedding: If True, return embeddings instead of loss
            return_projection: If True (with return_embedding), return projections too
        """
        # If requesting embeddings, just return from online encoder
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)
        
        # Training mode: x should be concatenated pair [view1; view2]
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        
        # Forward through online network
        online_projections, _ = self.online_encoder(x)
        online_predictions = self.online_predictor(online_projections)
        
        # Split predictions into two views
        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)
        
        # Forward through target network (no gradients)
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projections, _ = target_encoder(x)
            target_projections = target_projections.detach()
            
            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)
        
        # Compute symmetric loss
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        
        loss = loss_one + loss_two
        return loss.mean()


def byol_loss(online_pred_1, online_pred_2, target_proj_1, target_proj_2):
    """
    BYOL loss for efficient model: normalized Mean Squared Error.
    """
    # L2 normalize all vectors
    online_pred_1 = F.normalize(online_pred_1, dim=-1, p=2)
    online_pred_2 = F.normalize(online_pred_2, dim=-1, p=2)
    target_proj_1 = F.normalize(target_proj_1, dim=-1, p=2)
    target_proj_2 = F.normalize(target_proj_2, dim=-1, p=2)
    
    # Compute symmetric MSE loss
    loss_1 = (2 - 2 * (online_pred_1 * target_proj_2).sum(dim=-1)).mean()
    loss_2 = (2 - 2 * (online_pred_2 * target_proj_1).sum(dim=-1)).mean()
    
    return loss_1 + loss_2

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ema_decay(epoch, num_epochs, schedule='constant', 
                  base_decay=0.996, start_decay=0.996, end_decay=0.9999):
    """Compute EMA decay rate for current epoch based on schedule."""
    if schedule == 'constant':
        return base_decay
    elif schedule == 'cosine':
        progress = epoch / max(num_epochs - 1, 1)
        return end_decay - (end_decay - start_decay) * (np.cos(np.pi * progress) + 1) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def get_warmup_lr(epoch, base_lr, warmup_epochs):
    """Compute learning rate during warmup phase."""
    if epoch >= warmup_epochs or warmup_epochs == 0:
        return base_lr
    else:
        return base_lr * (epoch + 1) / warmup_epochs


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
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
    
    # Update EMA decay for original model
    if MODEL_TYPE == "original":
        model.target_ema_updater.beta = current_ema_decay
    
    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for x1, x2, _ in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        
        # Forward pass (different for each model type)
        if MODEL_TYPE == "efficient":
            pred1, pred2, proj1, proj2 = model(x1, x2)
            loss = byol_loss(pred1, pred2, proj1, proj2)
        else:  # original
            images = torch.cat((x1, x2), dim=0)
            loss = model(images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (if enabled)
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        # Update target network
        if MODEL_TYPE == "efficient":
            model.update_target_network(momentum=current_ema_decay)
        else:  # original
            model.update_moving_average()
        
        # Track loss
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    
    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x1, x2, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ", leave=False):
            x1, x2 = x1.to(device), x2.to(device)
            
            if MODEL_TYPE == "efficient":
                pred1, pred2, proj1, proj2 = model(x1, x2)
                val_loss += byol_loss(pred1, pred2, proj1, proj2).item()
            else:  # original
                images = torch.cat((x1, x2), dim=0)
                val_loss += model(images).item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['lr'].append(current_lr)
    history['ema_decay'].append(current_ema_decay)
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    print(f"  EMA decay:  {current_ema_decay:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        print(f"  ✓ New best model (val_loss: {best_val_loss:.4f})")
    
    print()
    
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
    for x1, x2, _ in tqdm(test_loader, desc="Test"):
        x1, x2 = x1.to(device), x2.to(device)
        
        if MODEL_TYPE == "efficient":
            pred1, pred2, proj1, proj2 = model(x1, x2)
            test_loss += byol_loss(pred1, pred2, proj1, proj2).item()
        else:  # original
            images = torch.cat((x1, x2), dim=0)
            test_loss += model(images).item()

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
        'dataset': DATASET_NAME,
        'label_type': args.label_type,
    }
}, checkpoint_path)

print(f"✓ Model checkpoint saved to {checkpoint_path}")

# Save training history
np.save(OUTPUT_DIR / 'training_history.npy', history)
print(f"✓ Training history saved to {OUTPUT_DIR / 'training_history.npy'}")

# Plot training history (default behavior unless disabled)
if not args.no_plot_history:
    print("\nGenerating training curve plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training History - {MODEL_TYPE.upper()} Model', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].axhline(y=best_val_loss, color='g', linestyle='--', label=f'Best Val ({best_val_loss:.4f})', alpha=0.7)
    axes[0, 0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss Difference (Overfitting indicator)
    loss_diff = [val - train for train, val in zip(history['train_loss'], history['val_loss'])]
    axes[0, 1].plot(epochs, loss_diff, 'purple', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val Loss - Train Loss')
    axes[0, 1].set_title('Overfitting Indicator (Val - Train)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    axes[1, 0].plot(epochs, history['lr'], 'orange', linewidth=2)
    axes[1, 0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: EMA Decay
    axes[1, 1].plot(epochs, history['ema_decay'], 'green', linewidth=2)
    axes[1, 1].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('EMA Decay')
    axes[1, 1].set_title('EMA Decay Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = OUTPUT_DIR / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {plot_path}")
    
    # Also save a zoomed version (last 20% of training)
    if len(epochs) > 10:
        start_idx = int(len(epochs) * 0.8)
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle(f'Training History (Final 20%) - {MODEL_TYPE.upper()} Model', fontsize=14)
        
        epochs_zoom = list(epochs)[start_idx:]
        train_zoom = history['train_loss'][start_idx:]
        val_zoom = history['val_loss'][start_idx:]
        
        # Zoomed loss plot
        axes2[0].plot(epochs_zoom, train_zoom, 'b-', label='Train Loss', linewidth=2)
        axes2[0].plot(epochs_zoom, val_zoom, 'r-', label='Val Loss', linewidth=2)
        axes2[0].axhline(y=best_val_loss, color='g', linestyle='--', label=f'Best Val ({best_val_loss:.4f})', alpha=0.7)
        if best_epoch >= start_idx:  # Only show vertical line if best epoch is in zoomed range
            axes2[0].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})', alpha=0.7)
        axes2[0].set_xlabel('Epoch')
        axes2[0].set_ylabel('Loss')
        axes2[0].set_title('Loss (Zoomed)')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        # Zoomed loss difference
        loss_diff_zoom = loss_diff[start_idx:]
        axes2[1].plot(epochs_zoom, loss_diff_zoom, 'purple', linewidth=2)
        axes2[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        if best_epoch >= start_idx:  # Only show vertical line if best epoch is in zoomed range
            axes2[1].axvline(x=best_epoch, color='g', linestyle=':', linewidth=2, alpha=0.7)
        axes2[1].set_xlabel('Epoch')
        axes2[1].set_ylabel('Val Loss - Train Loss')
        axes2[1].set_title('Overfitting Indicator (Zoomed)')
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        zoom_plot_path = OUTPUT_DIR / 'training_curves_zoomed.png'
        plt.savefig(zoom_plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Zoomed training curves saved to {zoom_plot_path}")
    
    plt.close('all')

# =============================================================================
# EXTRACT EMBEDDINGS
# =============================================================================
def extract_embeddings_from_loader(model, dataloader, model_type, max_batches=None):
    """
    Extract representations and projections from a DataLoader.
    
    Args:
        model: Trained BYOL model
        dataloader: DataLoader that yields (x1, x2, mdist) tuples
        model_type: "efficient" or "original"
        max_batches: Limit number of batches (None = all)
    
    Returns:
        representations: (N, 512) representations from encoder
        projections: (N, 256) projections from projector head
    """
    model.eval()
    
    all_representations = []
    all_projections = []
    
    with torch.no_grad():
        for batch_idx, (x1, x2, _) in enumerate(tqdm(dataloader, desc="Extracting")):
            if max_batches and batch_idx >= max_batches:
                break
            
            x1 = x1.to(device)
            
            if model_type == "efficient":
                # Extract from efficient model
                representation = model.online_encoder(x1)
                projection = model.online_projector(representation)
            else:  # original
                # Extract from original model using return_embedding
                projection, representation = model(x1, return_embedding=True, return_projection=True)
            
            all_representations.append(representation.cpu().numpy())
            all_projections.append(projection.cpu().numpy())
    
    representations = np.vstack(all_representations)
    projections = np.vstack(all_projections)
    
    return representations, projections

print("\nExtracting embeddings from DataLoaders...")

# Extract from train loader
print("\n  Train set:")
train_representations, train_projections = extract_embeddings_from_loader(
    model, train_loader, MODEL_TYPE, max_batches=None
)
print(f"    Representations: {train_representations.shape}")
print(f"    Projections: {train_projections.shape}")

# Extract from val loader
print("\n  Val set:")
val_representations, val_projections = extract_embeddings_from_loader(
    model, val_loader, MODEL_TYPE, max_batches=None
)
print(f"    Representations: {val_representations.shape}")
print(f"    Projections: {val_projections.shape}")

# Extract from test loader
print("\n  Test set:")
test_representations, test_projections = extract_embeddings_from_loader(
    model, test_loader, MODEL_TYPE, max_batches=None
)
print(f"    Representations: {test_representations.shape}")
print(f"    Projections: {test_projections.shape}")

# Save embeddings
embeddings_dir = OUTPUT_DIR / 'embeddings'
embeddings_dir.mkdir(exist_ok=True)

np.save(embeddings_dir / 'train_representations.npy', train_representations)
np.save(embeddings_dir / 'train_projections.npy', train_projections)
np.save(embeddings_dir / 'val_representations.npy', val_representations)
np.save(embeddings_dir / 'val_projections.npy', val_projections)
np.save(embeddings_dir / 'test_representations.npy', test_representations)
np.save(embeddings_dir / 'test_projections.npy', test_projections)

# Save corresponding labels
np.save(embeddings_dir / 'train_labels.npy', train_labels[:len(train_representations)])
np.save(embeddings_dir / 'val_labels.npy', val_labels[:len(val_representations)])
np.save(embeddings_dir / 'test_labels.npy', test_labels[:len(test_representations)])

print(f"\n✓ Embeddings saved to {embeddings_dir}/")

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
    
    # Function to identify pure class samples
    def get_pure_class_colors(labels, label_type):
        """
        Assign colors to pure class samples, grey for non-pure.
        
        Args:
            labels: (N, D) array of labels for current classification type
            label_type: 'initial', 'morphology', 'environment', or 'derived'
        
        Returns:
            colors: (N,) array of color indices (-1 for grey, 0-K for pure classes)
            class_names: List of class names for legend
        """
        n_samples, n_classes = labels.shape
        colors = np.full(n_samples, -1, dtype=int)  # -1 = grey (non-pure)
        
        # Check each sample: pure if exactly one 1, rest 0s
        for i in range(n_samples):
            label_vec = labels[i]
            n_ones = np.sum(label_vec == 1)
            
            # Pure class: exactly one 1
            if n_ones == 1:
                class_idx = np.where(label_vec == 1)[0][0]
                colors[i] = class_idx
        
        class_names = CLASS_NAMES.get(label_type, [f"Class {i}" for i in range(n_classes)])
        return colors, class_names
    
    # Function to plot UMAP with pure class colors
    def plot_umap_pure_classes(embeddings, labels, title_suffix, save_prefix, split_name):
        """Generate UMAP plots for each classification type with pure class coloring"""
        print(f"  Computing UMAP for {title_suffix}...")
        
        # Fit UMAP once for all classification types
        reducer = umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric='euclidean',
            random_state=SEED
        )
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Generate one plot per classification type
        for class_type in ['initial', 'morphology', 'environment', 'derived']:
            # Get label range for this classification type
            label_start, label_end = LABEL_RANGES[class_type]
            
            # Extract labels for this classification type from FULL labels
            # Need to reload full labels for proper indexing
            if args.label_type == 'full':
                # Already have full labels
                type_labels = labels[:, label_start:label_end]
            else:
                # Need to reconstruct from saved full labels
                # Load the appropriate full labels based on split
                if split_name == 'train':
                    full_labels = train_labels_full[:len(embeddings)]
                elif split_name == 'test':
                    full_labels = test_labels_full[:len(embeddings)]
                else:
                    continue  # Skip val set for UMAP
                
                type_labels = full_labels[:, label_start:label_end]
            
            # Get pure class colors
            colors, class_names = get_pure_class_colors(type_labels, class_type)
            
            # Count pure vs non-pure samples
            n_pure = np.sum(colors >= 0)
            n_nonpure = np.sum(colors == -1)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            fig.suptitle(f'UMAP - {title_suffix} - {class_type.capitalize()} Classification', 
                        fontsize=14)
            
            # Plot non-pure samples in grey first (background)
            mask_nonpure = colors == -1
            if np.any(mask_nonpure):
                ax.scatter(
                    embedding_2d[mask_nonpure, 0],
                    embedding_2d[mask_nonpure, 1],
                    c='lightgrey',
                    s=10,
                    alpha=0.3,
                    label=f'Non-pure ({n_nonpure})'
                )
            
            # Plot pure classes with distinct colors
            n_classes = len(class_names)
            cmap = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
            
            for class_idx in range(n_classes):
                mask_class = colors == class_idx
                n_class = np.sum(mask_class)
                
                if n_class > 0:
                    ax.scatter(
                        embedding_2d[mask_class, 0],
                        embedding_2d[mask_class, 1],
                        c=[cmap(class_idx)],
                        s=15,
                        alpha=0.7,
                        label=f'{class_names[class_idx]} ({n_class})'
                    )
            
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'{n_pure} pure samples, {n_nonpure} non-pure')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot to OUTPUT_DIR (not embeddings_dir)
            umap_path = OUTPUT_DIR / f'{save_prefix}_{class_type}.png'
            plt.savefig(umap_path, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved {class_type} to {umap_path}")
            plt.close()
    
    # Store full labels before any filtering (needed for all classification types)
    # Reload full labels
    labels_full = np.load(LABELS_PATH)
    train_labels_full = labels_full[train_idx]
    test_labels_full = labels_full[test_idx]
    
    # Plot UMAPs for train representations
    plot_umap_pure_classes(
        train_representations,
        train_labels[:len(train_representations)],
        "Train Representations (512-dim)",
        "umap_train_representations",
        "train"
    )
    
    # Plot UMAPs for test representations
    plot_umap_pure_classes(
        test_representations,
        test_labels[:len(test_representations)],
        "Test Representations (512-dim)",
        "umap_test_representations",
        "test"
    )
    
    # Plot UMAPs for train projections
    plot_umap_pure_classes(
        train_projections,
        train_labels[:len(train_projections)],
        "Train Projections (256-dim)",
        "umap_train_projections",
        "train"
    )
    
    # Plot UMAPs for test projections
    plot_umap_pure_classes(
        test_projections,
        test_labels[:len(test_projections)],
        "Test Projections (256-dim)",
        "umap_test_projections",
        "test"
    )
    
    print(f"\n✓ UMAP plots saved to {OUTPUT_DIR}/")

print(f"\n{'='*70}")
print(f"SCRIPT COMPLETE")
print(f"{'='*70}")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"{'='*70}\n")