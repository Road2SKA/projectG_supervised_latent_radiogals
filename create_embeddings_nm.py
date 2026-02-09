#!/usr/bin/env python3
"""
BYOL Implementation for Radio Galaxy Classification
Training script for SLURM GPU submission
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

# =============================================================================
# HELPER FUNCTIONS (from snippet)
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
# AUGMENTATION UTILS (from snippet)
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
# EMA CLASS (from snippet)
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
# MLP HEADS (from snippet)
# =============================================================================

def MLP(dim, projection_size, hidden_size=4096):
    """Standard MLP projection head for BYOL"""
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# =============================================================================
# LOSS FUNCTION (from snippet)
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
    
    # Data configuration
    ap.add_argument("--data-dir", type=Path, 
                    default=Path('/idia/projects/roadtoska/projectG/projectG_supervised_latent_radiogals'),
                    help="Root directory containing images.npy and labels.npy")
    ap.add_argument("--dataset", type=str, default="LOTSS",
                    choices=["LOTSS", "MOCK"],
                    help="Dataset to use: LOTSS (real data) or MOCK (synthetic data)")
    
    # Training hyperparameters
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for training (default: 32)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 0.0003)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs (default: 100)")
    ap.add_argument("--ema-decay", type=float, default=0.99,
                    help="EMA decay rate for target network (default: 0.99)")
    
    # Dataset pairing strategy
    ap.add_argument("--weighting", type=str, default="closest",
                    choices=["closest", "ponderate"],
                    help="Weight function for sampling pairs: 'closest' or 'ponderate' (default: closest)")
    ap.add_argument("--prob", type=float, default=0.5,
                    help="Probability of pairing from same class (default: 0.5)")
    
    # Model architecture
    ap.add_argument("--projection-dim", type=int, default=256,
                    help="Projection head output dimension (default: 256)")
    ap.add_argument("--hidden-dim", type=int, default=4096,
                    help="Hidden layer dimension in MLP heads (default: 4096)")
    
    # Output configuration
    ap.add_argument("--output-dir", type=Path,
                    default=Path('/idia/projects/roadtoska/projectG/projectG_supervised_latent_radiogals/outputs'),
                    help="Base output directory for checkpoints and embeddings")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Custom run name (default: timestamp)")
    
    # Data subsampling
    ap.add_argument("--subsample", type=int, default=None,
                    help="Subsample dataset to N samples (for quick testing)")
    
    # Checkpointing
    ap.add_argument("--checkpoint-freq", type=int, default=10,
                    help="Save checkpoint every N epochs (default: 10)")
    
    # Random seed
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    
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
    RUN_ID += f"_w{args.weighting}_p{P_PAIR_FROM_CLASS}"

OUTPUT_DIR = OUTPUT_BASE / f'run_{RUN_ID}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = OUTPUT_DIR / 'byol_model_best.pt'

print(f"\n{'='*70}")
print(f"CONFIGURATION")
print(f"Output directory: {OUTPUT_DIR}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"{'='*70}")
print(f"Dataset:        {DATASET_NAME}")
print(f"Data dir:       {DATA_DIR}")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Learning rate:  {LEARNING_RATE}")
print(f"Epochs:         {NUM_EPOCHS}")
print(f"EMA decay:      {EMA_DECAY}")
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
# CREATE DATASETS WITH SNIPPET-STYLE AUGMENTATIONS
# =============================================================================
print("\nCreating datasets...")

# Convert numpy arrays to DataFrames
train_labels_df = pd.DataFrame(train_labels)
val_labels_df = pd.DataFrame(val_labels)
test_labels_df = pd.DataFrame(test_labels)

print(f"  Converted labels to DataFrames")

# Default augmentation from snippet (adapted for greyscale)
# Note: ColorJitter and Normalize are designed for RGB, so we skip them for greyscale
DEFAULT_AUG = torch.nn.Sequential(
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(180),
)

train_dataset = BYOLSupDataset(
    tags_data=train_labels_df,
    img_data=train_images,
    transform=DEFAULT_AUG,
    friend_transform=DEFAULT_AUG,
    weightfunc=WEIGHTING_FUNC,
    p_pair_from_class=P_PAIR_FROM_CLASS
)

val_dataset = BYOLSupDataset(
    tags_data=val_labels_df,
    img_data=val_images,
    transform=DEFAULT_AUG,
    friend_transform=DEFAULT_AUG,
    weightfunc=WEIGHTING_FUNC,  
    p_pair_from_class=P_PAIR_FROM_CLASS
)

test_dataset = BYOLSupDataset(
    tags_data=test_labels_df,
    img_data=test_images,
    transform=DEFAULT_AUG,
    friend_transform=DEFAULT_AUG,
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
# BASE ENCODER (ResNet-style for 89x89 greyscale)
# =============================================================================
class BYOLEncoder(nn.Module):
    """ResNet-style encoder for 89x89 greyscale images"""
    def __init__(self):
        super().__init__()
        
        # Initial conv: 89x89 -> 45x45
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks: 45x45 -> 23x23 -> 12x12 -> 6x6
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        
        # Global pooling: 6x6 -> 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to match expected feature dimension
        self.fc = nn.Linear(512, 1280)
        
    def _make_layer(self, in_channels, out_channels, stride=1):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Final projection
        x = self.fc(x)
        return x

# =============================================================================
# NET WRAPPER (from snippet - adapted for direct encoder)
# =============================================================================
class NetWrapper(nn.Module):
    """
    Wrapper for base network following snippet pattern.
    Manages projection head and representation extraction.
    """
    def __init__(self, net, projection_size, projection_hidden_size):
        super().__init__()
        self.net = net
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
    
    @singleton('projector')
    def _get_projector(self, hidden):
        """Create projector on first forward pass (singleton pattern)"""
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
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
        
        Returns:
            If return_projection=True: (projection, representation) tuple
            If return_projection=False: representation tensor
        """
        representation = self.get_representation(x)
        
        if not return_projection:
            return representation
        
        # Get or create projector (singleton pattern ensures single creation)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# =============================================================================
# BYOL MODEL (following snippet architecture exactly)
# =============================================================================
class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) following original implementation.
    Uses EMA for target network update.
    """
    def __init__(
        self,
        net,
        image_size,
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        use_momentum=True
    ):
        super().__init__()
        self.net = net
        
        # Online encoder with projection head
        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size
        )
        
        # Target encoder (EMA of online)
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        
        # Predictor (only for online network)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        
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
            x: Input batch (can be single view or will be augmented internally)
            return_embedding: If True, return embeddings instead of loss
            return_projection: If True (with return_embedding), return projections too
        
        Returns:
            If return_embedding=True: (projection, representation) or representation
            If return_embedding=False: BYOL loss (scalar)
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
        
        # Compute symmetric loss: predict1 -> target2, predict2 -> target1
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        
        loss = loss_one + loss_two
        return loss.mean()

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print("\nInitializing model...")
encoder = BYOLEncoder()
model = BYOL(
    encoder, 
    image_size=89,
    projection_size=PROJECTION_DIM, 
    projection_hidden_size=HIDDEN_DIM,
    moving_average_decay=EMA_DECAY,
    use_momentum=True
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"{'='*70}")
print(f"MODEL ARCHITECTURE")
print(f"{'='*70}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Encoder output dim:   1280")
print(f"Projection dim:       {PROJECTION_DIM}")
print(f"Hidden dim:           {HIDDEN_DIM}")
print(f"{'='*70}\n")

if use_cuda:
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.0f} MB")

# =============================================================================
# TRAINING SETUP
# =============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'lr': []
}

best_val_loss = float('inf')
best_model_state = None

print(f"✓ Optimizer: Adam (lr={LEARNING_RATE})")
print(f"✓ Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
print(f"✓ Loss: BYOL symmetric MSE (from snippet)")

# =============================================================================
# TRAINING LOOP
# =============================================================================
print(f"\n{'='*70}")
print(f"STARTING TRAINING")
print(f"{'='*70}\n")

for epoch in range(NUM_EPOCHS):
    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for x1, x2, _ in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        
        # Concatenate views as expected by snippet's forward method
        images = torch.cat((x1, x2), dim=0)
        
        # Forward pass (returns loss directly)
        loss = model(images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target network using EMA
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
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    
    # Save checkpoint periodically
    if (epoch + 1) % args.checkpoint_freq == 0:
        checkpoint_periodic = OUTPUT_DIR / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'avg_val_loss': avg_val_loss,
            'history': history,
            'config': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
                'ema_decay': EMA_DECAY,
                'projection_dim': PROJECTION_DIM,
                'hidden_dim': HIDDEN_DIM,
                'weighting': args.weighting,
                'p_pair_from_class': P_PAIR_FROM_CLASS,
                'dataset': DATASET_NAME,
            }
        }, checkpoint_periodic)
        print(f"  Checkpoint saved: {checkpoint_periodic}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"  ✓ New best model (val_loss: {best_val_loss:.4f})")
    
    print()
    
    # Step scheduler
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

# Save final model checkpoint
torch.save({
    'model_state_dict': best_model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': NUM_EPOCHS,
    'best_val_loss': best_val_loss,
    'test_loss': avg_test_loss,
    'history': history,
    'config': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'ema_decay': EMA_DECAY,
        'projection_dim': PROJECTION_DIM,
        'hidden_dim': HIDDEN_DIM,
        'weighting': args.weighting,
        'p_pair_from_class': P_PAIR_FROM_CLASS,
        'dataset': DATASET_NAME,
    }
}, checkpoint_path)

print(f"✓ Model checkpoint saved to {checkpoint_path}")

# Save training history
np.save(OUTPUT_DIR / 'training_history.npy', history)
print(f"✓ Training history saved to {OUTPUT_DIR / 'training_history.npy'}")

# =============================================================================
# EXTRACT EMBEDDINGS
# =============================================================================
def extract_embeddings_from_loader(model, dataloader, max_batches=None):
    """
    Extract embeddings from a DataLoader.
    
    Args:
        model: Trained BYOL model
        dataloader: DataLoader yielding (x1, x2, mdist) tuples
        max_batches: Limit number of batches (None = all)
    
    Returns:
        embeddings: (N, 1280) features from backbone
        projections: (N, 256) projected features
    """
    model.eval()
    
    all_embeddings = []
    all_projections = []
    
    with torch.no_grad():
        for batch_idx, (x1, x2, _) in enumerate(tqdm(dataloader, desc="Extracting")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Use x1 (either view works for feature extraction)
            x1 = x1.to(device)
            
            # Extract features using return_embedding=True
            projections, features = model(x1, return_embedding=True, return_projection=True)
            
            all_embeddings.append(features.cpu().numpy())
            all_projections.append(projections.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    projections = np.vstack(all_projections)
    
    return embeddings, projections

print("\nExtracting embeddings from DataLoaders...")

# Extract from train loader
print("\n  Train set:")
train_embeddings, train_projections = extract_embeddings_from_loader(
    model, train_loader, max_batches=None
)
print(f"    Embeddings: {train_embeddings.shape}")
print(f"    Projections: {train_projections.shape}")

# Extract from val loader
print("\n  Val set:")
val_embeddings, val_projections = extract_embeddings_from_loader(
    model, val_loader, max_batches=None
)
print(f"    Embeddings: {val_embeddings.shape}")
print(f"    Projections: {val_projections.shape}")

# Extract from test loader
print("\n  Test set:")
test_embeddings, test_projections = extract_embeddings_from_loader(
    model, test_loader, max_batches=None
)
print(f"    Embeddings: {test_embeddings.shape}")
print(f"    Projections: {test_projections.shape}")

# Save embeddings
embeddings_dir = OUTPUT_DIR / 'embeddings'
embeddings_dir.mkdir(exist_ok=True)

np.save(embeddings_dir / 'train_embeddings.npy', train_embeddings)
np.save(embeddings_dir / 'train_projections.npy', train_projections)
np.save(embeddings_dir / 'val_embeddings.npy', val_embeddings)
np.save(embeddings_dir / 'val_projections.npy', val_projections)
np.save(embeddings_dir / 'test_embeddings.npy', test_embeddings)
np.save(embeddings_dir / 'test_projections.npy', test_projections)

# Save corresponding labels
np.save(embeddings_dir / 'train_labels.npy', train_labels[:len(train_embeddings)])
np.save(embeddings_dir / 'val_labels.npy', val_labels[:len(val_embeddings)])
np.save(embeddings_dir / 'test_labels.npy', test_labels[:len(test_embeddings)])

print(f"\n✓ Embeddings saved to {embeddings_dir}/")

print(f"\n{'='*70}")
print(f"SCRIPT COMPLETE")
print(f"{'='*70}")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"{'='*70}\n")