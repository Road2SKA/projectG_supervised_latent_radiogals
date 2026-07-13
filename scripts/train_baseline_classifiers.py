"""
Train end-to-end baseline classifiers for radio galaxy classification.

Usage:
    python train_baseline_classifiers.py \\
        --run_dir /path/to/byol/run \\
        --data_dir /path/to/data \\
        --model cnn \\
        --label_set classical

Models: cnn | scatternet | simplescatternet | vit | dualssn | enb0
"""

import sys
import scipy.special

# Mock the missing function so Kymatio doesn't crash on import
if not hasattr(scipy.special, 'sph_harm'):
    def dummy_sph_harm(*args, **kwargs):
        raise NotImplementedError(
            "sph_harm was mocked because it was missing from scipy.special. "
            "This should only happen if you are trying to use 3D scattering, "
            "but you are using 2D."
        )
    scipy.special.sph_harm = dummy_sph_harm

# Now proceed with your original imports
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from suplat.utils.class_weights import compute_class_weights
from suplat.models.baseline_models import (
    CNN, ScatterNet, SimpleScatterNet, DualScatterSqueezeNet
)
from suplat.models.byol_models import create_efficientnet_b0_backbone

# ── Label sets (same as classifiers.ipynb) ───────────────────────────────────
ALL_CLASS_NAMES = [
    'FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles',
    'C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
    'Straight jets', 'Multi hotspots', 'Cont. jets', 'Banding',
    'One-sided', 'Restarted', 'Cluster', 'Merger', 'Diffuse', 'Unknown',
]

LABEL_SETS = {
    "classical":      [0, 1],
    "classical_pure": [0, 1],
    "initial":        list(range(0, 5)),
    "initial_pure":   list(range(0, 5)),
    "environment":    list(range(16, 20)),
    "derived":        None,          # computed from label combinations
    "morphology":     list(range(5, 16)),
    "all":            list(range(0, 20)),
    "pure":           list(range(0, 20)),
    "full":           list(range(0, 20)),
}

VAL_FRAC = 0.15


# ── Dataset ───────────────────────────────────────────────────────────────────
class RadioImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # images: (N, H, W) float32 or (N, 1, H, W)
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]  # add channel dim
        self.images = torch.from_numpy(images.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[idx]


class ScatterDataset(Dataset):
    """Dataset that returns (image, scat_coeff, label) triples."""
    def __init__(self, images, scat, labels):
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]
        self.images = torch.from_numpy(images.astype(np.float32))
        self.scat   = torch.from_numpy(scat.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.scat[idx], self.labels[idx]


# ── Training helpers ──────────────────────────────────────────────────────────
def weighted_class_mean_loss(logits, targets, alpha, n_classes):
    """Weighted mean of cross-entropy over classes.

    logits  : (B, n_classes, 2)
    targets : (B, n_classes) int64
    alpha   : (n_classes,) float32 tensor, already on device
    Returns : scalar loss
    """
    per_class = torch.stack([
        F.cross_entropy(logits[:, c, :], targets[:, c], reduction='none')
        for c in range(n_classes)
    ], dim=1)                                      # (B, n_classes)
    alpha_sum = alpha.sum().clamp(min=1e-6)
    return ((per_class * alpha).sum(1) / alpha_sum).mean()


def evaluate_metrics(y_true, y_pred, y_prob, class_names):
    n = len(class_names)
    aucs = []
    for i in range(n):
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(float('nan'))
        else:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))

    f1_per  = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_mac  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec_mac = recall_score(y_true, y_pred, average='macro', zero_division=0)
    acc     = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))
    mac_auc = float(np.nanmean(aucs))

    return {
        'f1_macro':         float(f1_mac),
        'auc_macro':        mac_auc,
        'accuracy':         float(acc),
        'recall_macro':     float(rec_mac),
        'f1_per_class':     f1_per,
        'auc_per_class':    [float(a) if not np.isnan(a) else None for a in aucs],
        'recall_per_class': rec_per,
        'class_names':      class_names,
    }


def make_loader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# ── Forward pass wrappers ─────────────────────────────────────────────────────
def forward_model(model, batch, model_name, device):
    """Unified forward for single-input and dual-input models."""
    if model_name == 'dualssn':
        imgs, scats, labels = batch
        imgs, scats, labels = imgs.to(device), scats.to(device), labels.to(device)
        logits = model(imgs, scats)
        return logits, labels
    else:
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        return logits, labels


class MultiLabelWrapper(nn.Module):
    def __init__(self, inner, n_cl):
        super().__init__()
        self.inner = inner
        self.n_cl  = n_cl

    def forward(self, *args):
        out = self.inner(*args)       # (B, n_classes * 2)
        return out.view(out.size(0), self.n_cl, 2)


class DualWrapper(nn.Module):
    def __init__(self, inner, n_cl):
        super().__init__()
        self.inner = inner
        self.n_cl  = n_cl

    def forward(self, img, scat):
        out = self.inner(img, scat)
        return out.view(out.size(0), self.n_cl, 2)


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(model_name, n_classes, img_shape, scat_shape=None):
    """Returns a model whose output is (B, n_classes, 2) for multi-label CE."""
    n_out = n_classes * 2

    if model_name == 'cnn':
        base = CNN(img_shape, num_classes=n_out)
    elif model_name == 'scatternet':
        assert scat_shape is not None
        base = ScatterNet(scat_shape, num_classes=n_out)
    elif model_name == 'simplescatternet':
        assert scat_shape is not None
        base = SimpleScatterNet(scat_shape, num_classes=n_out)
    elif model_name == 'dualssn':
        assert scat_shape is not None
        base = DualScatterSqueezeNet(img_shape, scat_shape, num_classes=n_out)
    elif model_name == 'vit':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Average first conv weights across 3 channels → 1 channel
        conv = vit.conv_proj
        new_conv = nn.Conv2d(1, conv.out_channels,
                             kernel_size=conv.kernel_size,
                             stride=conv.stride,
                             padding=conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        vit.conv_proj = new_conv
        vit.heads = nn.Linear(768, n_out)
        base = vit
    elif model_name == 'enb0':
        backbone = create_efficientnet_b0_backbone(num_channels=1)
        base = nn.Sequential(
            backbone,
            nn.Flatten(),
            nn.Linear(1280, n_out)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == 'dualssn':
        return DualWrapper(base, n_classes)

    return MultiLabelWrapper(base, n_classes)


# ── Scattering helper ─────────────────────────────────────────────────────────
def compute_scattering(images, J=2, L=8, device='cpu'):
    """Compute Kymatio Scattering2D coefficients for all images."""
    from kymatio.torch import Scattering2D
    H, W = images.shape[-2], images.shape[-1]
    scat = Scattering2D(J=J, shape=(H, W), L=L).to(device)

    dummy = torch.zeros(1, 1, H, W, device=device)
    with torch.no_grad():
        scat_out = scat(dummy)  # (1, 1, C_scat, H_s, W_s)
        scat_shape = tuple(scat_out.squeeze(0).flatten(0, 1).shape)  # (C_scat, H_s, W_s)
    print(f"  Scattering shape: {scat_shape}")

    if images.ndim == 3:
        imgs_t = torch.from_numpy(images.astype(np.float32))[:, None, :, :]
    else:
        imgs_t = torch.from_numpy(images.astype(np.float32))

    all_coeffs = []
    batch_size = 128
    for i in range(0, len(imgs_t), batch_size):
        batch = imgs_t[i:i + batch_size].to(device)
        with torch.no_grad():
            coeffs = scat(batch)  # (B, 1, C_scat, H_s, W_s)
        all_coeffs.append(coeffs.flatten(1, 2).cpu())  # (B, C_scat, H_s, W_s)
    return torch.cat(all_coeffs, dim=0).numpy(), scat_shape


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir',    required=True, type=Path)
    parser.add_argument('--data_dir',   required=True, type=Path)
    parser.add_argument('--model',      default='enb0',
                        choices=['cnn', 'scatternet', 'simplescatternet',
                                 'vit', 'dualssn', 'enb0'])
    parser.add_argument('--label_set',  default='initial_pure',
                        choices=list(LABEL_SETS.keys()))
    parser.add_argument('--eval_fri_frii_pure', action='store_true',
                        help='(with --label_set full) evaluate on FRI/FRII-pure sources only')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int, default=15)
    parser.add_argument('--cv_folds',   type=int, default=1)
    parser.add_argument('--n_runs', type=int, default=1,
                        help="Number of training runs with consecutive seeds "
                             "(seed, seed+1, …). Enables error bars and ensembling. "
                             "Default: 1.")
    parser.add_argument('--run_name',   type=str, default=None,
                        help="Custom run name prefix (default: run_dir basename + timestamp)")
    parser.add_argument('--force', action='store_true',
                        help="Retrain even if outputs already exist.")
    parser.add_argument('--skip_sweep', action='store_true',
                        help="Skip the label-fraction sweep entirely.")
    parser.add_argument('--class_weight_mode', type=str, default='initial',
                        choices=["score", "initial", "morphology", "environment", "classical", "all"],
                        help="Upweight rare samples: 'score' (interest tier 1-4) or label-set name "
                             "(inverse frequency). Default: 'initial'.")
    parser.add_argument('--class_weight_strength', type=float, default=1.0,
                        help="Magnitude of class upweighting (0=uniform, 1.0=fully balanced). "
                             "Default: 1.0.")
    parser.add_argument('--byol_run_dir', type=Path, default=None,
                        help="BYOL run directory whose data/train_idx.npy + data/test_idx.npy "
                             "define the train/test split (default: --run_dir itself).")
    parser.add_argument('--data_seed', type=int, default=None,
                        help="Data seed used to locate data_splits/<seed>/ directly, "
                             "without needing a BYOL checkpoint. Overrides --byol_run_dir "
                             "split resolution if provided.")
    parser.add_argument('--gen_dir', type=Path, default=None,
                        help="Path to generative model directory containing decoder.pt and nsf.pt. "
                             "When provided, skips the label-fraction sweep and instead trains the "
                             "classifier augmented with generated images at fractions 0.5, 1.0, and 2.0 "
                             "of the real training set size, repeated n_runs times each.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader worker processes for data loading (default: 4).")
    parser.add_argument('--compile', action='store_true',
                        help="Apply torch.compile() to the model for faster GPU execution.")
    args = parser.parse_args()
    if args.byol_run_dir is None:
        args.byol_run_dir = args.run_dir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Shadow module-level make_loader with num_workers from args
    _nw = args.num_workers
    make_loader = lambda ds, bs, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=bs, shuffle=shuffle, num_workers=_nw, pin_memory=True
    )

    # ── Output directory ──────────────────────────────────────────────────
    _prefix = args.run_name if args.run_name else args.run_dir.name
    _dir_name = f'{_prefix}_{args.model}'
    _existing = sorted(args.run_dir.glob(f'{_dir_name}_*'))
    if _existing and not args.force:
        out_dir = _existing[-1]
        print(f"Resuming existing run: {out_dir.name}  (use --force to retrain from scratch)")
    else:
        _ts = datetime.now().strftime('%Y%m%d_%H%M')
        out_dir = args.run_dir / f'{_dir_name}_{_ts}'
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {out_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    images = np.load(args.data_dir / 'images_filtered.npy')
    labels = np.load(args.data_dir / 'labels_filtered.npy')
    print(f"Images: {images.shape}, Labels: {labels.shape}")

    # ── Label subset ──────────────────────────────────────────────────────
    DERIVED_CLASS_NAMES = [
        'Pure hybrid',       # col2 & ~col0 & ~col1
        'FR hybrid',         # col2 & (col0 | col1)
        'Curved FRI',        # col0 & (col5 | col6)
        'Curved FRII',       # col1 & (col5 | col6)
        'Straight+multi-HS', # col10 & col11
    ]

    def _make_derived(y):
        c = lambda i: y[:, i].astype(bool)
        return np.stack([
            ( c(2) & ~c(0) & ~c(1)).astype(np.int64),
            ( c(2) &  (c(0) | c(1))).astype(np.int64),
            ( c(0) &  (c(5) | c(6))).astype(np.int64),
            ( c(1) &  (c(5) | c(6))).astype(np.int64),
            (c(10) &   c(11)).astype(np.int64),
        ], axis=1)

    label_cols  = LABEL_SETS[args.label_set]
    if args.label_set == "derived":
        class_names = DERIVED_CLASS_NAMES
    else:
        class_names = [ALL_CLASS_NAMES[i] for i in label_cols]
    n_classes   = len(class_names)
    print(f"Label set: {args.label_set} ({n_classes} classes: {class_names})")

    # ── Train/test split ──────────────────────────────────────────────────
    if args.data_seed is not None:
        _splits_dir = args.byol_run_dir.parent.parent / "data_splits" / str(args.data_seed)
        split_label = f"data_seed={args.data_seed}"
    else:
        byol_data = args.byol_run_dir / "data"
        _ckpt_raw = torch.load(args.byol_run_dir / "byol_model_best.pt",
                               map_location="cpu", weights_only=False)
        _data_seed  = int(_ckpt_raw["config"]["data_seed"])
        _splits_dir = args.byol_run_dir.parent / "data_splits" / str(_data_seed)
        byol_data   = args.byol_run_dir / "data"
        split_label = args.byol_run_dir.name
    def _load_idx(name):
        p = _splits_dir / name
        if args.data_seed is None:
            return np.load(p if p.exists() else byol_data / name)
        return np.load(p)
    trainval_idx = _load_idx("train_idx.npy")
    test_idx     = _load_idx("test_idx.npy")
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=VAL_FRAC, random_state=args.seed
    )
    print(f"Using split from: {split_label}")
    print(f"  train+val: {len(trainval_idx)}  test: {len(test_idx)}")

    test_images = images[test_idx]
    if args.label_set == "derived":
        test_labels = _make_derived(labels[test_idx])
    else:
        test_labels = labels[test_idx][:, label_cols]

    # ── Pure-source filtering (test set only for "full", both splits otherwise)
    if args.label_set == "pure":
        test_pure_mask  = labels[test_idx].sum(axis=1) == 1
        test_images     = test_images[test_pure_mask]
        test_labels     = test_labels[test_pure_mask]
        print(f"Pure filter (all 20): {test_pure_mask.sum()} test retained")
    elif args.label_set in ("classical_pure", "initial_pure"):
        test_pure_mask  = labels[test_idx][:, 0:5].sum(axis=1) == 1
        test_images     = test_images[test_pure_mask]
        test_labels     = test_labels[test_pure_mask]
        print(f"Initial-pure filter : {test_pure_mask.sum()} test retained")
    elif args.label_set == "full" and args.eval_fri_frii_pure:
        _y_te = labels[test_idx]
        test_pure_mask  = (_y_te[:, 0:5].sum(axis=1) == 1) & (_y_te[:, 0] | _y_te[:, 1]).astype(bool)
        test_images     = test_images[test_pure_mask]
        test_labels     = test_labels[test_pure_mask]
        print(f"FRI/FRII-pure eval filter : {test_pure_mask.sum()} test retained")

    print(f"Test set: {len(test_images)}")

    # ── Fold definitions ──────────────────────────────────────────────────
    if args.cv_folds == 1:
        fold_splits = [(train_idx, val_idx)]
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = [
            (trainval_idx[tr], trainval_idx[va])
            for tr, va in kf.split(trainval_idx)
        ]
        print(f"K-fold CV: {args.cv_folds} folds on {len(trainval_idx)} trainval samples")

    all_run_results = []

    if args.gen_dir is not None:
        print("Skipping supervised training loop (--gen_dir mode).")
    for run_i in range(1, args.n_runs + 1) if args.gen_dir is None else []:
        run_seed = args.seed + (run_i - 1)
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        run_out = out_dir / f'run{run_i}' if args.n_runs > 1 else out_dir
        run_out.mkdir(parents=True, exist_ok=True)

        if args.n_runs > 1:
            print(f"\n{'='*60}\nRun {run_i} / {args.n_runs}  (seed={run_seed})\n{'='*60}")

        fold_results_list = []

        for fold_i, (fold_train_idx, fold_val_idx) in enumerate(fold_splits):
          fold_out = run_out / f'fold{fold_i + 1}' if args.cv_folds > 1 else run_out
          if (fold_out / 'results.json').exists() and not args.force:
              print(f"  [cached] Run {run_i} — loading from {fold_out.name}")
              with open(fold_out / 'results.json') as _f:
                  results = json.load(_f)
              fold_results_list.append(results)
              continue

          if args.cv_folds > 1:
              print(f"\n{'='*60}\nFold {fold_i + 1} / {args.cv_folds}\n{'='*60}")

          train_images2 = images[fold_train_idx]
          val_images    = images[fold_val_idx]
          if args.label_set == "derived":
              train_labels2 = _make_derived(labels[fold_train_idx])
              val_labels    = _make_derived(labels[fold_val_idx])
          else:
              train_labels2 = labels[fold_train_idx][:, label_cols]
              val_labels    = labels[fold_val_idx][:, label_cols]

          # Pure-source filtering on training and val splits
          tr_mask = np.ones(len(fold_train_idx), dtype=bool)
          if args.label_set == "pure":
              tr_mask = labels[fold_train_idx].sum(axis=1) == 1
              va_mask = labels[fold_val_idx].sum(axis=1)   == 1
              train_images2, train_labels2 = train_images2[tr_mask], train_labels2[tr_mask]
              val_images,    val_labels    = val_images[va_mask],    val_labels[va_mask]
          elif args.label_set in ("classical_pure", "initial_pure"):
              tr_mask = labels[fold_train_idx][:, 0:5].sum(axis=1) == 1
              va_mask = labels[fold_val_idx][:, 0:5].sum(axis=1)   == 1
              train_images2, train_labels2 = train_images2[tr_mask], train_labels2[tr_mask]
              val_images,    val_labels    = val_images[va_mask],    val_labels[va_mask]

          # Per-class alpha (computed from full 20-col labels, filtered consistently)
          if args.label_set == 'derived':
              _dc = train_labels2.sum(axis=0).astype(float)
              _mean_dc = _dc.mean()
              if args.class_weight_mode is None or args.class_weight_strength == 0.0:
                  _alpha_arr = np.ones(n_classes, dtype=np.float32)
              else:
                  _alpha_arr = (_mean_dc / np.maximum(_dc, 1)) ** args.class_weight_strength
          else:
              _alpha_full = compute_class_weights(
                  labels[fold_train_idx][tr_mask, :20],
                  args.class_weight_mode,
                  args.class_weight_strength,
              )                                               # (20,)
              _alpha_arr = _alpha_full[label_cols]            # (n_classes,)
          alpha = torch.tensor(_alpha_arr, dtype=torch.float32, device=device)
          print(f"  Train: {len(train_images2)}, Val: {len(val_images)}")

          fold_out = run_out / f'fold{fold_i + 1}' if args.cv_folds > 1 else run_out
          fold_out.mkdir(parents=True, exist_ok=True)

          # ── Image size ────────────────────────────────────────────────────
          H, W = train_images2.shape[-2], train_images2.shape[-1]
          img_shape = (1, H, W)

          # ── Transforms ───────────────────────────────────────────────────
          _aug = transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.RandomRotation(degrees=180),
          ])
          if args.model == 'vit':
              tf_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(), transforms.RandomRotation(degrees=180)])
              tf_val   = transforms.Compose([transforms.Resize(224)])
              print(f"  Upsampling images to 224×224 for {args.model} (train: +flip/rotate aug)")
          elif args.model == 'enb0':
              tf_train = _aug
              tf_val   = None
              print(f"  Training at native 89×89 for {args.model} (train: +flip/rotate aug)")
          else:
              tf_train = None
              tf_val   = None

          # ── Scattering coefficients ───────────────────────────────────────
          scat_shape = None
          if args.model in ('scatternet', 'simplescatternet', 'dualssn'):
              print("Computing scattering coefficients...")
              all_for_scat = np.concatenate([train_images2, val_images, test_images], axis=0)
              all_scat, scat_shape = compute_scattering(all_for_scat, J=2, L=8,
                                                         device=str(device))
              n_tr2, n_va, n_te = len(train_images2), len(val_images), len(test_images)
              scat_tr  = all_scat[:n_tr2]
              scat_va  = all_scat[n_tr2:n_tr2 + n_va]
              scat_te  = all_scat[n_tr2 + n_va:]
              print(f"  Scattering done: train={scat_tr.shape}, val={scat_va.shape}, test={scat_te.shape}")

          # ── Build model ───────────────────────────────────────────────────
          print(f"Building model: {args.model}")
          model = build_model(args.model, n_classes, img_shape, scat_shape)
          model = model.to(device)
          if args.compile:
              model = torch.compile(model)
              print("  torch.compile() applied")
          n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
          print(f"  Parameters: {n_params:,}")

          # ── DataLoaders ───────────────────────────────────────────────────
          if args.model in ('scatternet', 'simplescatternet'):
              # Only scattering input (no raw image)
              scat_tr_ds  = ScatterDataset(train_images2, scat_tr, train_labels2)
              scat_va_ds  = ScatterDataset(val_images,    scat_va, val_labels)
              scat_te_ds  = ScatterDataset(test_images,   scat_te, test_labels)

              train_dl = make_loader(scat_tr_ds, args.batch_size, shuffle=True)
              val_dl   = make_loader(scat_va_ds, args.batch_size, shuffle=False)
              test_dl  = make_loader(scat_te_ds, args.batch_size, shuffle=False)
              use_scat_only = True
          elif args.model == 'dualssn':
              train_dl = make_loader(ScatterDataset(train_images2, scat_tr, train_labels2),
                                     args.batch_size, shuffle=True)
              val_dl   = make_loader(ScatterDataset(val_images, scat_va, val_labels),
                                     args.batch_size, shuffle=False)
              test_dl  = make_loader(ScatterDataset(test_images, scat_te, test_labels),
                                     args.batch_size, shuffle=False)
              use_scat_only = False
          else:
              train_dl = make_loader(RadioImageDataset(train_images2, train_labels2, tf_train),
                                     args.batch_size, shuffle=True)
              val_dl   = make_loader(RadioImageDataset(val_images, val_labels, tf_val),
                                     args.batch_size, shuffle=False)
              test_dl  = make_loader(RadioImageDataset(test_images, test_labels, tf_val),
                                     args.batch_size, shuffle=False)
              use_scat_only = False

          # ── Optimiser ─────────────────────────────────────────────────────
          optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
              optimiser, mode='min', factor=0.5, patience=5
          )

          # ── Training loop ─────────────────────────────────────────────────
          train_losses, val_losses = [], []
          best_val_loss  = float('inf')
          best_state     = None
          epochs_no_impr = 0

          for epoch in range(1, args.epochs + 1):
              model.train()
              epoch_loss = 0.0

              for batch in train_dl:
                  if args.model == 'dualssn':
                      imgs, scats, labels_b = batch
                      imgs, scats, labels_b = imgs.to(device), scats.to(device), labels_b.to(device)
                      logits = model(imgs, scats)
                  elif use_scat_only:
                      _, scats, labels_b = batch
                      scats, labels_b = scats.to(device), labels_b.to(device)
                      logits = model(scats)
                  else:
                      imgs, labels_b = batch
                      imgs, labels_b = imgs.to(device), labels_b.to(device)
                      logits = model(imgs)

                  loss = weighted_class_mean_loss(logits, labels_b, alpha, n_classes)
                  optimiser.zero_grad()
                  loss.backward()
                  optimiser.step()
                  epoch_loss += loss.item() * len(labels_b)

              train_losses.append(epoch_loss / len(train_dl.dataset))

              # Validation
              model.eval()
              val_loss_total = 0.0
              with torch.no_grad():
                  for batch in val_dl:
                      if args.model == 'dualssn':
                          imgs, scats, labels_b = batch
                          imgs, scats, labels_b = imgs.to(device), scats.to(device), labels_b.to(device)
                          logits = model(imgs, scats)
                      elif use_scat_only:
                          _, scats, labels_b = batch
                          scats, labels_b = scats.to(device), labels_b.to(device)
                          logits = model(scats)
                      else:
                          imgs, labels_b = batch
                          imgs, labels_b = imgs.to(device), labels_b.to(device)
                          logits = model(imgs)
                      val_loss_total += (sum(
                          F.cross_entropy(logits[:, c, :], labels_b[:, c])
                          for c in range(n_classes)
                      ) / n_classes).item() * len(labels_b)

              val_loss = val_loss_total / len(val_dl.dataset)
              val_losses.append(val_loss)
              scheduler.step(val_loss)

              if val_loss < best_val_loss:
                  best_val_loss  = val_loss
                  best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                  epochs_no_impr = 0
              else:
                  epochs_no_impr += 1
                  if epochs_no_impr >= args.patience:
                      print(f"Early stopping at epoch {epoch}")
                      break

              if epoch % 10 == 0 or epoch == 1:
                  print(f"Epoch {epoch:3d}  train={train_losses[-1]:.4f}  "
                        f"val={val_losses[-1]:.4f}  "
                        f"lr={optimiser.param_groups[0]['lr']:.2e}")

          model.load_state_dict(best_state)
          print(f"Restored best model (val loss = {best_val_loss:.4f})")

          # ── Training curve ────────────────────────────────────────────────
          fig, ax = plt.subplots(figsize=(6, 3))
          ax.plot(train_losses, label='Train')
          ax.plot(val_losses,   label='Validation')
          best_ep = int(np.argmin(val_losses))
          ax.axvline(best_ep, color='grey', linestyle='--', linewidth=1,
                     label=f'Best ({best_ep + 1})')
          ax.set_xlabel('Epoch')
          ax.set_ylabel('Loss')
          ax.set_title(f'{args.model} training curve — {args.label_set}')
          ax.legend()
          plt.tight_layout()
          plt.savefig(fold_out / 'training_curve.png', dpi=100)
          plt.close()

          # ── Test inference ────────────────────────────────────────────────
          model.eval()
          all_probs, all_preds, all_labels = [], [], []

          with torch.no_grad():
              for batch in test_dl:
                  if args.model == 'dualssn':
                      imgs, scats, labels_b = batch
                      imgs, scats = imgs.to(device), scats.to(device)
                      logits = model(imgs, scats)
                  elif use_scat_only:
                      _, scats, labels_b = batch
                      scats = scats.to(device)
                      logits = model(scats)
                  else:
                      imgs, labels_b = batch
                      imgs = imgs.to(device)
                      logits = model(imgs)

                  probs = F.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
                  preds = (probs >= 0.5).astype(int)
                  all_probs.append(probs)
                  all_preds.append(preds)
                  all_labels.append(labels_b.numpy())

          test_probs      = np.concatenate(all_probs,  axis=0)
          test_preds      = np.concatenate(all_preds,  axis=0)
          test_labels_arr = np.concatenate(all_labels, axis=0)

          # ── Metrics ───────────────────────────────────────────────────────
          results = evaluate_metrics(test_labels_arr, test_preds, test_probs, class_names)
          print(f"\nMacro F1:     {results['f1_macro']:.4f}")
          print(f"Macro AUC:    {results['auc_macro']:.4f}")
          print(f"Macro Recall: {results['recall_macro']:.4f}")
          print(f"Accuracy:     {results['accuracy']:.4f}")

          # ── Save outputs ──────────────────────────────────────────────────
          np.save(fold_out / 'test_probs.npy',      test_probs)
          np.save(fold_out / 'test_preds.npy',      test_preds)
          np.save(fold_out / 'test_labels.npy',     test_labels_arr)
          np.save(fold_out / 'test_source_idx.npy', test_idx)

          torch.save({
              'state_dict':  best_state,
              'model':       args.model,
              'label_set':   args.label_set,
              'class_names': class_names,
              'n_classes':   n_classes,
              'scat_shape':  scat_shape,
              'img_shape':   img_shape,
              'seed':        args.seed,
          }, fold_out / 'model_best.pt')
          torch.save(model, fold_out / 'model_best_full.pt')

          results['label_set']          = args.label_set
          results['eval_fri_frii_pure'] = args.eval_fri_frii_pure
          with open(fold_out / 'results.json', 'w') as f:
              json.dump(results, f, indent=2)

          print(f"\nAll outputs saved to: {fold_out}")
          fold_results_list.append(results)

        if args.cv_folds > 1:
            f1s  = [r['f1_macro']     for r in fold_results_list]
            aucs = [r['auc_macro']    for r in fold_results_list]
            accs = [r['accuracy']     for r in fold_results_list]
            recs = [r['recall_macro'] for r in fold_results_list]
            agg = {
                'cv_folds':          args.cv_folds,
                'f1_macro_mean':     float(np.mean(f1s)),
                'f1_macro_std':      float(np.std(f1s)),
                'auc_macro_mean':    float(np.mean(aucs)),
                'auc_macro_std':     float(np.std(aucs)),
                'accuracy_mean':     float(np.mean(accs)),
                'accuracy_std':      float(np.std(accs)),
                'recall_macro_mean': float(np.mean(recs)),
                'recall_macro_std':  float(np.std(recs)),
                'per_fold':          fold_results_list,
            }
            print(f"\nK-fold summary ({args.cv_folds} folds):")
            print(f"  Macro F1:     {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
            print(f"  Macro AUC:    {agg['auc_macro_mean']:.4f} ± {agg['auc_macro_std']:.4f}")
            print(f"  Macro Recall: {agg['recall_macro_mean']:.4f} ± {agg['recall_macro_std']:.4f}")
            print(f"  Accuracy:     {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
            with open(run_out / 'cv_results.json', 'w') as f:
                json.dump(agg, f, indent=2)

        all_run_results.append(fold_results_list)

    # ── Multi-run summary ─────────────────────────────────────────────────────
    if args.gen_dir is None and args.n_runs > 1 and args.cv_folds == 1:
        run_results = [rlist[0] for rlist in all_run_results]

        print(f"\n{'='*60}")
        print(f"Multi-run summary  ({args.n_runs} runs, {args.label_set})")
        print(f"{'='*60}")

        metrics_order = [
            ('f1_macro',     'Macro F1    '),
            ('auc_macro',    'Macro AUC   '),
            ('recall_macro', 'Macro Recall'),
            ('accuracy',     'Accuracy    '),
        ]
        summary = {}
        for key, label in metrics_order:
            vals = [r[key] for r in run_results]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[key] = {'mean': mean, 'std': std,
                            'min': float(np.min(vals)), 'max': float(np.max(vals)),
                            'per_run': [float(v) for v in vals]}
            print(f"  {label}: {mean:.4f} ± {std:.4f}"
                  f"  (min={np.min(vals):.4f}, max={np.max(vals):.4f})")

        # ── Ensemble scores ───────────────────────────────────────────────────
        all_probs_runs = []
        for run_i in range(1, args.n_runs + 1):
            p = np.load(out_dir / f'run{run_i}' / 'test_probs.npy')
            all_probs_runs.append(p)
        test_labels_arr = np.load(out_dir / 'run1' / 'test_labels.npy')

        f1_vals  = [r['f1_macro'] for r in run_results]
        ranked   = list(np.argsort(f1_vals)[::-1])   # indices, best first

        print(f"\n  Ensemble scores (runs ranked by F1, threshold 0.5):")
        ensemble_summary = {}
        seen_k = set()
        for k, tag in [(1, 'top-1'), (5, 'top-5'), (args.n_runs, 'top-all')]:
            k = min(k, args.n_runs)
            if k in seen_k:
                continue
            seen_k.add(k)
            ens_probs = np.mean([all_probs_runs[i] for i in ranked[:k]], axis=0)
            ens_preds = (ens_probs >= 0.5).astype(int)
            ens = evaluate_metrics(test_labels_arr, ens_preds, ens_probs, class_names)
            lbl = tag if k == args.n_runs or tag == 'top-all' else tag
            if k == args.n_runs and tag != 'top-all':
                lbl = f'top-{k}'
            print(f"  {lbl:8s}  F1={ens['f1_macro']:.4f}  AUC={ens['auc_macro']:.4f}"
                  f"  Recall={ens['recall_macro']:.4f}  Acc={ens['accuracy']:.4f}")
            ensemble_summary[lbl] = {
                'k': k, 'f1_macro': ens['f1_macro'], 'auc_macro': ens['auc_macro'],
                'recall_macro': ens['recall_macro'], 'accuracy': ens['accuracy'],
            }

        summary_out = {
            'n_runs':    args.n_runs,
            'label_set': args.label_set,
            'per_metric': summary,
            'ensemble':   ensemble_summary,
        }
        with open(out_dir / 'run_summary.json', 'w') as f:
            json.dump(summary_out, f, indent=2)
        print(f"\n  Summary saved to: {out_dir / 'run_summary.json'}")

    # ── Label-fraction sweep ───────────────────────────────────────────────────
    # Runs n_runs independent sweeps (one per training run, matching seeds).
    # 100% fraction is read from the already-completed results.json — no
    # retraining.  Sub-100% fractions are retrained from scratch with the
    # run's seed so that std is meaningful across all fractions.
    # One label_fraction_metrics.json is saved per run_out directory so that
    # the notebook's rglob() picks them all up and computes mean ± std.
    # Skipped for scatter-based models (dualssn, scatternet, simplescatternet).
    _skip_sweep = args.skip_sweep or args.model in ('dualssn', 'scatternet', 'simplescatternet')

    # Shared preamble for sweep and gen-aug
    _H, _W     = images.shape[-2], images.shape[-1]
    _aug_sw = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
    ])
    if args.model == 'vit':
        _tf_sw       = transforms.Compose([transforms.Resize(224)])
        _tf_train_sw = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(), transforms.RandomRotation(degrees=180)])
    elif args.model == 'enb0':
        _tf_sw       = None
        _tf_train_sw = _aug_sw
    else:
        _tf_sw       = None
        _tf_train_sw = None
    print(f"  Transforms: val={'Resize(224)' if args.model == 'vit' else 'none'}  "
          f"train={'+flip/rotate' if args.model in ('vit', 'enb0') else 'none'}")
    _alpha_sw  = torch.ones(n_classes, dtype=torch.float32, device=device)

    # Val images / labels (rebuilt from val_idx for correctness)
    _va_images_sw = images[val_idx]
    _va_labels_sw = (_make_derived(labels[val_idx]) if args.label_set == "derived"
                     else labels[val_idx][:, label_cols])
    if args.label_set in ("classical_pure", "initial_pure"):
        _va_pure      = labels[val_idx][:, 0:5].sum(axis=1) == 1
        _va_images_sw = _va_images_sw[_va_pure]
        _va_labels_sw = _va_labels_sw[_va_pure]

    _te_dl_sw = make_loader(
        RadioImageDataset(test_images, test_labels, _tf_sw),
        args.batch_size, shuffle=False)

    if args.gen_dir is not None:
        # ── Generative augmentation experiment ────────────────────────────────
        GEN_FRACS = [0.5, 1.0, 2.0]

        import zuko
        from suplat.models.generative_models import FlowMatchingUNet

        _dec_ckpt = torch.load(args.gen_dir / "decoder.pt", map_location="cpu", weights_only=False)
        _decoder  = FlowMatchingUNet(z_dim=_dec_ckpt["feat_dim"], base_ch=_dec_ckpt["base_ch"])
        _decoder.load_state_dict(_dec_ckpt["model_state_dict"])
        _decoder.to(device).eval()

        _nsf_ckpt = torch.load(args.gen_dir / "nsf.pt", map_location="cpu", weights_only=False)
        _nsf      = zuko.flows.NSF(
            features=_nsf_ckpt["feat_dim"],
            context=_nsf_ckpt["n_labels"],
            transforms=_nsf_ckpt["n_transforms"],
        )
        _nsf.load_state_dict(_nsf_ckpt["model_state_dict"])
        _nsf.to(device).eval()

        _n_initial_labels = _nsf_ckpt["n_labels"]  # 5
        _gen_out_dir = out_dir / "gen_aug"
        _gen_out_dir.mkdir(exist_ok=True)
        _gen_summary = {}

        print(f"\n{'='*60}")
        print(f"Gen-augmentation experiment  ({args.label_set}, {args.model}, {args.n_runs} run(s))")
        print(f"Gen dir: {args.gen_dir}")
        print(f"{'='*60}")

        for _gfrac in GEN_FRACS:
            _frac_dir = _gen_out_dir / f"frac_{_gfrac:.2f}"
            _frac_dir.mkdir(exist_ok=True)
            _frac_run_results = []

            for _ri in range(1, args.n_runs + 1):
                _run_seed = args.seed + (_ri - 1)
                _run_dir  = _frac_dir / f"run{_ri}"
                _run_dir.mkdir(exist_ok=True)
                _res_path = _run_dir / "results.json"

                if _res_path.exists() and not args.force:
                    print(f"  frac={_gfrac} run={_ri}: cached — skipping.")
                    with open(_res_path) as _f:
                        _frac_run_results.append(json.load(_f))
                    continue

                torch.manual_seed(_run_seed)
                np.random.seed(_run_seed)

                # ── Real training images (full set, with pure filter if applicable) ──
                _tr_im = images[train_idx]
                _tr_lb = (_make_derived(labels[train_idx]) if args.label_set == "derived"
                          else labels[train_idx][:, label_cols])
                _tr_initial = labels[train_idx][:, :_n_initial_labels].astype(np.float32)

                if args.label_set in ("classical_pure", "initial_pure"):
                    _pm = labels[train_idx][:, 0:5].sum(axis=1) == 1
                    _tr_im, _tr_lb, _tr_initial = _tr_im[_pm], _tr_lb[_pm], _tr_initial[_pm]
                elif args.label_set == "pure":
                    _pm = labels[train_idx].sum(axis=1) == 1
                    _tr_im, _tr_lb, _tr_initial = _tr_im[_pm], _tr_lb[_pm], _tr_initial[_pm]

                n_real  = len(_tr_im)
                n_gen   = int(_gfrac * n_real)
                print(f"\n  frac={_gfrac}  run={_ri}  real={n_real}  gen={n_gen}", flush=True)

                # ── Generate images ────────────────────────────────────────────────
                # Sample conditions by cycling through training labels
                _cond_idx = np.resize(np.arange(n_real), n_gen)  # tile if n_gen > n_real
                _cond     = torch.tensor(_tr_initial[_cond_idx], dtype=torch.float32)
                _gen_imgs_list, _gen_lbs_list = [], []

                _GEN_BS = 64
                with torch.no_grad():
                    for _b0 in range(0, n_gen, _GEN_BS):
                        _b1   = min(_b0 + _GEN_BS, n_gen)
                        _cy   = _cond[_b0:_b1].to(device)
                        _z    = _nsf(_cy).sample()          # (B, feat_dim)
                        _xgen = _decoder.sample(_z)          # (B, 1, 89, 89) float [0,1]
                        # Convert to uint8 to match images_filtered.npy dtype
                        _xgen_u8 = (_xgen.squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        _gen_imgs_list.append(_xgen_u8)
                        _gen_lbs_list.append(
                            (_make_derived(labels[train_idx[_cond_idx[_b0:_b1]]])
                             if args.label_set == "derived"
                             else labels[train_idx[_cond_idx[_b0:_b1]]][:, label_cols])
                        )

                _gen_imgs = np.concatenate(_gen_imgs_list, axis=0)   # (n_gen, 89, 89) uint8
                _gen_lbs  = np.concatenate(_gen_lbs_list, axis=0)    # (n_gen, n_classes)

                # ── Combine real + generated ───────────────────────────────────────
                _aug_imgs = np.concatenate([_tr_im, _gen_imgs], axis=0)
                _aug_lbs  = np.concatenate([_tr_lb, _gen_lbs],  axis=0)
                print(f"    Augmented training set: {len(_aug_imgs)} total ({n_real} real + {n_gen} gen)", flush=True)

                # ── Train classifier ───────────────────────────────────────────────
                torch.manual_seed(_run_seed); np.random.seed(_run_seed)
                _mdl_g = build_model(args.model, n_classes, (1, _H, _W)).to(device)
                if args.compile:
                    _mdl_g = torch.compile(_mdl_g)
                _opt_g = torch.optim.Adam(_mdl_g.parameters(), lr=args.lr, weight_decay=1e-4)
                _sch_g = torch.optim.lr_scheduler.ReduceLROnPlateau(_opt_g, mode='min', factor=0.5, patience=5)
                _alpha_g = torch.ones(n_classes, dtype=torch.float32, device=device)

                _tr_dl_g = make_loader(RadioImageDataset(_aug_imgs, _aug_lbs, _tf_train_sw), args.batch_size, shuffle=True)
                _va_dl_g = make_loader(RadioImageDataset(_va_images_sw, _va_labels_sw, _tf_sw), args.batch_size, shuffle=False)

                _bv_g, _bs_g, _ni_g = float('inf'), None, 0
                for _ep_g in range(1, args.epochs + 1):
                    _mdl_g.train()
                    for _im_g, _lb_g in _tr_dl_g:
                        _im_g, _lb_g = _im_g.to(device), _lb_g.to(device)
                        _loss_g = weighted_class_mean_loss(_mdl_g(_im_g), _lb_g, _alpha_g, n_classes)
                        _opt_g.zero_grad(); _loss_g.backward(); _opt_g.step()
                    _mdl_g.eval()
                    _vl_g = 0.0
                    with torch.no_grad():
                        for _im_g, _lb_g in _va_dl_g:
                            _im_g, _lb_g = _im_g.to(device), _lb_g.to(device)
                            _vl_g += (sum(F.cross_entropy(_mdl_g(_im_g)[:, c, :], _lb_g[:, c])
                                          for c in range(n_classes)) / n_classes).item() * len(_lb_g)
                    _vl_g /= len(_va_dl_g.dataset)
                    _sch_g.step(_vl_g)
                    if _vl_g < _bv_g:
                        _bv_g, _bs_g, _ni_g = _vl_g, {k: v.cpu().clone() for k, v in _mdl_g.state_dict().items()}, 0
                    else:
                        _ni_g += 1
                        if _ni_g >= args.patience:
                            print(f"      Early stop ep {_ep_g}", flush=True); break
                _mdl_g.load_state_dict(_bs_g)

                # ── Evaluate on test set ───────────────────────────────────────────
                _mdl_g.eval()
                _pb_g, _pd_g, _lb_g_all = [], [], []
                with torch.no_grad():
                    for _im_g, _lb_g in _te_dl_sw:
                        _im_g = _im_g.to(device)
                        _pb   = F.softmax(_mdl_g(_im_g), dim=-1)[:, :, 1].cpu().numpy()
                        _pb_g.append(_pb); _pd_g.append((_pb >= 0.5).astype(int)); _lb_g_all.append(_lb_g.numpy())
                _ms_g = evaluate_metrics(np.concatenate(_lb_g_all), np.concatenate(_pd_g), np.concatenate(_pb_g), class_names)
                print(f"      F1={_ms_g['f1_macro']:.4f}  AUC={_ms_g['auc_macro']:.4f}  Rec={_ms_g['recall_macro']:.4f}  Acc={_ms_g['accuracy']:.4f}", flush=True)

                _res_g = {"gen_frac": _gfrac, "n_real": n_real, "n_gen": n_gen,
                          "f1_macro": _ms_g["f1_macro"], "auc_macro": _ms_g["auc_macro"],
                          "accuracy": _ms_g["accuracy"], "recall_macro": _ms_g["recall_macro"]}
                with open(_res_path, "w") as _f: json.dump(_res_g, _f, indent=2)
                _frac_run_results.append(_res_g)

            # Aggregate across runs
            _metrics_agg = {m: float(np.mean([r[m] for r in _frac_run_results]))
                            for m in ("f1_macro", "auc_macro", "accuracy", "recall_macro")}
            if args.n_runs > 1:
                _metrics_agg.update({f"{m}_std": float(np.std([r[m] for r in _frac_run_results]))
                                     for m in ("f1_macro", "auc_macro", "accuracy", "recall_macro")})
            _gen_summary[str(_gfrac)] = _metrics_agg
            print(f"\n  frac={_gfrac}: mean F1={_metrics_agg['f1_macro']:.4f}  AUC={_metrics_agg['auc_macro']:.4f}  Rec={_metrics_agg['recall_macro']:.4f}  Acc={_metrics_agg['accuracy']:.4f}", flush=True)

        with open(_gen_out_dir / "gen_aug_summary.json", "w") as _f:
            json.dump(_gen_summary, _f, indent=2)
        print(f"\nGen-aug summary saved → {_gen_out_dir / 'gen_aug_summary.json'}")

    elif not _skip_sweep:
        _FRACS_SUB = [0.01, 0.05, 0.10, 0.25, 0.50]  # 100% loaded from results.json

        print(f"\n{'='*60}")
        print(f"Label-fraction sweep  ({args.label_set}, {args.model}, {args.n_runs} run(s))")
        print(f"{'='*60}")

        for _ri in range(1, args.n_runs + 1):
            _run_seed   = args.seed + (_ri - 1)
            _run_out    = out_dir / f'run{_ri}' if args.n_runs > 1 else out_dir
            _frac_cache = _run_out / "label_fraction_metrics.json"

            if _frac_cache.exists() and not args.force:
                print(f"  Run {_ri}: fraction cache exists — skipping.")
                continue

            print(f"\n  --- Run {_ri} / {args.n_runs}  (seed={_run_seed}) ---", flush=True)
            _frac_out: dict = {}

            # 100% — load from this run's already-completed results (no retraining)
            _res_path = _run_out / "results.json"
            if _res_path.exists():
                with open(_res_path) as _f:
                    _r100 = json.load(_f)
                _frac_out["1.0"] = {"Supervised": {
                    "f1":       _r100["f1_macro"],
                    "auc":      _r100["auc_macro"],
                    "accuracy": _r100["accuracy"],
                    "recall":   _r100["recall_macro"],
                }}
                print(f"    100%: F1={_r100['f1_macro']:.4f}  (from results.json)", flush=True)
            else:
                print(f"    100%: results.json not found — skipping 100% point.", flush=True)

            # Sub-100% fractions: retrain with this run's seed
            _rng_sw = np.random.default_rng(_run_seed)
            _va_dl_sw = make_loader(
                RadioImageDataset(_va_images_sw, _va_labels_sw, _tf_sw),
                args.batch_size, shuffle=False)

            for _frac in _FRACS_SUB:
                print(f"\n    Fraction {_frac:.0%}:", flush=True)
                _n_sw      = max(n_classes * 2, int(_frac * len(train_idx)))
                _fi        = _rng_sw.choice(len(train_idx), size=_n_sw, replace=False)
                _tr_idx_sw = train_idx[_fi]

                _tr_im_sw = images[_tr_idx_sw]
                _tr_lb_sw = (_make_derived(labels[_tr_idx_sw]) if args.label_set == "derived"
                             else labels[_tr_idx_sw][:, label_cols])
                if args.label_set in ("classical_pure", "initial_pure"):
                    _tr_pm    = labels[_tr_idx_sw][:, 0:5].sum(axis=1) == 1
                    _tr_im_sw = _tr_im_sw[_tr_pm]
                    _tr_lb_sw = _tr_lb_sw[_tr_pm]

                if len(_tr_im_sw) == 0:
                    print("      No samples after filtering — skipping.", flush=True)
                    continue
                print(f"      Training on {len(_tr_im_sw)} samples", flush=True)

                torch.manual_seed(_run_seed)
                np.random.seed(_run_seed)
                _mdl_sw = build_model(args.model, n_classes, (1, _H, _W)).to(device)
                if args.compile:
                    _mdl_sw = torch.compile(_mdl_sw)
                _opt_sw = torch.optim.Adam(_mdl_sw.parameters(),
                                           lr=args.lr, weight_decay=1e-4)
                _sch_sw = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    _opt_sw, mode='min', factor=0.5, patience=5)
                _tr_dl_sw = make_loader(
                    RadioImageDataset(_tr_im_sw, _tr_lb_sw, _tf_train_sw),
                    args.batch_size, shuffle=True)

                _bv_sw, _bs_sw, _ni_sw = float('inf'), None, 0
                for _ep_sw in range(1, args.epochs + 1):
                    _mdl_sw.train()
                    for _im_sw, _lb_sw in _tr_dl_sw:
                        _im_sw, _lb_sw = _im_sw.to(device), _lb_sw.to(device)
                        _loss_sw = weighted_class_mean_loss(
                            _mdl_sw(_im_sw), _lb_sw, _alpha_sw, n_classes)
                        _opt_sw.zero_grad()
                        _loss_sw.backward()
                        _opt_sw.step()
                    _mdl_sw.eval()
                    _vl_sw = 0.0
                    with torch.no_grad():
                        for _im_sw, _lb_sw in _va_dl_sw:
                            _im_sw, _lb_sw = _im_sw.to(device), _lb_sw.to(device)
                            _vl_sw += (sum(
                                F.cross_entropy(_mdl_sw(_im_sw)[:, c, :], _lb_sw[:, c])
                                for c in range(n_classes)
                            ) / n_classes).item() * len(_lb_sw)
                    _vl_sw /= len(_va_dl_sw.dataset)
                    _sch_sw.step(_vl_sw)
                    if _vl_sw < _bv_sw:
                        _bv_sw = _vl_sw
                        _bs_sw = {k: v.cpu().clone()
                                  for k, v in _mdl_sw.state_dict().items()}
                        _ni_sw = 0
                    else:
                        _ni_sw += 1
                        if _ni_sw >= args.patience:
                            print(f"      Early stop ep {_ep_sw}", flush=True)
                            break
                _mdl_sw.load_state_dict(_bs_sw)

                _mdl_sw.eval()
                _pb_sw, _pd_sw, _lb_sw_all = [], [], []
                with torch.no_grad():
                    for _im_sw, _lb_sw in _te_dl_sw:
                        _im_sw = _im_sw.to(device)
                        _pb = F.softmax(_mdl_sw(_im_sw), dim=-1)[:, :, 1].cpu().numpy()
                        _pb_sw.append(_pb)
                        _pd_sw.append((_pb >= 0.5).astype(int))
                        _lb_sw_all.append(_lb_sw.numpy())
                _tp_sw = np.concatenate(_pb_sw)
                _yp_sw = np.concatenate(_pd_sw)
                _yt_sw = np.concatenate(_lb_sw_all)
                _ms    = evaluate_metrics(_yt_sw, _yp_sw, _tp_sw, class_names)
                _frac_out[str(_frac)] = {"Supervised": {
                    "f1":       _ms["f1_macro"],
                    "auc":      _ms["auc_macro"],
                    "accuracy": _ms["accuracy"],
                    "recall":   _ms["recall_macro"],
                }}
                print(f"      F1={_ms['f1_macro']:.4f}  AUC={_ms['auc_macro']:.4f}"
                      f"  Acc={_ms['accuracy']:.4f}", flush=True)

            if _frac_out:
                with open(_frac_cache, "w") as _fh:
                    json.dump(_frac_out, _fh, indent=2)
                print(f"    Saved → {_frac_cache}", flush=True)

    else:
        print(f"\nLabel-fraction sweep skipped for model '{args.model}'.")


if __name__ == '__main__':
    main()
