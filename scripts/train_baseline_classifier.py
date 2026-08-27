"""
Train end-to-end baseline classifiers for radio galaxy classification.

Usage:
    python train_baseline_classifier2.py \\
        --run_dir /path/to/byol/run \\
        --data_dir /path/to/data \\
        --model cnn \\
        --label_set classical

Models: cnn | scatternet | simplescatternet | vit | dualssn | enb0

Refactored version of train_baseline_classifier.py.  All three execution
paths (full-label training, generative-aug, label-fraction sweep) share a
single ``_train_one_run`` function to avoid silent divergence.
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

import argparse
import json
import sys
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
from suplat.data.augmentations import get_augmentation
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
def weighted_bce_loss(logits, targets, alpha, n_classes):
    """Weighted mean of per-class BCE-with-logits. Matches train_finetuning.py.
    logits  : (B, n_classes)  raw floats
    targets : (B, n_classes)  int64 multi-hot labels
    alpha   : (n_classes,)    float32 tensor on device
    """
    unreduced = F.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction='none')     # (B, n_classes)
    alpha_sum = alpha.sum().clamp(min=1e-6)
    return ((unreduced * alpha).sum(1) / alpha_sum).mean()


def evaluate_metrics(y_true, y_pred, y_prob, class_names, is_binary=False):
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
    acc     = float((y_true == y_pred).mean()) if is_binary else accuracy_score(y_true, y_pred)
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


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(model_name, n_classes, img_shape, scat_shape=None):
    """Returns a model whose output is (B, n_classes) raw logits for BCE."""
    n_out = n_classes

    if model_name == 'vit':
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

    return base


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


# ── Core training function ────────────────────────────────────────────────────
def _train_one_run(
    train_imgs, train_lbs,       # training images + labels (already filtered/augmented)
    test_imgs, test_lbs,         # test images + labels (subset-filtered)
    te_labels_20,                # full 20-col test labels
    *,
    model_name, n_classes, class_names, device, args,
    alpha,                       # class-weight tensor (n_classes,), on CPU
    val_imgs=None, val_lbs=None, # optional early-stopping validation set
    scat_tr=None, scat_va=None, scat_te=None,  # precomputed scattering (or None)
    extra_results=None,          # dict merged into results (e.g. gen_frac, n_gen)
    seed=42,
    run_dir=None,                # if provided: save outputs here; if None: skip saving
    test_source_idx=None,        # saved as test_source_idx.npy when run_dir set
):
    """Train one classifier run and optionally save outputs.

    Returns:
        (results_dict, test_probs, test_preds, test_labels_arr,
         (train_losses, test_losses, val_losses))
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Derive shapes
    H, W       = train_imgs.shape[-2], train_imgs.shape[-1]
    img_shape  = (1, H, W)
    scat_shape = tuple(scat_tr.shape[1:]) if scat_tr is not None else None

    # Transforms (derived from model_name, same logic as original per-fold setup)
    if model_name == 'vit':
        tf_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
        ])
        tf_val = transforms.Compose([transforms.Resize(224)])
        print(f"  Upsampling images to 224×224 for {model_name} (train: +flip/rotate aug)")
    elif model_name == 'enb0':
        tf_train = get_augmentation('quart_ext')
        tf_val   = None
        if args.no_augmentation:
            tf_train = None
        print(f"  Training at native 89×89 for {model_name} (train: {'no aug' if args.no_augmentation else 'quart_ext aug'})")
    else:
        tf_train = None
        tf_val   = None

    # Move alpha to device
    alpha = alpha.to(device)

    # Build model
    print(f"Building model: {model_name}")
    model = build_model(model_name, n_classes, img_shape, scat_shape)
    model = model.to(device)
    if args.compile:
        model = torch.compile(model)
        print("  torch.compile() applied")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # DataLoaders
    _nw = args.num_workers
    def _mk(ds, bs, shuf):
        return DataLoader(ds, batch_size=bs, shuffle=shuf,
                          num_workers=_nw, pin_memory=True)

    use_scat_only = model_name in ('scatternet', 'simplescatternet')
    if model_name in ('scatternet', 'simplescatternet', 'dualssn'):
        train_dl = _mk(ScatterDataset(train_imgs, scat_tr, train_lbs),  args.batch_size, True)
        test_dl  = _mk(ScatterDataset(test_imgs,  scat_te, test_lbs),   args.batch_size, False)
        val_dl   = (_mk(ScatterDataset(val_imgs, scat_va, val_lbs),      args.batch_size, False)
                    if val_imgs is not None else None)
    else:
        train_dl = _mk(RadioImageDataset(train_imgs, train_lbs, tf_train), args.batch_size, True)
        test_dl  = _mk(RadioImageDataset(test_imgs,  test_lbs,  tf_val),   args.batch_size, False)
        val_dl   = (_mk(RadioImageDataset(val_imgs,  val_lbs,   tf_val),   args.batch_size, False)
                    if val_imgs is not None else None)

    print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs) if val_imgs is not None else 0}")

    # Optimiser + scheduler
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=5)

    # Forward helper (training + validation)
    def _forward(mdl, batch):
        if model_name == 'dualssn':
            imgs, scats, labels_b = batch
            imgs, scats, labels_b = imgs.to(device), scats.to(device), labels_b.to(device)
            return mdl(imgs, scats), labels_b
        elif use_scat_only:
            _, scats, labels_b = batch
            scats, labels_b = scats.to(device), labels_b.to(device)
            return mdl(scats), labels_b
        else:
            imgs, labels_b = batch
            imgs, labels_b = imgs.to(device), labels_b.to(device)
            return mdl(imgs), labels_b

    # Training loop
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss  = float('inf')
    best_state     = None
    epochs_no_impr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_dl:
            logits, labels_b = _forward(model, batch)
            loss = weighted_bce_loss(logits, labels_b, alpha, n_classes)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(labels_b)
        train_losses.append(epoch_loss / len(train_dl.dataset))

        # Test loss (always, for curve)
        model.eval()
        test_loss_total = 0.0
        with torch.no_grad():
            for batch in test_dl:
                logits, labels_b = _forward(model, batch)
                test_loss_total += F.binary_cross_entropy_with_logits(
                    logits, labels_b.float()).item() * len(labels_b)
        test_losses.append(test_loss_total / len(test_dl.dataset))

        # Validation loss + early stopping (only when val_dl is provided)
        if val_dl is not None:
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_dl:
                    logits, labels_b = _forward(model, batch)
                    val_loss_total += F.binary_cross_entropy_with_logits(
                        logits, labels_b.float()).item() * len(labels_b)
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
        else:
            scheduler.step(train_losses[-1])

        if epoch % 10 == 0 or epoch == 1:
            _msg = (f"Epoch {epoch:3d}  train={train_losses[-1]:.4f}  "
                    f"test={test_losses[-1]:.4f}")
            if val_dl is not None:
                _msg += f"  val={val_losses[-1]:.4f}"
            _msg += f"  lr={optimiser.param_groups[0]['lr']:.2e}"
            print(_msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss = {best_val_loss:.4f})")

    # Test inference
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_dl:
            if model_name == 'dualssn':
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
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels_b.numpy())

    test_probs      = np.concatenate(all_probs,  axis=0)
    test_preds      = np.concatenate(all_preds,  axis=0)
    test_labels_arr = np.concatenate(all_labels, axis=0)

    # Metrics
    results = evaluate_metrics(test_labels_arr, test_preds, test_probs, class_names,
                               is_binary=args.label_set.endswith('_binary'))
    results['label_set']          = args.label_set
    results['eval_fri_frii_pure'] = args.eval_fri_frii_pure
    if extra_results:
        results.update(extra_results)

    print(f"\nMacro F1:     {results['f1_macro']:.4f}")
    print(f"Macro AUC:    {results['auc_macro']:.4f}")
    print(f"Macro Recall: {results['recall_macro']:.4f}")
    print(f"Accuracy:     {results['accuracy']:.4f}")

    # Save outputs
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        np.save(run_dir / 'test_probs.npy',     test_probs)
        np.save(run_dir / 'test_preds.npy',     test_preds)
        np.save(run_dir / 'test_labels.npy',    test_labels_arr)
        np.save(run_dir / 'test_labels_20.npy', te_labels_20)
        if test_source_idx is not None:
            np.save(run_dir / 'test_source_idx.npy', test_source_idx)

        torch.save({
            'state_dict':  best_state,
            'model':       model_name,
            'label_set':   args.label_set,
            'class_names': class_names,
            'n_classes':   n_classes,
            'scat_shape':  scat_shape,
            'img_shape':   img_shape,
            'seed':        seed,
        }, run_dir / 'model_best.pt')
        torch.save(model, run_dir / 'model_best_full.pt')

        with open(run_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Per-run training curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_losses, 'b-',  label='train')
        ax.plot(test_losses,  'r--', label='test')
        if val_losses:
            ax.plot(val_losses, 'g:', label='validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} — {args.label_set}')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(run_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\nAll outputs saved to: {run_dir}")

    return results, test_probs, test_preds, test_labels_arr, (train_losses, test_losses, val_losses)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir',    required=True, type=Path)
    parser.add_argument('--data_dir',   required=True, type=Path)
    parser.add_argument('--model',      default='enb0',
                        choices=['cnn', 'scatternet', 'simplescatternet',
                                 'vit', 'dualssn', 'enb0'])
    parser.add_argument('--label_set',  default='initial_pure',
                        help="Label set. Append '_binary' for element-wise accuracy (e.g. initial_binary).")
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
    parser.add_argument('--class_weight_mode', type=str, default=None,
                        choices=["score", "initial", "initial_pure", "morphology", "morphology_pure",
                                 "environment", "environment_pure", "classical", "classical_pure", "all", "all_pure"],
                        help="Upweight rare samples: 'score' (interest tier 1-4) or label-set name "
                             "(inverse frequency). Default: None (uniform).")
    parser.add_argument('--class_weight_strength', type=float, default=0.0,
                        help="Magnitude of class upweighting (0=uniform, default). "
                             "Default: 0.0.")
    parser.add_argument('--byol_run_dir', type=Path, default=None,
                        help="BYOL run directory whose data/train_idx.npy + data/test_idx.npy "
                             "define the train/test split (default: --run_dir itself).")
    parser.add_argument('--data_seed', type=int, default=42,
                        help="Data seed used to locate data_splits/<seed>/. Default: 42.")
    parser.add_argument('--gen_dir', type=Path, default=None,
                        help="Path to generative model directory containing decoder_<variant>.pt "
                             "and nsf_<variant>.pt. When provided, skips the label-fraction sweep "
                             "and instead trains the classifier augmented with generated images at "
                             "fractions 0.5, 1.0, and 2.0 of the real training set size, repeated "
                             "n_runs times each.")
    parser.add_argument('--gen_variant', type=str, default='all',
                        choices=['all', 'full', 'initial', 'initial_pure'],
                        help="Which generator variant to load: 'all'/'full' (conditioned on all 20 labels), "
                             "'initial' (conditioned on initial 5 labels), or 'initial_pure' "
                             "(conditioned on initial_pure 5-class pure labels). "
                             "Selects decoder_<variant>.pt and nsf_<variant>.pt. Default: all.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader worker processes for data loading (default: 4).")
    parser.add_argument('--compile', action='store_true',
                        help="Apply torch.compile() to the model for faster GPU execution.")
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Use 10%% of test set as validation for early stopping.')
    parser.add_argument('--gen_only', action='store_true', default=False,
                        help="Train exclusively on generated images (no real training data), "
                             "with n_gen = n_real (frac=1.0), then evaluate on the real test set. "
                             "Requires --gen_dir. Outputs go to gen_only/ instead of with_generative/.")
    parser.add_argument('--no_augmentation', action='store_true', default=False,
                        help="Disable training augmentation (tf_train=None). Default: False.")
    parser.add_argument('--gen_frac', type=float, default=None,
                        help="Pin gen-aug to a single fraction of real data instead of sweeping "
                             "[0.5, 1.0, 2.0]. Requires --gen_dir. E.g. --gen_frac 1.0.")
    args = parser.parse_args()
    _ls_base = args.label_set[:-7] if args.label_set.endswith('_binary') else args.label_set
    if _ls_base not in LABEL_SETS:
        parser.error(f"Unknown label set: {args.label_set!r}")
    if args.gen_variant == 'full':
        args.gen_variant = 'all'
    _byol_run_dir_explicit = args.byol_run_dir is not None
    if args.byol_run_dir is None:
        args.byol_run_dir = args.run_dir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────
    _cw_base = f'cw{args.class_weight_mode}' if args.class_weight_mode else 'cwNone'
    _cw_tag  = _cw_base if args.class_weight_strength == 1.0 else f'{_cw_base}{args.class_weight_strength}'
    _es_tag = '_ES' if args.early_stopping else ''
    if args.run_name:
        _dir_name = args.run_name
    else:
        _dir_name = f'{args.run_dir.name}_{args.model}_{_cw_tag}{_es_tag}'
    if args.gen_dir and args.gen_only:
        _gen_subdir = 'gen_only'
    elif args.gen_dir:
        _gen_subdir = 'with_generative'
    else:
        _gen_subdir = 'without_generative'
    out_dir = args.run_dir / _gen_subdir / _dir_name
    if out_dir.exists() and not args.force:
        print(f"Resuming existing run: {out_dir.name}  (use --force to retrain from scratch)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {out_dir}")

    # ── Config log ────────────────────────────────────────────────────────
    _cfg_path = out_dir / 'config.log'
    if not _cfg_path.exists():
        with open(_cfg_path, 'w') as _cf:
            for _k, _v in sorted(vars(args).items()):
                _cf.write(f'{_k}: {_v}\n')

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

    _ls_base    = args.label_set[:-7] if args.label_set.endswith('_binary') else args.label_set
    label_cols  = LABEL_SETS[_ls_base]
    if args.label_set == "derived":
        class_names = DERIVED_CLASS_NAMES
    else:
        class_names = [ALL_CLASS_NAMES[i] for i in label_cols]
    n_classes   = len(class_names)
    print(f"Label set: {args.label_set} ({n_classes} classes: {class_names})")

    # ── Train/test split ──────────────────────────────────────────────────
    if args.data_seed is not None and not _byol_run_dir_explicit:
        _seed_str = str(args.data_seed)
        _p = args.run_dir
        _splits_dir = None
        for _ in range(6):
            _p = _p.parent
            if (_p / "data_splits" / _seed_str).exists():
                _splits_dir = _p / "data_splits" / _seed_str
                break
        if _splits_dir is None:
            raise FileNotFoundError(
                f"Could not find data_splits/{_seed_str}/ in any ancestor of {args.run_dir}")
        split_label = f"data_seed={args.data_seed}"
    elif args.data_seed is not None:
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
    train_idx = trainval_idx  # always train on full training set
    print(f"Using split from: {split_label}")
    print(f"  train+val: {len(trainval_idx)}  test: {len(test_idx)}")

    if args.early_stopping:
        val_test_idx, actual_test_idx = train_test_split(
            test_idx, test_size=0.9, random_state=args.seed
        )
    else:
        val_test_idx    = None
        actual_test_idx = test_idx

    test_images = images[actual_test_idx]
    if args.label_set == "derived":
        test_labels = _make_derived(labels[actual_test_idx])
    else:
        test_labels = labels[actual_test_idx][:, label_cols]

    # ── Pure-source filtering (test set) ─────────────────────────────────
    test_pure_mask = None
    if args.label_set == "pure":
        test_pure_mask  = labels[actual_test_idx].sum(axis=1) == 1
        test_images     = test_images[test_pure_mask]
        test_labels     = test_labels[test_pure_mask]
        print(f"Pure filter (all 20): {test_pure_mask.sum()} test retained")
    elif args.label_set in ("classical_pure", "initial_pure"):
        test_pure_mask  = labels[actual_test_idx][:, 0:5].sum(axis=1) == 1
        test_images     = test_images[test_pure_mask]
        test_labels     = test_labels[test_pure_mask]
        print(f"Initial-pure filter : {test_pure_mask.sum()} test retained")
    elif args.label_set == "full" and args.eval_fri_frii_pure:
        _y_te = labels[actual_test_idx]
        test_pure_mask  = (_y_te[:, 0:5].sum(axis=1) == 1) & (_y_te[:, 0] | _y_te[:, 1]).astype(bool)
        test_images     = test_images[test_pure_mask]
        test_labels     = test_labels[test_pure_mask]
        print(f"FRI/FRII-pure eval filter : {test_pure_mask.sum()} test retained")

    print(f"Test set: {len(test_images)}")

    # Full 20-class labels for the final test set — used for cross-eval in gen-aug.
    _te_labels_20_arr = labels[actual_test_idx][:, :20]
    if test_pure_mask is not None:
        _te_labels_20_arr = _te_labels_20_arr[test_pure_mask]

    # Pre-compute initial_pure mask over the actual test set (for sweep cross-eval).
    _ip_mask = (labels[actual_test_idx][:, :5].sum(axis=1) == 1)

    # ── Fold definitions ──────────────────────────────────────────────────
    if args.cv_folds == 1:
        fold_splits = [(train_idx, val_test_idx)]
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_splits = [
            (trainval_idx[tr], trainval_idx[va])
            for tr, va in kf.split(trainval_idx)
        ]
        print(f"K-fold CV: {args.cv_folds} folds on {len(trainval_idx)} trainval samples")

    # ════════════════════════════════════════════════════════════════════
    # PATH A — Full-label training (only when gen_dir is not set)
    # ════════════════════════════════════════════════════════════════════
    all_run_results = []
    all_train_losses, all_test_losses, all_val_losses = [], [], []

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
            if args.label_set == "derived":
                train_labels2 = _make_derived(labels[fold_train_idx])
            else:
                train_labels2 = labels[fold_train_idx][:, label_cols]

            if fold_val_idx is not None:
                val_images = images[fold_val_idx]
                if args.label_set == "derived":
                    val_labels = _make_derived(labels[fold_val_idx])
                else:
                    val_labels = labels[fold_val_idx][:, label_cols]
            else:
                val_images = val_labels = None

            # Pure-source filtering on training and val splits
            tr_mask = np.ones(len(fold_train_idx), dtype=bool)
            if args.label_set == "pure":
                tr_mask = labels[fold_train_idx].sum(axis=1) == 1
                train_images2, train_labels2 = train_images2[tr_mask], train_labels2[tr_mask]
                if val_images is not None:
                    va_mask = labels[fold_val_idx].sum(axis=1) == 1
                    val_images, val_labels = val_images[va_mask], val_labels[va_mask]
            elif args.label_set in ("classical_pure", "initial_pure"):
                tr_mask = labels[fold_train_idx][:, 0:5].sum(axis=1) == 1
                train_images2, train_labels2 = train_images2[tr_mask], train_labels2[tr_mask]
                if val_images is not None:
                    va_mask = labels[fold_val_idx][:, 0:5].sum(axis=1) == 1
                    val_images, val_labels = val_images[va_mask], val_labels[va_mask]

            # Per-class alpha (on CPU — moved to device inside _train_one_run)
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
            alpha = torch.tensor(_alpha_arr, dtype=torch.float32)  # CPU

            fold_out = run_out / f'fold{fold_i + 1}' if args.cv_folds > 1 else run_out
            fold_out.mkdir(parents=True, exist_ok=True)

            # Scattering coefficients (computed once per fold, passed into _train_one_run)
            scat_tr = scat_va = scat_te = None
            if args.model in ('scatternet', 'simplescatternet', 'dualssn'):
                print("Computing scattering coefficients...")
                _scat_parts = [train_images2]
                if val_images is not None:
                    _scat_parts.append(val_images)
                _scat_parts.append(test_images)
                all_scat, _scat_shape = compute_scattering(
                    np.concatenate(_scat_parts, axis=0), J=2, L=8, device=str(device))
                n_tr2 = len(train_images2)
                scat_tr = all_scat[:n_tr2]
                if val_images is not None:
                    n_va    = len(val_images)
                    scat_va = all_scat[n_tr2:n_tr2 + n_va]
                    scat_te = all_scat[n_tr2 + n_va:]
                else:
                    scat_va = None
                    scat_te = all_scat[n_tr2:]
                print(f"  Scattering done: train={scat_tr.shape}, "
                      f"val={'None' if scat_va is None else scat_va.shape}, "
                      f"test={scat_te.shape}")

            results, _, _, _, (trl, tel, val) = _train_one_run(
                train_images2, train_labels2,
                test_images, test_labels,
                _te_labels_20_arr,
                model_name=args.model, n_classes=n_classes, class_names=class_names,
                device=device, args=args, alpha=alpha,
                val_imgs=val_images, val_lbs=val_labels,
                scat_tr=scat_tr, scat_va=scat_va, scat_te=scat_te,
                seed=run_seed, run_dir=fold_out,
                test_source_idx=test_idx,
            )
            all_train_losses.append(trl)
            all_test_losses.append(tel)
            all_val_losses.append(val)
            fold_results_list.append(results)

        # K-fold aggregation
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

    # Aggregated multi-run training curve
    if all_train_losses:
        curve_alpha = max(0.2, 1.0 / max(1, len(all_train_losses)))
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (trl, tel, val) in enumerate(
                zip(all_train_losses, all_test_losses, all_val_losses)):
            ax.plot(trl, 'b-',  alpha=curve_alpha, label='train'      if i == 0 else None)
            ax.plot(tel, 'r--', alpha=curve_alpha, label='test'        if i == 0 else None)
            if val:
                ax.plot(val, 'g:', alpha=curve_alpha, label='validation' if i == 0 else None)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{args.model} — {args.label_set}')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

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

        # Ensemble scores
        all_probs_runs = []
        for run_i in range(1, args.n_runs + 1):
            p = np.load(out_dir / f'run{run_i}' / 'test_probs.npy')
            all_probs_runs.append(p)
        test_labels_arr = np.load(out_dir / 'run1' / 'test_labels.npy')

        f1_vals  = [r['f1_macro'] for r in run_results]
        ranked   = list(np.argsort(f1_vals)[::-1])

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

    # ════════════════════════════════════════════════════════════════════
    # Shared preamble for sweep (Path C) and gen-aug (Path B)
    # ════════════════════════════════════════════════════════════════════
    _skip_sweep = args.skip_sweep or args.model in ('dualssn', 'scatternet', 'simplescatternet')

    _H, _W = images.shape[-2], images.shape[-1]
    if args.model == 'vit':
        _tf_sw       = transforms.Compose([transforms.Resize(224)])
        _tf_train_sw = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(), transforms.RandomRotation(degrees=180)])
    elif args.model == 'enb0':
        _tf_sw       = None
        _tf_train_sw = get_augmentation('quart_ext')
        if args.no_augmentation:
            _tf_train_sw = None
    else:
        _tf_sw       = None
        _tf_train_sw = None
    print(f"  Transforms: val={'Resize(224)' if args.model == 'vit' else 'none'}  "
          f"train={'quart_ext' if args.model == 'enb0' else ('+flip/rotate' if args.model == 'vit' else 'none')}")

    if args.label_set == 'derived':
        _alpha_sw = torch.ones(n_classes, dtype=torch.float32)  # CPU
    else:
        _tr_pure_mask_sw = np.ones(len(train_idx), dtype=bool)
        if args.label_set in ("classical_pure", "initial_pure"):
            _tr_pure_mask_sw = labels[train_idx][:, 0:5].sum(axis=1) == 1
        _alpha_full_sw = compute_class_weights(
            labels[train_idx][_tr_pure_mask_sw, :20],
            args.class_weight_mode,
            args.class_weight_strength,
        )
        _alpha_sw = torch.tensor(_alpha_full_sw[label_cols], dtype=torch.float32)  # CPU

    # Val images/labels for sweep and gen-aug (from val_test_idx when early stopping)
    if val_test_idx is not None:
        _va_images_sw = images[val_test_idx]
        _va_labels_sw = (_make_derived(labels[val_test_idx]) if args.label_set == "derived"
                         else labels[val_test_idx][:, label_cols])
        if args.label_set in ("classical_pure", "initial_pure"):
            _va_pure      = labels[val_test_idx][:, 0:5].sum(axis=1) == 1
            _va_images_sw = _va_images_sw[_va_pure]
            _va_labels_sw = _va_labels_sw[_va_pure]
    else:
        _va_images_sw = _va_labels_sw = None

    # ════════════════════════════════════════════════════════════════════
    # PATH B — Generative augmentation
    # ════════════════════════════════════════════════════════════════════
    if args.gen_dir is not None:
        GEN_FRACS = ([args.gen_frac] if args.gen_frac is not None
                     else [1.0] if args.gen_only
                     else [0.5, 1.0, 2.0])

        import zuko
        from suplat.models.generative_models import FlowMatchingUNet

        _dec_file = args.gen_dir / f"decoder_{args.gen_variant}.pt"
        _nsf_file = args.gen_dir / f"nsf_{args.gen_variant}.pt"
        print(f"  Generator variant : {args.gen_variant}")
        print(f"  Decoder           : {_dec_file}")
        print(f"  NSF               : {_nsf_file}")
        _dec_ckpt = torch.load(_dec_file, map_location="cpu", weights_only=False)
        _decoder  = FlowMatchingUNet(z_dim=_dec_ckpt["feat_dim"], base_ch=_dec_ckpt["base_ch"])
        _decoder.load_state_dict(_dec_ckpt["model_state_dict"])
        _decoder.to(device).eval()

        _nsf_ckpt = torch.load(_nsf_file, map_location="cpu", weights_only=False)
        _nsf      = zuko.flows.NSF(
            features=_nsf_ckpt["feat_dim"],
            context=_nsf_ckpt["n_labels"],
            transforms=_nsf_ckpt["n_transforms"],
        )
        _nsf.load_state_dict(_nsf_ckpt["model_state_dict"])
        _nsf.to(device).eval()

        _n_initial_labels = _nsf_ckpt["n_labels"]  # 5
        _gen_summary = {}

        print(f"\n{'='*60}")
        print(f"Gen-augmentation experiment  ({args.label_set}, {args.model}, {args.n_runs} run(s))")
        print(f"Gen dir: {args.gen_dir}")
        print(f"{'='*60}")

        for _gfrac in GEN_FRACS:
            _frac_dir = out_dir / f"frac_{_gfrac:.2f}"
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

                # Real training images (full set, with pure filter if applicable)
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

                n_real = len(_tr_im)
                n_gen  = int(_gfrac * n_real)
                print(f"\n  frac={_gfrac}  run={_ri}  real={n_real}  gen={n_gen}", flush=True)

                # Generate images by cycling through training labels
                _cond_idx = np.resize(np.arange(n_real), n_gen)
                _cond     = torch.tensor(_tr_initial[_cond_idx], dtype=torch.float32)
                _gen_imgs_list, _gen_lbs_list = [], []

                _GEN_BS = 64
                with torch.no_grad():
                    for _b0 in range(0, n_gen, _GEN_BS):
                        _b1   = min(_b0 + _GEN_BS, n_gen)
                        _cy   = _cond[_b0:_b1].to(device)
                        _z    = _nsf(_cy).sample()          # (B, feat_dim)
                        _xgen = _decoder.sample(_z)          # (B, 1, 89, 89) float [0,1]
                        _xgen_u8 = (_xgen.squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        _gen_imgs_list.append(_xgen_u8)
                        _gen_lbs_list.append(
                            (_make_derived(labels[train_idx[_cond_idx[_b0:_b1]]])
                             if args.label_set == "derived"
                             else labels[train_idx[_cond_idx[_b0:_b1]]][:, label_cols])
                        )

                _gen_imgs = np.concatenate(_gen_imgs_list, axis=0)   # (n_gen, 89, 89) uint8
                _gen_lbs  = np.concatenate(_gen_lbs_list, axis=0)    # (n_gen, n_classes)

                # Combine real + generated (or use generated only)
                if args.gen_only:
                    _aug_imgs = _gen_imgs
                    _aug_lbs  = _gen_lbs
                    print(f"    Gen-only training set: {n_gen} generated images "
                          f"(no real data, n_real={n_real})", flush=True)
                else:
                    _aug_imgs = np.concatenate([_tr_im, _gen_imgs], axis=0)
                    _aug_lbs  = np.concatenate([_tr_lb, _gen_lbs],  axis=0)
                    print(f"    Augmented training set: {len(_aug_imgs)} total "
                          f"({n_real} real + {n_gen} gen)", flush=True)

                # Uniform alpha for gen-aug (on CPU)
                _alpha_g = torch.ones(n_classes, dtype=torch.float32)

                results, _, _, _, _ = _train_one_run(
                    _aug_imgs, _aug_lbs,
                    test_images, test_labels,
                    _te_labels_20_arr,
                    model_name=args.model, n_classes=n_classes, class_names=class_names,
                    device=device, args=args, alpha=_alpha_g,
                    val_imgs=_va_images_sw, val_lbs=_va_labels_sw,
                    extra_results={"gen_frac": _gfrac, "n_real": n_real, "n_gen": n_gen},
                    seed=_run_seed, run_dir=_run_dir,
                )
                print(f"      F1={results['f1_macro']:.4f}  AUC={results['auc_macro']:.4f}"
                      f"  Rec={results['recall_macro']:.4f}  Acc={results['accuracy']:.4f}",
                      flush=True)
                _frac_run_results.append(results)

            # Aggregate across runs for this fraction
            _metrics_agg = {m: float(np.mean([r[m] for r in _frac_run_results]))
                            for m in ("f1_macro", "auc_macro", "accuracy", "recall_macro")}
            if args.n_runs > 1:
                _metrics_agg.update({f"{m}_std": float(np.std([r[m] for r in _frac_run_results]))
                                     for m in ("f1_macro", "auc_macro", "accuracy", "recall_macro")})
            _gen_summary[str(_gfrac)] = _metrics_agg
            print(f"\n  frac={_gfrac}: mean F1={_metrics_agg['f1_macro']:.4f}"
                  f"  AUC={_metrics_agg['auc_macro']:.4f}"
                  f"  Rec={_metrics_agg['recall_macro']:.4f}"
                  f"  Acc={_metrics_agg['accuracy']:.4f}", flush=True)

        with open(out_dir / "gen_aug_summary.json", "w") as _f:
            json.dump(_gen_summary, _f, indent=2)
        print(f"\nGen-aug summary saved → {out_dir / 'gen_aug_summary.json'}")

    # ════════════════════════════════════════════════════════════════════
    # PATH C — Label-fraction sweep (only when gen_dir is None)
    # ════════════════════════════════════════════════════════════════════
    elif not _skip_sweep:
        _FRACS_SUB = [0.01, 0.05, 0.10, 0.25, 0.50]  # 100% loaded from results.json

        print(f"\n{'='*60}")
        print(f"Label-fraction sweep  ({args.label_set}, {args.model}, {args.n_runs} run(s))")
        print(f"{'='*60}")

        for _ri in range(1, args.n_runs + 1):
            _run_seed   = args.seed + (_ri - 1)
            _run_out    = out_dir / f'run{_ri}' if args.n_runs > 1 else out_dir
            _frac_cache    = _run_out / "label_fraction_metrics.json"
            _need_ip_frac  = args.label_set not in (
                "initial_pure", "classical_pure", "morphology_pure",
                "environment_pure", "derived")
            _frac_ip_cache = _run_out / "label_fraction_metrics_initial_pure_eval.json"

            if (_frac_cache.exists()
                    and (not _need_ip_frac or _frac_ip_cache.exists())
                    and not args.force):
                print(f"  Run {_ri}: fraction cache exists — skipping.")
                continue

            print(f"\n  --- Run {_ri} / {args.n_runs}  (seed={_run_seed}) ---", flush=True)
            _frac_out: dict    = {}
            _ALL_FRACS_IP      = ["1.0"] + [str(f) for f in _FRACS_SUB]
            _frac_ip_out: dict = {k: {} for k in _ALL_FRACS_IP}

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
                # ip cross-eval 100% — derive from saved test_probs/test_labels
                if _need_ip_frac:
                    _pp100 = _run_out / "test_probs.npy"
                    _ll100 = _run_out / "test_labels.npy"
                    if _pp100.exists() and _ll100.exists():
                        _tp100 = np.load(_pp100); _tl100 = np.load(_ll100)
                        _ipm   = _tl100[:, :5].sum(axis=1) == 1
                        if _ipm.any():
                            _p5  = _tp100[_ipm, :5]
                            _p5n = _p5 / _p5.sum(axis=1, keepdims=True).clip(min=1e-9)
                            _yt5 = _tl100[_ipm, :5].argmax(axis=1)
                            _yp5 = _p5n.argmax(axis=1)
                            from sklearn.metrics import (f1_score as _f1s, accuracy_score as _accs,
                                recall_score as _recs, roc_auc_score as _aucs)
                            from sklearn.preprocessing import label_binarize as _lbin
                            try:
                                _a5 = float(_aucs(_lbin(_yt5, classes=list(range(5))),
                                                  _p5n, multi_class="ovr", average="macro"))
                            except Exception:
                                _a5 = float("nan")
                            _frac_ip_out["1.0"] = {"Supervised": {
                                "f1":       float(_f1s(_yt5, _yp5, average="macro", zero_division=0)),
                                "auc":      _a5,
                                "accuracy": float(_accs(_yt5, _yp5)),
                                "recall":   float(_recs(_yt5, _yp5, average="macro", zero_division=0)),
                            }}
                            print(f"    100% ip: F1={_frac_ip_out['1.0']['Supervised']['f1']:.4f}",
                                  flush=True)
            else:
                print(f"    100%: results.json not found — skipping 100% point.", flush=True)

            # Sub-100% fractions: retrain with this run's seed, no file saving
            _rng_sw = np.random.default_rng(_run_seed)

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

                results, _tp_sw, _, _yt_sw, _ = _train_one_run(
                    _tr_im_sw, _tr_lb_sw,
                    test_images, test_labels,
                    _te_labels_20_arr,
                    model_name=args.model, n_classes=n_classes, class_names=class_names,
                    device=device, args=args, alpha=_alpha_sw,
                    val_imgs=_va_images_sw, val_lbs=_va_labels_sw,
                    seed=_run_seed, run_dir=None,  # no file saving for sweep fractions
                )
                _frac_out[str(_frac)] = {"Supervised": {
                    "f1":       results["f1_macro"],
                    "auc":      results["auc_macro"],
                    "accuracy": results["accuracy"],
                    "recall":   results["recall_macro"],
                }}
                print(f"      F1={results['f1_macro']:.4f}  AUC={results['auc_macro']:.4f}"
                      f"  Acc={results['accuracy']:.4f}", flush=True)

                # ip cross-eval
                if _need_ip_frac and _yt_sw.ndim == 2 and _yt_sw.shape[1] >= 5:
                    _ipm_f = _yt_sw[:, :5].sum(axis=1) == 1
                    if _ipm_f.any():
                        _p5f  = _tp_sw[_ipm_f, :5]
                        _p5fn = _p5f / _p5f.sum(axis=1, keepdims=True).clip(min=1e-9)
                        _yt5f = _yt_sw[_ipm_f, :5].argmax(axis=1)
                        _yp5f = _p5fn.argmax(axis=1)
                        from sklearn.metrics import (f1_score as _f1s, accuracy_score as _accs,
                            recall_score as _recs, roc_auc_score as _aucs)
                        from sklearn.preprocessing import label_binarize as _lbin
                        try:
                            _a5f = float(_aucs(_lbin(_yt5f, classes=list(range(5))),
                                               _p5fn, multi_class="ovr", average="macro"))
                        except Exception:
                            _a5f = float("nan")
                        _frac_ip_out[str(_frac)] = {"Supervised": {
                            "f1":       float(_f1s(_yt5f, _yp5f, average="macro", zero_division=0)),
                            "auc":      _a5f,
                            "accuracy": float(_accs(_yt5f, _yp5f)),
                            "recall":   float(_recs(_yt5f, _yp5f, average="macro", zero_division=0)),
                        }}

            if not _frac_cache.exists() or args.force:
                if _frac_out:
                    with open(_frac_cache, "w") as _fh:
                        json.dump(_frac_out, _fh, indent=2)
                    print(f"    Saved → {_frac_cache}", flush=True)
            if _need_ip_frac and any(_frac_ip_out.values()):
                with open(_frac_ip_cache, "w") as _fh:
                    json.dump(_frac_ip_out, _fh, indent=2)
                print(f"    Saved → {_frac_ip_cache}", flush=True)

    else:
        print(f"\nLabel-fraction sweep skipped for model '{args.model}'.")


if __name__ == '__main__':
    main()
