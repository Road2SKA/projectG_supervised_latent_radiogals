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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
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
    "classical":   [0, 1],
    "initial":     list(range(0, 5)),
    "environment": list(range(16, 20)),
    "derived":     [2, 5, 6, 10, 11],
    "morphology":  list(range(5, 15)),
    "all":         list(range(0, 20)),
    "pure":        list(range(0, 20)),
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
def compute_pos_weights(y_train, device):
    y_t = torch.from_numpy(y_train.astype(np.float32))
    pos_counts = y_t.sum(dim=0).clamp(min=1)
    neg_counts = len(y_t) - pos_counts
    return (neg_counts / pos_counts).clamp(max=20).to(device)


def weighted_bce_loss(logits, targets, pos_weights, n_classes):
    """
    logits: (B, n_classes, 2)
    targets: (B, n_classes) int64
    """
    loss = sum(
        F.cross_entropy(
            logits[:, c, :], targets[:, c],
            weight=torch.tensor([1.0, pos_weights[c].item()], device=logits.device)
        )
        for c in range(n_classes)
    ) / n_classes
    return loss


def evaluate_metrics(y_true, y_pred, y_prob, class_names):
    n = len(class_names)
    aucs = []
    for i in range(n):
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(float('nan'))
        else:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc    = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))
    mac_auc = float(np.nanmean(aucs))

    return {
        'f1_macro':      float(f1_mac),
        'auc_macro':     mac_auc,
        'accuracy':      float(acc),
        'f1_per_class':  f1_per,
        'auc_per_class': [float(a) if not np.isnan(a) else None for a in aucs],
        'class_names':   class_names,
    }


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


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
        backbone = create_efficientnet_b0_backbone(num_channels=1, img_size=224)
        base = nn.Sequential(
            backbone,
            nn.Flatten(),
            nn.Linear(1280, n_out)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Wrap output as (B, n_classes, 2)
    class MultiLabelWrapper(nn.Module):
        def __init__(self, inner, n_cl):
            super().__init__()
            self.inner = inner
            self.n_cl  = n_cl

        def forward(self, *args):
            out = self.inner(*args)       # (B, n_classes * 2)
            return out.view(out.size(0), self.n_cl, 2)

    # DualSSN already has its own forward; wrap appropriately
    if model_name == 'dualssn':
        class DualWrapper(nn.Module):
            def __init__(self, inner, n_cl):
                super().__init__()
                self.inner = inner
                self.n_cl  = n_cl

            def forward(self, img, scat):
                out = self.inner(img, scat)
                return out.view(out.size(0), self.n_cl, 2)
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
    parser.add_argument('--model',      required=True,
                        choices=['cnn', 'scatternet', 'simplescatternet',
                                 'vit', 'dualssn', 'enb0'])
    parser.add_argument('--label_set',  default='classical',
                        choices=list(LABEL_SETS.keys()))
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int, default=15)
    parser.add_argument('--cv_folds',   type=int, default=1)
    parser.add_argument('--run_name',   type=str, default=None,
                        help="Custom run name prefix (default: run_dir basename + timestamp)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────
    _ts = datetime.now().strftime('%Y%m%d_%H%M')
    _prefix = args.run_name if args.run_name else args.run_dir.name
    out_dir = args.run_dir / 'baselines' / f'{_prefix}_{args.model}_{_ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    images = np.load(args.data_dir / 'images_filtered.npy')
    labels = np.load(args.data_dir / 'labels_filtered.npy')
    print(f"Images: {images.shape}, Labels: {labels.shape}")

    # ── Label subset ──────────────────────────────────────────────────────
    label_cols  = LABEL_SETS[args.label_set]
    class_names = [ALL_CLASS_NAMES[i] for i in label_cols]
    n_classes   = len(class_names)
    print(f"Label set: {args.label_set} ({n_classes} classes: {class_names})")

    # ── 70/15/15 split — same logic as create_embeddings.py ──────────────
    all_idx = np.arange(len(images))
    trainval_idx, test_idx = train_test_split(all_idx,      test_size=0.30, random_state=args.seed)
    train_idx,    val_idx  = train_test_split(trainval_idx, test_size=0.50, random_state=args.seed)

    test_images = images[test_idx]
    test_labels = labels[test_idx][:, label_cols]
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

    fold_results_list = []

    for fold_i, (fold_train_idx, fold_val_idx) in enumerate(fold_splits):
        if args.cv_folds > 1:
            print(f"\n{'='*60}\nFold {fold_i + 1} / {args.cv_folds}\n{'='*60}")

        train_images2 = images[fold_train_idx]
        train_labels2 = labels[fold_train_idx][:, label_cols]
        val_images    = images[fold_val_idx]
        val_labels    = labels[fold_val_idx][:, label_cols]
        print(f"  Train: {len(train_images2)}, Val: {len(val_images)}")

        fold_out = out_dir / f'fold{fold_i + 1}' if args.cv_folds > 1 else out_dir
        fold_out.mkdir(parents=True, exist_ok=True)

        # ── Image size ────────────────────────────────────────────────────
        H, W = train_images2.shape[-2], train_images2.shape[-1]
        img_shape = (1, H, W)

        # ── Transforms ───────────────────────────────────────────────────
        needs_upsample = args.model in ('vit', 'enb0')
        if needs_upsample:
            tf = transforms.Compose([transforms.Resize(224)])
            print(f"  Upsampling images to 224×224 for {args.model}")
        else:
            tf = None

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
            train_dl = make_loader(RadioImageDataset(train_images2, train_labels2, tf),
                                   args.batch_size, shuffle=True)
            val_dl   = make_loader(RadioImageDataset(val_images, val_labels, tf),
                                   args.batch_size, shuffle=False)
            test_dl  = make_loader(RadioImageDataset(test_images, test_labels, tf),
                                   args.batch_size, shuffle=False)
            use_scat_only = False

        # ── Pos weights ───────────────────────────────────────────────────
        pos_weights = compute_pos_weights(train_labels2, device)

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

                loss = weighted_bce_loss(logits, labels_b, pos_weights, n_classes)
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
                    val_loss_total += weighted_bce_loss(logits, labels_b, pos_weights, n_classes).item() * len(labels_b)

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
        print(f"\nMacro F1:  {results['f1_macro']:.4f}")
        print(f"Macro AUC: {results['auc_macro']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.4f}")

        # ── Save outputs ──────────────────────────────────────────────────
        np.save(fold_out / 'test_probs.npy',  test_probs)
        np.save(fold_out / 'test_preds.npy',  test_preds)
        np.save(fold_out / 'test_labels.npy', test_labels_arr)

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

        with open(fold_out / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nAll outputs saved to: {fold_out}")
        fold_results_list.append(results)

    if args.cv_folds > 1:
        f1s  = [r['f1_macro']  for r in fold_results_list]
        aucs = [r['auc_macro'] for r in fold_results_list]
        accs = [r['accuracy']  for r in fold_results_list]
        agg = {
            'cv_folds':       args.cv_folds,
            'f1_macro_mean':  float(np.mean(f1s)),
            'f1_macro_std':   float(np.std(f1s)),
            'auc_macro_mean': float(np.mean(aucs)),
            'auc_macro_std':  float(np.std(aucs)),
            'accuracy_mean':  float(np.mean(accs)),
            'accuracy_std':   float(np.std(accs)),
            'per_fold':       fold_results_list,
        }
        print(f"\nK-fold summary ({args.cv_folds} folds):")
        print(f"  Macro F1:  {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
        print(f"  Macro AUC: {agg['auc_macro_mean']:.4f} ± {agg['auc_macro_std']:.4f}")
        print(f"  Accuracy:  {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
        with open(out_dir / 'cv_results.json', 'w') as f:
            json.dump(agg, f, indent=2)


if __name__ == '__main__':
    main()
