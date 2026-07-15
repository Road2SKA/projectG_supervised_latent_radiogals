#!/usr/bin/env python3
"""
Fine-tune a BYOL-pretrained encoder for multi-label radio galaxy classification.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.utils.class_weights import compute_class_weights
from suplat.data.augmentations import get_augmentation
from suplat.data.data_samplers import ImagesAndLabelsDataset
from suplat.models.byol_models import (
    BYOLEfficientNetB0,
    BYOLPretrainedBackbone,
    create_convnext_tiny_backbone,
    create_resnet18_backbone,
    create_resnet50_backbone,
)

TRAINING_MODE_DESCRIPTIONS = {
    1: "Freeze embeddings: frozen encoder and projector; train linear classifier only.",
    2: "Freeze features: frozen encoder; fine-tune projector and train linear classifier.",
    3: "No freeze: fine-tune encoder and projector; train linear classifier.",
    4: "Supervised: initialize from scratch and train encoder, projector, and linear classifier.",
}

def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected an integer >= 1.")
    return value

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("using gpu")
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    print("using cpu")
    device = torch.device('cpu')

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    """Parse command-line arguments for BYOL encoder fine-tuning."""
    ap = argparse.ArgumentParser(
        description="Fine-tune a BYOL-pretrained model for multi-label classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Training modes:\n"
            + "\n".join(
                f"  {mode}: {description}"
                for mode, description in TRAINING_MODE_DESCRIPTIONS.items()
            )
        ),
    )

    # Required fine-tuning configuration
    ap.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the pretrained BYOL checkpoint to fine-tune.",
    )
    ap.add_argument(
        "--training-mode",
        type=int,
        default=3,
        choices=tuple(TRAINING_MODE_DESCRIPTIONS),
        help="Training strategy. See descriptions below.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of fine-tuning epochs.",
    )
    ap.add_argument(
        "--n-runs", "--n-models",
        type=positive_int,
        default=10,
        dest="n_runs",
        help="Number of training runs with different initialisations. "
             "Run i uses seed DATA_SEED + (i-1). Default: 10.",
    )
    # Data and run configuration
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss"),
        help="Directory containing images_filtered.npy and labels_filtered.npy.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for train/test loaders.",
    )
    ap.add_argument(
        "--augmentation",
        type=str,
        default="quart",
        choices=["quart", "cont", "quart_ext", "cont_ext", "baronperez"],
        help="Augmentation pipeline for training images.",
    )
    # Optimization and output
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of DataLoader worker processes.",
    )

    ap.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Unique name for this fine-tuning run (used in output directory).",
    )
    ap.add_argument(
        "--class-weight-mode",
        type=str,
        default=None,
        choices=["score", "initial", "morphology", "environment", "classical", "all"],
        help="Upweight rare samples in training loss: 'score' (interest tier 1-4) or "
             "a label-set name (inverse frequency). Default: None (uniform).",
    )
    ap.add_argument(
        "--class-weight-strength",
        type=float,
        default=0.0,
        help="Magnitude of class upweighting (0=uniform, default). "
             "w = clip(1 + strength*(raw_norm - 1), min=0).",
    )
    ap.add_argument(
        "--f-label",
        type=float,
        default=None,
        help="Override label fraction (0–1). Defaults to the value stored in the checkpoint config.",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs with no improvement on test loss). Default: 15.",
    )

    return ap.parse_args()


class BYOLFineTuner(nn.Module):
    def __init__(self, byol_model, num_classes=21, training_mode=3):
        super().__init__()

        self.encoder = byol_model.online_encoder
        self.projector = byol_model.online_projector
        self.training_mode = training_mode

        # BYOL projection_dim = 128 in your case
        self.classifier = nn.Linear(128, num_classes)

        self.apply_training_mode(training_mode)

    def apply_training_mode(self, training_mode):
        """
        training_mode:
            1 = freeze encoder + projector, train classifier only
            2 = freeze encoder, train projector + classifier
            3 = train encoder + projector + classifier
            4 = train encoder + projector + classifier from scratch
        """
        if training_mode not in TRAINING_MODE_DESCRIPTIONS:
            raise ValueError(
                f"Unknown training_mode={training_mode}. "
                f"Expected one of {sorted(TRAINING_MODE_DESCRIPTIONS)}."
            )

        for p in self.encoder.parameters():
            p.requires_grad = training_mode in {3, 4}
        for p in self.projector.parameters():
            p.requires_grad = training_mode in {2, 3, 4}
        for p in self.classifier.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)

        if mode:
            if self.training_mode in {1, 2}:
                self.encoder.eval()
            if self.training_mode == 1:
                self.projector.eval()

        return self

    def forward(self, x):
        z = self.encoder(x)
        z = self.projector(z)
        logits = self.classifier(z)
        return logits

args = parse_args()

BYOL_PATH = args.model_path / "byol_model_best.pt"
finetune_name = "finetuning" if args.run_name == "" else f"finetuning_{args.run_name}"
OUTPUT_DIR = args.model_path / "data" / "classifiers" / finetune_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = torch.load(BYOL_PATH, map_location="cpu", weights_only=True)
print(checkpoint["config"])

# Data paths
IMAGES_PATH = args.data_dir / 'images_filtered.npy'
LABELS_PATH = args.data_dir / 'labels_filtered.npy'

images = np.load(IMAGES_PATH).astype(np.float32)/255
labels = np.load(LABELS_PATH)

print(labels.shape)

DATA_SEED = checkpoint["config"]['data_seed']
F_LABEL = args.f_label if args.f_label is not None else checkpoint["config"]['f_label']
TRAINING_MODE = args.training_mode
LEARNING_RATE = args.lr
N_RUNS = args.n_runs

TRAIN_RATIO, TEST_RATIO = 0.70, 0.30

all_idx = np.arange(len(images))
train_idx, test_idx = train_test_split(all_idx, test_size=TEST_RATIO, random_state=DATA_SEED)
train_images, train_labels = images[train_idx], labels[train_idx]
test_images  = images[test_idx]
test_labels  = labels[test_idx]

byol_strong_aug = get_augmentation(args.augmentation)

# Test set: no augmentation
test_lab_ds = ImagesAndLabelsDataset(tags_data=pd.DataFrame(test_labels), img_data=test_images,
                                     transform=None)
labelled_test_loader = DataLoader(test_lab_ds, batch_size=args.batch_size, shuffle=False,
                                  drop_last=False, num_workers=args.num_workers, pin_memory=use_cuda)


def _make_labeled_loader(f_label, seed):
    """Create a labelled-subset DataLoader at fraction f_label using the given seed."""
    if f_label <= 0.0:
        _mask = np.zeros(len(train_idx), dtype=bool)
    elif f_label >= 1.0:
        _mask = np.ones(len(train_idx), dtype=bool)
    else:
        _strat_key = np.argmax(train_labels[:, :min(5, train_labels.shape[1])], axis=1)
        _n_lab = max(2, int(round(f_label * len(train_idx))))
        try:
            _sss = StratifiedShuffleSplit(n_splits=1, train_size=_n_lab, random_state=seed)
            _lab_rel, _ = next(_sss.split(train_images, _strat_key))
        except ValueError:
            print("Stratification failed, falling back to random selection")
            _lab_rel = np.random.choice(len(train_idx), _n_lab, replace=False)
        _mask = np.zeros(len(train_idx), dtype=bool)
        _mask[_lab_rel] = True
    _lab_imgs = train_images[_mask]
    _lab_labs = train_labels[_mask]
    _alpha = compute_class_weights(
        _lab_labs[:, :20], args.class_weight_mode, args.class_weight_strength)
    if labels.shape[1] > 20:
        _alpha = np.append(_alpha, np.ones(labels.shape[1] - 20, dtype=np.float32))
    _alpha_t   = torch.tensor(_alpha, dtype=torch.float32)
    _alpha_sum = float(_alpha_t.sum().clamp(min=1e-6))
    _lab_df = pd.DataFrame(_lab_labs)
    _lab_ds = ImagesAndLabelsDataset(
        tags_data=_lab_df, img_data=_lab_imgs, transform=byol_strong_aug)
    _loader = DataLoader(
        _lab_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=(len(_lab_ds) > args.batch_size),
        num_workers=args.num_workers, pin_memory=use_cuda)
    return _lab_ds, _loader, _alpha_t, _alpha_sum


lab_ds, labelled_train_loader, _alpha_t, _alpha_sum = _make_labeled_loader(F_LABEL, DATA_SEED)

print(f"With current settings, Train/Test sizes are {len(lab_ds)}/{len(test_lab_ds)}")

print(f"Training mode {TRAINING_MODE}: {TRAINING_MODE_DESCRIPTIONS[TRAINING_MODE]}")
print(f"Training {N_RUNS} run(s) with different initialisations")

num_classes = labels.shape[1]

NUM_EPOCHS = args.epochs


def seed_model_initialization(model_idx):
    seed = DATA_SEED + model_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    return seed


def seed_training_randomness():
    random.seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(DATA_SEED)


def build_finetune_model(model_idx):
    run_seed = seed_model_initialization(model_idx)

    model = BYOLEfficientNetB0(
        projection_dim=checkpoint["config"]['projection_dim'],
        hidden_dim=checkpoint["config"]['hidden_dim'],
        bn_momentum=0.1,
        feature_compression_mode=checkpoint["config"]['projector'],
        dropout_rate=0.2,
    )

    if TRAINING_MODE == 4:
        print(
            "Training from scratch with checkpoint architecture and newly initialized weights"
        )
    else:
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    finetune_model = BYOLFineTuner(
        byol_model=model,
        num_classes=num_classes,
        training_mode=TRAINING_MODE,
    )

    return finetune_model.to(device), run_seed


def train_one_model(finetune_model, train_loader, alpha_t, alpha_sum):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, finetune_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_state = None
    epochs_no_impr = 0

    for epoch in range(NUM_EPOCHS):

        # ======================
        # TRAIN
        # ======================
        finetune_model.train()

        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            x1, x1_aug, x1_lab = batch

            x1_aug = x1_aug.float().to(device)
            y = x1_lab.float().to(device)
            w_dev = alpha_t.to(device)

            if y.ndim == 3 and y.shape[1] == 1:
                y = y.squeeze(1)

            optimizer.zero_grad()

            logits = finetune_model(x1_aug)
            unreduced = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss = (unreduced * w_dev).sum(1) / alpha_sum
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # ======================
        # TEST
        # ======================
        finetune_model.eval()

        test_loss = 0.0
        test_batches = 0

        with torch.no_grad():
            for batch in labelled_test_loader:
                x1, x1_aug, x1_lab = batch

                x1_aug = x1_aug.float().to(device)
                y = x1_lab.float().to(device)

                if y.ndim == 3 and y.shape[1] == 1:
                    y = y.squeeze(1)

                logits = finetune_model(x1_aug)
                loss = F.binary_cross_entropy_with_logits(logits, y)

                test_loss += loss.detach().item()
                test_batches += 1

        avg_test_loss = test_loss / test_batches
        test_losses.append(avg_test_loss)
        scheduler.step(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_state = {k: v.cpu().clone() for k, v in finetune_model.state_dict().items()}
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"| Train: {avg_train_loss:.4f} "
            f"| Test: {avg_test_loss:.4f} "
            f"| lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if epochs_no_impr >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        finetune_model.load_state_dict(best_state)

    return train_losses, test_losses, optimizer


def compute_finetuning_metrics(finetune_model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    labels_list = []
    probs_list = []

    finetune_model.eval()
    with torch.inference_mode():
        for _, x_aug, y in loader:
            x_aug = x_aug.float().to(device, non_blocking=use_cuda)
            y = y.float().to(device, non_blocking=use_cuda)

            if y.ndim == 3 and y.shape[1] == 1:
                y = y.squeeze(1)

            probs_list.append(torch.sigmoid(finetune_model(x_aug)).cpu())
            labels_list.append(y.cpu())

    y_true = torch.cat(labels_list).to(torch.bool)
    y_prob = torch.cat(probs_list)
    y_pred = y_prob >= 0.5
    n_members = y_true.sum(dim=0)

    tp = (y_pred & y_true).sum(dim=0).float()
    fp = (y_pred & ~y_true).sum(dim=0).float()
    fn = (~y_pred & y_true).sum(dim=0).float()
    tn = (~y_pred & ~y_true).sum(dim=0).float()

    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    accuracy = (tp + tn) / y_true.shape[0]

    y_true_np = y_true.numpy()
    y_prob_np = y_prob.numpy()
    auc = []
    for i in range(num_classes):
        if np.unique(y_true_np[:, i]).size < 2:
            auc.append(None)
        else:
            auc.append(float(roc_auc_score(y_true_np[:, i], y_prob_np[:, i])))

    print(f'shape of y_true: {y_true.shape[0]}')

    metrics = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "accuracy": accuracy.tolist(),
        "auc": auc,
        "macro_f1": [float(f1.mean().item())],
        "n_members": n_members.tolist(),
    }
    return metrics, y_prob_np, y_pred.numpy(), y_true_np


all_train_losses = []
all_test_losses = []
all_train_metrics = []
all_test_metrics = []

for model_idx in range(N_RUNS):
    run_number = model_idx + 1
    print(f"Starting run {run_number}/{N_RUNS}")

    finetune_model, run_seed = build_finetune_model(model_idx)
    seed_training_randomness()
    train_losses, test_losses, optimizer = train_one_model(
        finetune_model, labelled_train_loader, _alpha_t, _alpha_sum)

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

    checkpoint_path = OUTPUT_DIR / f"finetuned_model_{run_number}.pt"
    torch.save(
        {
            "model_state_dict": finetune_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "pretrained_model_path": str(BYOL_PATH),
                "training_mode": TRAINING_MODE,
                "training_mode_description": TRAINING_MODE_DESCRIPTIONS[TRAINING_MODE],
                "lr": LEARNING_RATE,
                "weight_decay": args.weight_decay,
                "epochs": NUM_EPOCHS,
                "num_classes": num_classes,
                "data_seed": DATA_SEED,
                "initialization_seed": run_seed,
                "f_label": F_LABEL,
                "model_index": run_number,
                "n_runs": N_RUNS,
            },
            "train_losses": train_losses,
            "test_losses": test_losses,
        },
        checkpoint_path,
    )
    print(f"Finetuned model checkpoint saved to {checkpoint_path}")

    train_metrics, _, _, _ = compute_finetuning_metrics(finetune_model, lab_ds)
    test_metrics, test_probs_arr, test_preds_arr, test_labels_arr = compute_finetuning_metrics(
        finetune_model, test_lab_ds
    )
    all_train_metrics.append(train_metrics)
    all_test_metrics.append(test_metrics)

    run_dir = OUTPUT_DIR / f'run{run_number}' if N_RUNS > 1 else OUTPUT_DIR
    run_dir.mkdir(exist_ok=True)
    np.save(run_dir / 'test_probs.npy',  test_probs_arr.astype(np.float32))
    np.save(run_dir / 'test_preds.npy',  test_preds_arr)
    np.save(run_dir / 'test_labels.npy', test_labels_arr)
    print(f"Per-run arrays saved to: {run_dir}")

    del finetune_model
    if use_cuda:
        torch.cuda.empty_cache()

fig, ax = plt.subplots(figsize=(6, 4))
curve_alpha = 0.45 if N_RUNS > 1 else 1.0
for model_idx, (train_losses, test_losses) in enumerate(
    zip(all_train_losses, all_test_losses)
):
    ax.plot(
        train_losses,
        'b-',
        alpha=curve_alpha,
        label="train" if model_idx == 0 else None,
    )
    ax.plot(
        test_losses,
        'r--',
        alpha=curve_alpha,
        label="test" if model_idx == 0 else None,
    )
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Finetuning")
ax.set_yscale("log")
ax.legend()
ax.grid(True)
fig.tight_layout()
curve_path = OUTPUT_DIR / "learning_curve.png"
fig.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Learning curve saved to {curve_path}")

metrics = {
    key: {
        "train": [run_metrics[key] for run_metrics in all_train_metrics],
        "test": [run_metrics[key] for run_metrics in all_test_metrics],
    }
    for key in all_train_metrics[0]
}

metrics_path = OUTPUT_DIR / "finetuning_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Finetuning metrics saved to {metrics_path}")

# =============================================================================
# LABEL-FRACTION SWEEP
# Train N_RUNS models at each sub-100% fraction; save test_probs.npy /
# test_labels.npy to OUTPUT_DIR/frac_{f:.2f}/run{i}/ for notebook loading.
# 100% results already live in OUTPUT_DIR/run{i}/.
# Each run i uses a different labelled-subset seed (DATA_SEED + i - 1) so
# that std across runs reflects genuine variability.
# =============================================================================
_FRACS_SUB = [0.01, 0.05, 0.10, 0.25, 0.50]

print(f"\n{'='*60}")
print(f"Label-fraction sweep ({len(_FRACS_SUB)} fractions × {N_RUNS} runs)")
print(f"{'='*60}")

for _frac in _FRACS_SUB:
    _frac_dir = OUTPUT_DIR / f"frac_{_frac:.2f}"

    # Check whether every run for this fraction is already complete
    _all_done = all(
        (_frac_dir / f"run{_ri}" / "test_probs.npy").exists()
        for _ri in range(1, N_RUNS + 1)
    )
    if _all_done:
        print(f"\nfrac={_frac:.0%}: all {N_RUNS} run(s) already exist, skipping.")
        continue

    print(f"\nfrac={_frac:.0%} ── {int(round(_frac * len(train_idx)))} labelled train sources")
    _frac_dir.mkdir(parents=True, exist_ok=True)

    for _ri in range(1, N_RUNS + 1):
        _run_dir = _frac_dir / f"run{_ri}"
        if (_run_dir / "test_probs.npy").exists():
            print(f"  run{_ri}: already exists, skipping.")
            continue
        _run_dir.mkdir(exist_ok=True)

        # Different labelled subset per run (mirrors baseline script behaviour)
        _run_seed = DATA_SEED + (_ri - 1)
        _, _train_loader, _a_t, _a_sum = _make_labeled_loader(_frac, _run_seed)

        _ft_model, _ = build_finetune_model(_ri - 1)
        seed_training_randomness()
        train_one_model(_ft_model, _train_loader, _a_t, _a_sum)

        _, _probs, _, _lbls = compute_finetuning_metrics(_ft_model, test_lab_ds)
        np.save(_run_dir / "test_probs.npy",  _probs.astype(np.float32))
        np.save(_run_dir / "test_labels.npy", _lbls)
        print(f"  run{_ri}: saved to {_run_dir}")

        del _ft_model
        if use_cuda:
            torch.cuda.empty_cache()

print(f"\nLabel-fraction sweep complete.")

