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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.data.augmentations import get_augmentation
from suplat.data.data_samplers import ImagesAndLabelsDataset
from suplat.models.byol_models import (
    BYOLEfficientNetB0,
    BYOLPretrainedBackbone,
    create_convnext_tiny_backbone,
    create_resnet18_backbone,
    create_resnet50_backbone,
)

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
        description="Fine-tune a BYOL-pretrained model for multi-label classification."
    )

    # Required fine-tuning configuration
    ap.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the pretrained BYOL checkpoint to fine-tune.",
    )
    ap.add_argument(
        "--freeze-layers",
        type=int,
        default=5,
        help=(
            "Number of early encoder feature blocks to freeze. "
            "For EfficientNet-B0, 0 trains all blocks and 9 freezes all feature blocks."
        ),
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
        default="standard",
        choices=["standard", "extended"],
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

    return ap.parse_args()


class BYOLFineTuner(nn.Module):
    def __init__(self, byol_model, num_classes=21, freeze_until_feature_idx=0):
        super().__init__()

        self.encoder = byol_model.online_encoder
        self.projector = byol_model.online_projector

        # BYOL projection_dim = 128 in your case
        self.classifier = nn.Linear(128, num_classes)

        self.freeze_encoder_until(freeze_until_feature_idx)

    def freeze_encoder_until(self, freeze_until_feature_idx):
        """
        freeze_until_feature_idx:
            0 = train all encoder
            5 = freeze encoder.features[0] ... encoder.features[4]
            9 = freeze all EfficientNet features
        """

        # First train everything
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

        # Then freeze early EfficientNet blocks
        for idx, block in enumerate(self.encoder.features):
            if idx < freeze_until_feature_idx:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x):
        z = self.encoder(x)
        z = self.projector(z)
        logits = self.classifier(z)
        return logits

args = parse_args()

BYOL_PATH = args.model_path / "byol_model_best.pt"
finetune_name = "finetuning" if args.run_name == "" else f"finetuning_{args.run_name}"
OUTPUT_DIR = args.model_path / finetune_name
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
F_LABEL = checkpoint["config"]['f_label']
FREEZE_LAYERS = args.freeze_layers
LEARNING_RATE = args.lr

TRAIN_RATIO, TEST_RATIO = 0.70, 0.30

all_idx = np.arange(len(images))
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

byol_strong_aug = get_augmentation("standard")

lab_df = pd.DataFrame(labelled_labels)
lab_ds = ImagesAndLabelsDataset(tags_data=lab_df, img_data=labelled_images,
                         transform=byol_strong_aug)

labelled_train_loader = DataLoader(lab_ds, batch_size=256, shuffle=True, drop_last=True,
                          num_workers=1, pin_memory=use_cuda)

test_lab_ds = ImagesAndLabelsDataset(tags_data=pd.DataFrame(test_labels), img_data=test_images,
                         transform=byol_strong_aug)
labelled_test_loader = DataLoader(test_lab_ds, batch_size=256, shuffle=True, drop_last=True,
                                  num_workers=1, pin_memory=use_cuda)

print(f"With current settings, Train/Test sizes are {len(lab_ds)}/{len(test_lab_ds)}")

model = BYOLEfficientNetB0(
    projection_dim=checkpoint["config"]['projection_dim'],
    hidden_dim=checkpoint["config"]['hidden_dim'],
    bn_momentum=0.1,
    feature_compression_mode=checkpoint["config"]['projector'],
    dropout_rate=0.2,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

num_classes = labels.shape[1]

finetune_model = BYOLFineTuner(
    byol_model=model,
    num_classes=num_classes,
    freeze_until_feature_idx=FREEZE_LAYERS,
)

finetune_model = finetune_model.to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, finetune_model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

criterion = nn.BCEWithLogitsLoss()

NUM_EPOCHS = args.epochs

train_losses = []
test_losses = []

for epoch in range(NUM_EPOCHS):

    # ======================
    # TRAIN
    # ======================
    finetune_model.train()

    train_loss = 0.0
    train_batches = 0

    for batch in labelled_train_loader:
        x1, x1_aug, x1_lab = batch

        x1_aug = x1_aug.float().to(device)
        y = x1_lab.float().to(device)

        if y.ndim == 3 and y.shape[1] == 1:
            y = y.squeeze(1)

        optimizer.zero_grad()

        logits = finetune_model(x1_aug)
        loss = criterion(logits, y)

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
            loss = criterion(logits, y)

            test_loss += loss.detach().item()
            test_batches += 1

    avg_test_loss = test_loss / test_batches
    test_losses.append(avg_test_loss)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| Train: {avg_train_loss:.4f} "
        f"| Test: {avg_test_loss:.4f}"
    )

checkpoint_path = OUTPUT_DIR / "finetuned_model.pt"
torch.save(
    {
        "model_state_dict": finetune_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "pretrained_model_path": str(BYOL_PATH),
            "freeze_layers": FREEZE_LAYERS,
            "lr": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "num_classes": num_classes,
            "data_seed": DATA_SEED,
            "f_label": F_LABEL,
        },
        "train_losses": train_losses,
        "test_losses": test_losses,
    },
    checkpoint_path,
)
print(f"Finetuned model checkpoint saved to {checkpoint_path}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(train_losses, 'b-', label="train")
ax.plot(test_losses, 'r--', label="test")
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
