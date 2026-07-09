"""
Floor baselines for BYOL probe AUC table.

Computes macro AUC for:
  1. Pixel probe          — raw 89×89 pixels flattened (7921 features)
  2. Random EfficientNet  — 1280-dim features from an untrained EfficientNet-B0

These set the lower bound: any BYOL encoder below these is worse than nothing.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score


SPLITS_DIR  = Path("outputs/data_splits/42")
IMAGES_PATH = Path("data/preprocessed/lotss/images_filtered.npy")


def probe(X_tr, X_te, y_tr, y_te, name: str):
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    aucs, f1s = [], []
    for k in range(y_tr.shape[1]):
        if y_tr[:, k].sum() == 0 or y_te[:, k].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr[:, k])
        p = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te[:, k], p))
        f1s.append(f1_score(y_te[:, k], (p > 0.5).astype(int), zero_division=0))

    macro_auc = float(np.mean(aucs))
    macro_f1  = float(np.mean(f1s))
    print(f"  {name:30s}  AUC={macro_auc:.4f}  F1={macro_f1:.4f}  (n_labels={len(aucs)})")
    return macro_auc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/preprocessed/lotss"))
    args = parser.parse_args()

    images_path = args.data_dir / "images_filtered.npy"

    print("Loading data...")
    images = np.load(images_path).astype(np.float32) / 255.0
    print(f"  images: {images.shape}")

    train_idx = np.load(SPLITS_DIR / "labelled_train_idx_f1.npy")
    test_idx  = np.load(SPLITS_DIR / "test_idx.npy")
    y_tr      = np.load(SPLITS_DIR / "labelled_train_labels_f1.npy")
    y_te      = np.load(SPLITS_DIR / "test_labels.npy")

    tr_imgs = images[train_idx]   # (N_tr, 89, 89)
    te_imgs = images[test_idx]    # (N_te, 89, 89)

    print(f"\n  Train: {tr_imgs.shape[0]} samples  |  Test: {te_imgs.shape[0]} samples")
    print(f"  Labels: {y_tr.shape[1]} columns\n")

    # ── 1. Pixel probe ─────────────────────────────────────────────────────────
    print("Running floor baselines:")
    X_tr_px = tr_imgs.reshape(len(tr_imgs), -1)
    X_te_px = te_imgs.reshape(len(te_imgs), -1)
    probe(X_tr_px, X_te_px, y_tr, y_te, "Pixel probe (7921d)")

    # ── 2. Random EfficientNet ─────────────────────────────────────────────────
    try:
        import torch
        import torchvision.models as models

        print("  Loading random-init EfficientNet-B0...")
        model = models.efficientnet_b0(weights=None)
        model.classifier = torch.nn.Identity()  # drop classification head → 1280d
        model.eval()

        def extract(imgs):
            # imgs: (N, 89, 89) float32 in [0,1]
            # EfficientNet expects (N, 3, H, W)
            t = torch.from_numpy(imgs[:, None, :, :]).repeat(1, 3, 1, 1)
            with torch.no_grad():
                features = []
                bs = 256
                for i in range(0, len(t), bs):
                    features.append(model(t[i:i+bs]).numpy())
            return np.concatenate(features, axis=0)

        X_tr_rnd = extract(tr_imgs)
        X_te_rnd = extract(te_imgs)
        probe(X_tr_rnd, X_te_rnd, y_tr, y_te, "Random EfficientNet-B0 (1280d)")

    except ImportError:
        print("  [SKIP] torch/torchvision not available for random EfficientNet baseline")

    print("\nDone.")


if __name__ == "__main__":
    main()
