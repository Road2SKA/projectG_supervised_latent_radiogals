"""
Train sklearn + MLP classifiers on pre-extracted BYOL feature vectors.

Usage:
    python train_sklearn_classifiers.py \\
        --run_dir outputs/run_enb0_... \\
        --feature_type projections \\
        --label_set classical_pure

Feature files read from <run_dir>/data/:
    labelled_train_{feature_type}.npy   (N_lab, D)
    test_{feature_type}.npy             (N_test, D)
    labelled_train_idx.npy              indices into train_labels
    train_labels.npy                    (N_train, 20) — full 20-dim multi-hot
    test_labels.npy                     (N_test, 20)

Results saved to:
    <run_dir>/sklearn_classifiers/<label_set>_<feature_type>_<timestamp>/
        config.json
        results.json
        mlp_best.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ── Label set definitions (mirrors train_byol.py / train_classification_baselines.py) ──

ALL_CLASS_NAMES = [
    'FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles',
    'C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
    'Straight jets', 'Multi hotspots', 'Cont. jets', 'Banding',
    'One-sided', 'Restarted', 'Cluster', 'Merger', 'Diffuse', 'Unknown',
]

DERIVED_CLASS_NAMES = [
    'Pure hybrid',        # col2 & ~col0 & ~col1
    'FR hybrid',          # col2 & (col0 | col1)
    'Curved FRI',         # col0 & (col5 | col6)
    'Curved FRII',        # col1 & (col5 | col6)
    'Straight+multi-HS',  # col10 & col11
]

LABEL_SETS = {
    "classical":       [0, 1],
    "classical_pure":  [0, 1],
    "initial":         list(range(0, 5)),
    "initial_pure":    list(range(0, 5)),
    "morphology":      list(range(5, 16)),
    "morphology_pure": list(range(5, 16)),
    "environment":     list(range(16, 20)),
    "derived":         None,
    "full":            list(range(0, 20)),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_derived(y: np.ndarray) -> np.ndarray:
    c = lambda i: y[:, i].astype(bool)
    return np.stack([
        ( c(2) & ~c(0) & ~c(1)).astype(np.int64),
        ( c(2) &  (c(0) | c(1))).astype(np.int64),
        ( c(0) &  (c(5) | c(6))).astype(np.int64),
        ( c(1) &  (c(5) | c(6))).astype(np.int64),
        (c(10) &   c(11)).astype(np.int64),
    ], axis=1)


def apply_label_set(labels_20: np.ndarray, label_set: str):
    """
    Apply column selection and pure-source row filtering.

    Returns (labels_sub, row_mask). For non-pure label sets row_mask is all-True.
    """
    n        = len(labels_20)
    row_mask = np.ones(n, dtype=bool)

    if label_set == "derived":
        return _make_derived(labels_20), row_mask

    if label_set == "classical_pure":
        fri_frii = labels_20[:, 0:2]
        rest     = labels_20[:, 2:5]
        row_mask = (fri_frii.sum(axis=1) == 1) & (rest.sum(axis=1) == 0)

    elif label_set == "initial_pure":
        initial  = labels_20[:, 0:5]
        row_mask = initial.sum(axis=1) == 1

    elif label_set == "morphology_pure":
        morph    = labels_20[:, 5:16]
        row_mask = morph.sum(axis=1) == 1

    cols       = LABEL_SETS[label_set]
    labels_sub = labels_20[row_mask][:, cols]
    return labels_sub.astype(np.int64), row_mask


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, class_names: list) -> dict:
    n    = len(class_names)
    aucs = []
    for i in range(n):
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(None)
        else:
            aucs.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))

    return {
        "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro":     float(np.nanmean([a for a in aucs if a is not None])),
        "accuracy":      float(accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))),
        "f1_per_class":  f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "auc_per_class": aucs,
        "class_names":   class_names,
    }


# ── MLP ───────────────────────────────────────────────────────────────────────

class MultiLabelMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int,
                 hidden=(512, 256), dropout: float = 0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body      = nn.Sequential(*layers)
        self.head      = nn.Linear(prev, n_classes * 2)
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.body(x)).view(-1, self.n_classes, 2)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              n_classes: int, device: torch.device, args) -> nn.Module:
    """Train MLP with early stopping on a held-out val slice."""
    rng       = np.random.default_rng(args.seed)
    n_val     = max(1, int(len(X_train) * args.mlp_val_frac))
    val_idx   = rng.choice(len(X_train), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)

    X_tr = torch.from_numpy(X_train[train_idx]).float().to(device)
    y_tr = torch.from_numpy(y_train[train_idx]).long().to(device)
    X_va = torch.from_numpy(X_train[val_idx]).float().to(device)
    y_va = torch.from_numpy(y_train[val_idx]).long().to(device)

    # Per-class positive weights for imbalanced labels
    pos = y_tr.float().sum(dim=0).clamp(min=1)
    neg = len(y_tr) - pos
    pw  = (neg / pos).clamp(max=20)

    hidden = tuple(int(h) for h in args.mlp_hidden.split(","))
    model  = MultiLabelMLP(X_tr.shape[1], n_classes, hidden=hidden,
                           dropout=args.mlp_dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.mlp_lr,
                              weight_decay=args.mlp_wd)

    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(epoch))
        for i in range(0, len(X_tr), args.mlp_batch):
            idx    = perm[i:i + args.mlp_batch]
            logits = model(X_tr[idx])                         # (B, C, 2)
            loss   = sum(
                F.cross_entropy(logits[:, c, :], y_tr[idx][:, c],
                                weight=torch.tensor([1.0, pw[c].item()], device=device))
                for c in range(n_classes)
            ) / n_classes
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_v = model(X_va)
            val_loss = sum(
                F.cross_entropy(logits_v[:, c, :], y_va[:, c])
                for c in range(n_classes)
            ).item() / n_classes

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= args.mlp_patience:
                print(f"    Early stop at epoch {epoch}  (best val loss {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model


def predict_mlp(model: nn.Module, X: np.ndarray,
                device: torch.device, batch: int = 512):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(device)
            p  = F.softmax(model(xb), dim=-1)[:, :, 1].cpu().numpy()
            all_probs.append(p)
    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train sklearn + MLP classifiers on BYOL feature vectors."
    )
    p.add_argument("--run_dir",      required=True, type=Path,
                   help="BYOL run directory (must contain a data/ subfolder)")
    p.add_argument("--feature_type", default="projections",
                   choices=["projections", "encodings"],
                   help="Feature vectors to use (default: projections)")
    p.add_argument("--label_set",    default="classical_pure",
                   choices=list(LABEL_SETS.keys()),
                   help="Classification scheme (default: classical_pure)")
    p.add_argument("--seed",         type=int, default=42)

    # KNN
    p.add_argument("--n_neighbors",  type=int, default=15)

    # Random Forest
    p.add_argument("--n_estimators", type=int, default=200)

    # MLP
    p.add_argument("--mlp_hidden",   default="512,256",
                   help="Comma-separated hidden layer sizes (default: 512,256)")
    p.add_argument("--mlp_epochs",   type=int,   default=100)
    p.add_argument("--mlp_patience", type=int,   default=15)
    p.add_argument("--mlp_lr",       type=float, default=1e-3)
    p.add_argument("--mlp_wd",       type=float, default=1e-4)
    p.add_argument("--mlp_dropout",  type=float, default=0.3)
    p.add_argument("--mlp_batch",    type=int,   default=256)
    p.add_argument("--mlp_val_frac", type=float, default=0.15)

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = (args.run_dir / "sklearn_classifiers"
               / f"{args.label_set}_{args.feature_type}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Load features ─────────────────────────────────────────────────────────
    data_dir = args.run_dir / "data"
    ft       = args.feature_type

    train_feat_path = data_dir / f"labelled_train_{ft}.npy"
    test_feat_path  = data_dir / f"test_{ft}.npy"

    if not train_feat_path.exists():
        sys.exit(f"ERROR: {train_feat_path} not found. "
                 f"Run train_byol.py first, or check --feature_type.")
    if not test_feat_path.exists():
        sys.exit(f"ERROR: {test_feat_path} not found.")

    X_train_raw = np.load(train_feat_path).astype(np.float32)
    X_test_raw  = np.load(test_feat_path).astype(np.float32)

    # labelled_train_idx maps the labelled train features → rows in train_labels
    all_train_labels = np.load(data_dir / "train_labels.npy")        # (N_train, 20)
    lab_idx          = np.load(data_dir / "labelled_train_idx.npy")  # (N_lab,)
    y_train_full     = all_train_labels[lab_idx]                      # (N_lab, 20)
    y_test_full      = np.load(data_dir / "test_labels.npy")          # (N_test, 20)

    print(f"Train features : {X_train_raw.shape}")
    print(f"Test  features : {X_test_raw.shape}")

    # ── Label set + pure filtering ────────────────────────────────────────────
    y_train, train_mask = apply_label_set(y_train_full, args.label_set)
    y_test,  test_mask  = apply_label_set(y_test_full,  args.label_set)

    X_train = X_train_raw[train_mask]
    X_test  = X_test_raw[test_mask]

    if args.label_set == "derived":
        class_names = DERIVED_CLASS_NAMES
    else:
        class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[args.label_set]]
    n_classes = len(class_names)

    print(f"\nLabel set : {args.label_set}  ({n_classes} classes: {class_names})")
    print(f"Train     : {len(X_train)} samples")
    print(f"Test      : {len(X_test)} samples")

    # ── Feature normalisation ─────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── sklearn classifiers ───────────────────────────────────────────────────
    all_results = {}

    classifiers = [
        ("knn", MultiOutputClassifier(
            KNeighborsClassifier(n_neighbors=args.n_neighbors,
                                 metric="euclidean", n_jobs=-1))),
        ("random_forest", MultiOutputClassifier(
            RandomForestClassifier(n_estimators=args.n_estimators,
                                   random_state=args.seed, n_jobs=-1))),
        ("logistic_regression", MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=args.seed, n_jobs=-1))),
    ]

    for name, clf in classifiers:
        print(f"\nFitting {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = np.stack(
            [est.predict_proba(X_test)[:, 1] for est in clf.estimators_], axis=1
        )
        res = evaluate_metrics(y_test, y_pred, y_prob, class_names)
        all_results[name] = res
        print(f"  Macro F1:  {res['f1_macro']:.4f}")
        print(f"  Macro AUC: {res['auc_macro']:.4f}")
        print(f"  Accuracy:  {res['accuracy']:.4f}")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("\nFitting mlp...")
    mlp_model = train_mlp(X_train, y_train, n_classes, device, args)
    y_pred_mlp, y_prob_mlp = predict_mlp(mlp_model, X_test, device)
    res = evaluate_metrics(y_test, y_pred_mlp, y_prob_mlp, class_names)
    all_results["mlp"] = res
    print(f"  Macro F1:  {res['f1_macro']:.4f}")
    print(f"  Macro AUC: {res['auc_macro']:.4f}")
    print(f"  Accuracy:  {res['accuracy']:.4f}")

    torch.save({
        "state_dict":  mlp_model.state_dict(),
        "in_dim":      X_train.shape[1],
        "n_classes":   n_classes,
        "hidden":      args.mlp_hidden,
        "class_names": class_names,
        "label_set":   args.label_set,
    }, out_dir / "mlp_best.pt")

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "run_dir":      str(args.run_dir),
        "feature_type": args.feature_type,
        "label_set":    args.label_set,
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "class_names":  class_names,
        "classifiers":  all_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"\n{'Classifier':<22}  {'Macro F1':>9}  {'Macro AUC':>10}  {'Accuracy':>9}")
    print(f"  {'-'*55}")
    for name, res in all_results.items():
        print(f"  {name:<22}  {res['f1_macro']:>9.4f}  {res['auc_macro']:>10.4f}  {res['accuracy']:>9.4f}")


if __name__ == "__main__":
    main()
