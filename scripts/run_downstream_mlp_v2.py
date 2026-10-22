"""
run_downstream_mlp_v2.py — PyTorch MLP sweep on frozen BYOL projections.

For a given run directory, sweeps all combinations of:
  --hidden-sizes  e.g. "256x128 512x256 256"
  --weight-decays e.g. "0 1e-4 1e-3"

One unweighted CrossEntropyLoss run per config (class weighting is not the
bottleneck; see threshold_tuning_analysis.py). A loss-curve summary is printed
for the first config as a convergence sanity check.

NOTE: Does NOT overwrite mlp_projections.json (sklearn MLP, different script).

Output per config:
  {fold_dir}/data/downstream_classifiers/initial_pure_cwNone/
      mlp_v2_{arch_tag}_{wd_tag}_{feature_type}.json

Usage example:
    python scripts/run_downstream_mlp_v2.py \\
        --outputs-root outputs/byol_runs \\
        --run-glob     pd128_qext_v1_wd1e-3_lrconst_sw0.1_f1 \\
        --feature-type projections \\
        --seed         2 \\
        --data-seed    2 \\
        --cv-fold      0
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.data.label_sets import ALL_CLASS_NAMES, LABEL_SETS, apply_label_set
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, class_names: list) -> dict:
    n = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n)))
    aucs = []
    for i in range(n):
        if len(np.unique(y_true_bin[:, i])) < 2:
            aucs.append(None)
        else:
            aucs.append(float(roc_auc_score(y_true_bin[:, i], y_prob[:, i])))
    return {
        "f1_macro":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro":          float(np.nanmean([a for a in aucs if a is not None])),
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "recall_macro":       float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class":       f1_score(y_true, y_pred, average=None,
                                       labels=list(range(n)), zero_division=0).tolist(),
        "recall_per_class":   recall_score(y_true, y_pred, average=None,
                                           labels=list(range(n)), zero_division=0).tolist(),
        "accuracy_per_class": [float(accuracy_score((y_true == c).astype(int),
                                                    (y_pred == c).astype(int)))
                               for c in range(n)],
        "auc_per_class":      aucs,
        "class_names":        class_names,
    }


# ── MLP builder ───────────────────────────────────────────────────────────────

def build_mlp(in_dim: int, hidden: list, n_classes: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              hidden: list, weight_decay: float, epochs: int,
              seed: int, print_loss_curve: bool = False) -> tuple:
    """Return (model, train_loss_per_epoch_summary)."""
    torch.manual_seed(seed)
    n_classes = int(y_train.max()) + 1

    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).long()
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

    model = build_mlp(X_train.shape[1], hidden, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    model.train()
    loss_curve = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        loss_curve.append(epoch_loss / max(n_batches, 1))

    if print_loss_curve:
        print("    Loss curve (every 10 epochs):", flush=True)
        print("    Epoch budget chosen from loss curve; apply same budget to all cells.")
        for ep in range(0, epochs, 10):
            print(f"      ep {ep+1:3d}: {loss_curve[ep]:.4f}", flush=True)
        print(f"      ep {epochs:3d}: {loss_curve[-1]:.4f}", flush=True)

    return model, loss_curve


@torch.no_grad()
def predict_mlp(model: nn.Sequential, X: np.ndarray) -> tuple:
    model.eval()
    X_t   = torch.from_numpy(X).float()
    logits = model(X_t)
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = probs.argmax(axis=1)
    return preds, probs


# ── Per-run processing ────────────────────────────────────────────────────────

def process_run(run_dir: Path, feature_type: str, label_set: str,
                hidden_sizes_list: list, weight_decays: list,
                epochs: int, seed: int, data_seed: int, cv_fold: int,
                force: bool) -> None:

    _eff_data_seed = data_seed if data_seed is not None else seed
    _seed_dir      = run_dir / f"data_seed_{_eff_data_seed}" / f"training_seed_{seed}"
    _fold_dir      = _seed_dir / f"cross_val_{cv_fold}" if cv_fold is not None else _seed_dir
    clf_dir        = _fold_dir / "data" / "downstream_classifiers" / f"{label_set}_cwNone"

    print(f"\n[{run_dir.name}  fold={cv_fold}]", flush=True)

    # ── Locate data_splits/ ──────────────────────────────────────────────────
    _search = run_dir.parent
    for _ in range(5):
        if (_search / "data_splits").is_dir():
            break
        _search = _search.parent
    _splits_base = _search / "data_splits" / str(_eff_data_seed)
    splits_dir   = _splits_base / f"cross_val_{cv_fold}" if cv_fold is not None else _splits_base

    feat_dir = _fold_dir / "data" / "byol"
    _f_m     = re.search(r"_f([\d.]+)", run_dir.name)
    _f_tag   = ""
    if _f_m:
        _f_val = float(_f_m.group(1))
        _f_str = str(int(_f_val)) if _f_val == int(_f_val) else str(_f_val)
        _f_tag = f"_f{_f_str}"

    # ── Load features ────────────────────────────────────────────────────────
    train_feat_path = feat_dir / f"labelled_train_{feature_type}.npy"
    test_feat_path  = feat_dir / f"test_{feature_type}.npy"
    if not train_feat_path.exists() or not test_feat_path.exists():
        print(f"  Missing feature files — skipping.", flush=True)
        return

    X_train_raw = np.load(train_feat_path).astype(np.float32)
    X_test_raw  = np.load(test_feat_path).astype(np.float32)

    # ── Load labels ──────────────────────────────────────────────────────────
    run_lab_path = feat_dir / "labelled_train_labels.npy"
    lab_path     = splits_dir / f"labelled_train_labels{_f_tag}.npy"
    if run_lab_path.exists():
        y_train_full = np.load(run_lab_path)
    elif lab_path.exists():
        y_train_full = np.load(lab_path)
        if len(y_train_full) != len(X_train_raw):
            print(f"  Label/feature size mismatch — skipping.", flush=True)
            return
    else:
        print(f"  Missing label file — skipping.", flush=True)
        return

    test_labels_path = splits_dir / "test_labels.npy"
    if not test_labels_path.exists():
        print(f"  Missing test label file — skipping.", flush=True)
        return
    y_test_full = np.load(test_labels_path)

    # ── Apply label set ──────────────────────────────────────────────────────
    y_train_raw, train_mask = apply_label_set(y_train_full, label_set)
    y_test_raw,  test_mask  = apply_label_set(y_test_full,  label_set)

    X_train_filt = X_train_raw[train_mask]
    X_test_filt  = X_test_raw[test_mask]

    if len(X_train_filt) == 0:
        print(f"  Empty train set after label_set filter — skipping.", flush=True)
        return

    _base_ls    = label_set[:-11] if label_set.endswith("_individual") else label_set
    class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[_base_ls]]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_filt)
    X_test  = scaler.transform(X_test_filt)

    y_train = y_train_raw.argmax(axis=1)
    y_test  = y_test_raw.argmax(axis=1)

    print(f"  train={len(X_train)}  test={len(X_test)}  classes={len(class_names)}", flush=True)

    clf_dir.mkdir(parents=True, exist_ok=True)

    # ── Sweep configs ────────────────────────────────────────────────────────
    first_config = True
    for hidden_spec in hidden_sizes_list:
        # Parse "256x128" → [256, 128], "256" → [256]
        hidden = [int(x) for x in hidden_spec.split("x")]
        arch_tag = hidden_spec  # already in display format

        for wd in weight_decays:
            # Format weight decay tag
            if wd == 0 or wd == 0.0:
                wd_tag = "wd0"
            elif wd == 1e-4:
                wd_tag = "wd1e-4"
            elif wd == 1e-3:
                wd_tag = "wd1e-3"
            else:
                wd_tag = f"wd{wd:.0e}".replace("+", "")

            out_path = clf_dir / f"mlp_v2_{arch_tag}_{wd_tag}_{feature_type}.json"
            if out_path.exists() and not force:
                print(f"  mlp_v2_{arch_tag}_{wd_tag}: cached — skip", flush=True)
                continue

            print(f"  mlp_v2_{arch_tag}_{wd_tag}: training...", flush=True)
            model, loss_curve = train_mlp(
                X_train, y_train, hidden, float(wd), epochs, seed,
                print_loss_curve=first_config
            )
            first_config = False

            y_pred, y_prob = predict_mlp(model, X_test)
            metrics = evaluate_metrics(y_test, y_pred, y_prob, class_names)
            print(f"    F1={metrics['f1_macro']:.4f}  AUC={metrics['auc_macro']:.4f}  "
                  f"Acc={metrics['accuracy']:.4f}", flush=True)

            payload = {
                "run_dir":        str(run_dir),
                "feature_type":   feature_type,
                "label_set":      label_set,
                "arch":           hidden,
                "weight_decay":   float(wd),
                "epochs":         epochs,
                "epoch_budget_note": "fixed; chosen from loss curve",
                "n_train":        int(len(X_train)),
                "n_test":         int(len(X_test)),
                **metrics,
            }
            with open(out_path, "w") as fh:
                json.dump(payload, fh, indent=2)

            np.save(clf_dir / f"mlp_v2_{arch_tag}_{wd_tag}_{feature_type}_test_preds.npy", y_pred)
            np.save(clf_dir / f"mlp_v2_{arch_tag}_{wd_tag}_{feature_type}_test_probs.npy", y_prob)
            _lbl_path = clf_dir / f"{feature_type}_test_labels.npy"
            if not _lbl_path.exists():
                np.save(_lbl_path, y_test)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch MLP sweep on frozen BYOL projections.")
    parser.add_argument("--outputs-root",  default="outputs/byol_runs")
    parser.add_argument("--run-glob",      default="pd128_*",
                        help="Glob pattern for run directories.")
    parser.add_argument("--feature-type",  default="projections",
                        choices=["projections", "encodings"])
    parser.add_argument("--label-set",     default="initial_pure")
    parser.add_argument("--hidden-sizes",  nargs="+", default=["256x128", "512x256", "256"],
                        help="Hidden layer specs, e.g. 256x128 512x256 256.")
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-4, 1e-3],
                        help="Weight decay values for Adam.")
    parser.add_argument("--epochs",        type=int, default=100,
                        help="Fixed training epochs (default: 100).")
    parser.add_argument("--seed",          type=int, default=2)
    parser.add_argument("--data-seed",     type=int, default=None)
    parser.add_argument("--cv-fold",       type=int, default=None)
    parser.add_argument("--force",         action="store_true")
    args = parser.parse_args()

    if not args.label_set.endswith("_pure"):
        parser.error("--label-set must be a *_pure label set.")

    outputs_root = Path(args.outputs_root)
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs
                if re.search(r"_sw(?:cos|lin)?[\d.]+_f[\d.]+", rd.name)]
    if not run_dirs:
        print(f"No run directories matching '{args.run_glob}' under {outputs_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs)} run director(ies). Folds: {args.cv_fold}", flush=True)

    for rd in run_dirs:
        process_run(
            rd, args.feature_type, args.label_set,
            args.hidden_sizes, args.weight_decays,
            args.epochs, args.seed, args.data_seed, args.cv_fold,
            args.force,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
