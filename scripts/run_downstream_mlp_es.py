"""
run_downstream_mlp_es.py — MLP classifiers with early stopping (val_fraction=0.2).

Runs two classifier families and saves to NEW filenames so existing notebook
results (mlp_projections.json, mlp_v2_*_projections.json) are NOT overwritten:

  v1  sklearn MLPClassifier — same (256,128) arch as ref, but val_fraction=0.2
      Output: mlp_es02_{feature_type}.json
      NOTE: only supported for _pure label sets.

  v2  PyTorch MLP sweep — same 9 arch×wd configs as run_downstream_mlp_v2.py,
      but with early stopping (val_fraction=0.2, patience=20, max_epochs=300).
      Output: mlp_v2_{arch}_{wd}_es02_{feature_type}.json

Early stopping for v2: 20% of training data is held out as validation.  The
model with the lowest validation loss is restored at the end.  Training stops
when val loss has not improved for `patience` consecutive epochs.

val_fraction=0.2 for both families.

Label-set semantics:
  _pure label sets (e.g. initial_pure): single-label multiclass.
      Loss: CrossEntropyLoss;  prediction: softmax + argmax.
  All other label sets (no suffix, _binary, _individual): multilabel/multi-binary.
      Loss: BCEWithLogitsLoss;  prediction: sigmoid + 0.5 threshold.

Usage (single sw/fold, as called from sbatch array):
    python scripts/run_downstream_mlp_es.py \\
        --outputs-root outputs/byol_runs \\
        --run-glob     pd128_qext_v1_wd1e-3_lrconst_sw0.1_f1 \\
        --feature-type projections \\
        --label-set    initial_pure \\
        --seed         2 \\
        --data-seed    2 \\
        --cv-fold      0
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.data.label_sets import ALL_CLASS_NAMES, LABEL_SETS, apply_label_set, class_names_for


# ── Config ────────────────────────────────────────────────────────────────────

VAL_FRACTION  = 0.2
PATIENCE      = 20
MAX_EPOCHS_V2 = 300
BATCH_SIZE    = 256
LR            = 1e-3
ALPHA_V1      = 0.0001     # sklearn default L2 regularisation
N_ITER_NO_CHANGE_V1 = 15   # sklearn patience

V2_HIDDEN_SPECS  = ["256x128", "512x256", "256"]
V2_WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]


def _is_pure(label_set: str) -> bool:
    """Return True only for single-label multiclass sets (CrossEntropy/argmax)."""
    return label_set.endswith("_pure")


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_metrics(y_true, y_pred, y_prob, class_names, is_pure=True):
    n = len(class_names)

    if is_pure:
        # Single-label multiclass: y_true/y_pred are 1-D integer class labels.
        y_true_bin = label_binarize(y_true, classes=list(range(n)))
        if y_true_bin.ndim == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        aucs = []
        for i in range(n):
            if i >= y_prob.shape[1] or len(np.unique(y_true_bin[:, i])) < 2:
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
    else:
        # Multilabel/multi-binary: y_true/y_pred are 2-D multi-hot (N, n_classes).
        aucs = []
        for i in range(n):
            if len(np.unique(y_true[:, i])) < 2:
                aucs.append(None)
            else:
                aucs.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))
        return {
            "f1_macro":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro":          float(np.nanmean([a for a in aucs if a is not None])),
            "accuracy":           float(accuracy_score(y_true, y_pred)),
            "recall_macro":       float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_per_class":       f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "recall_per_class":   recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "accuracy_per_class": [(y_true[:, i] == y_pred[:, i]).mean()
                                   for i in range(n)],
            "auc_per_class":      aucs,
            "class_names":        class_names,
        }


# ── PyTorch helpers ───────────────────────────────────────────────────────────

def build_mlp(in_dim, hidden, n_classes):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def train_mlp_es(X_train_full, y_train_full, hidden, weight_decay, seed,
                 is_pure=True, val_fraction=VAL_FRACTION, patience=PATIENCE,
                 max_epochs=MAX_EPOCHS_V2):
    """
    Train with early stopping on a held-out val split.
    Returns (model_at_best_val, n_epochs_trained, best_val_epoch, best_val_loss).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    N = len(X_train_full)
    n_val = max(1, int(N * val_fraction))
    idx = rng.permutation(N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    if is_pure:
        n_classes = int(y_train_full.max()) + 1
        X_tr_t  = torch.from_numpy(X_tr).float()
        y_tr_t  = torch.from_numpy(y_tr).long()
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).long()
        criterion = nn.CrossEntropyLoss()
    else:
        n_classes = y_train_full.shape[1]
        X_tr_t  = torch.from_numpy(X_tr).float()
        y_tr_t  = torch.from_numpy(y_tr).float()
        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).float()
        criterion = nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = build_mlp(X_train_full.shape[1], hidden, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    best_val_loss  = float("inf")
    best_epoch     = 0
    best_state     = copy.deepcopy(model.state_dict())
    no_improve     = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, epoch, best_epoch, best_val_loss


@torch.no_grad()
def predict_mlp(model, X, is_pure=True):
    model.eval()
    logits = model(torch.from_numpy(X).float())
    if is_pure:
        probs = torch.softmax(logits, dim=1).numpy()
        return probs.argmax(axis=1), probs
    else:
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(np.float32)
        return preds, probs


# ── Data loading (shared between v1 and v2) ───────────────────────────────────

def load_data(run_dir, feature_type, label_set, data_seed, seed, cv_fold):
    """Returns (X_train, X_test, y_train, y_test, class_names, clf_dir, is_pure)
    or None on error.

    For _pure label sets y_train/y_test are 1-D integer class labels.
    For all other label sets they are 2-D float multi-hot arrays.
    """
    _eff_ds   = data_seed if data_seed is not None else seed
    _fold_dir = (run_dir / f"data_seed_{_eff_ds}" / f"training_seed_{seed}"
                 / f"cross_val_{cv_fold}") if cv_fold is not None else (
                 run_dir / f"data_seed_{_eff_ds}" / f"training_seed_{seed}")
    clf_dir   = (_fold_dir / "data" / "downstream_classifiers"
                 / f"{label_set}_cwNone")

    # Find data_splits/
    _search = run_dir.parent
    for _ in range(5):
        if (_search / "data_splits").is_dir():
            break
        _search = _search.parent
    splits_dir = ((_search / "data_splits" / str(_eff_ds) / f"cross_val_{cv_fold}")
                  if cv_fold is not None else
                  (_search / "data_splits" / str(_eff_ds)))

    feat_dir = _fold_dir / "data" / "byol"
    _f_m     = re.search(r"_f([\d.]+)", run_dir.name)
    _f_tag   = ""
    if _f_m:
        _fv = float(_f_m.group(1))
        _f_tag = f"_f{int(_fv) if _fv == int(_fv) else _fv}"

    for p in [feat_dir / f"labelled_train_{feature_type}.npy",
              feat_dir / f"test_{feature_type}.npy"]:
        if not p.exists():
            print(f"  Missing: {p} — skip", flush=True)
            return None

    X_train_raw = np.load(feat_dir / f"labelled_train_{feature_type}.npy").astype(np.float32)
    X_test_raw  = np.load(feat_dir / f"test_{feature_type}.npy").astype(np.float32)

    run_lab = feat_dir / "labelled_train_labels.npy"
    lab_p   = splits_dir / f"labelled_train_labels{_f_tag}.npy"
    if run_lab.exists():
        y_train_full = np.load(run_lab)
    elif lab_p.exists():
        y_train_full = np.load(lab_p)
    else:
        print(f"  Missing labels — skip", flush=True)
        return None
    if len(y_train_full) != len(X_train_raw):
        print(f"  Label/feature mismatch — skip", flush=True)
        return None

    tl_p = splits_dir / "test_labels.npy"
    if not tl_p.exists():
        print(f"  Missing test labels — skip", flush=True)
        return None
    y_test_full = np.load(tl_p)

    y_tr_raw, tr_mask = apply_label_set(y_train_full, label_set)
    y_te_raw, te_mask = apply_label_set(y_test_full,  label_set)
    if tr_mask.sum() == 0:
        print(f"  Empty train after label filter — skip", flush=True)
        return None

    class_names = class_names_for(label_set)
    pure = _is_pure(label_set)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw[tr_mask])
    X_test  = scaler.transform(X_test_raw[te_mask])

    if pure:
        y_train = y_tr_raw.argmax(axis=1)
        y_test  = y_te_raw.argmax(axis=1)
    else:
        y_train = y_tr_raw.astype(np.float32)
        y_test  = y_te_raw.astype(np.float32)

    clf_dir.mkdir(parents=True, exist_ok=True)
    print(f"  train={len(X_train)}  test={len(X_test)}  classes={len(class_names)}"
          f"  mode={'multiclass' if pure else 'multilabel'}",
          flush=True)
    return X_train, X_test, y_train, y_test, class_names, clf_dir, pure


# ── v1: sklearn MLP, val_fraction=0.2 ─────────────────────────────────────────

def run_v1(data, run_dir, feature_type, label_set, seed, force):
    X_train, X_test, y_train, y_test, class_names, clf_dir, is_pure = data

    if not is_pure:
        print(f"  mlp_es02: skipped — sklearn v1 only supports _pure label sets", flush=True)
        return

    out = clf_dir / f"mlp_es02_{feature_type}.json"
    if out.exists() and not force:
        print(f"  mlp_es02: cached — skip", flush=True)
        return

    print(f"  mlp_es02 (sklearn, val=0.2): training...", flush=True)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=ALPHA_V1,
        learning_rate_init=LR,
        max_iter=500,
        early_stopping=True,
        validation_fraction=VAL_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE_V1,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(X_train, y_train)
    print(f"    stopped at epoch {mlp.n_iter_}  best_val_score={mlp.best_validation_score_:.4f}",
          flush=True)

    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)
    metrics = evaluate_metrics(y_test, y_pred, y_prob, class_names, is_pure=True)
    print(f"    F1={metrics['f1_macro']:.4f}  AUC={metrics['auc_macro']:.4f}", flush=True)

    payload = {
        "run_dir": str(run_dir), "feature_type": feature_type,
        "label_set": label_set,
        "hidden_layer_sizes": [256, 128], "alpha": ALPHA_V1,
        "val_fraction": VAL_FRACTION, "n_iter_no_change": N_ITER_NO_CHANGE_V1,
        "n_iter": int(mlp.n_iter_), "best_val_score": float(mlp.best_validation_score_),
        "early_stopping": True,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        **metrics,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    np.save(clf_dir / f"mlp_es02_{feature_type}_test_preds.npy", y_pred)
    np.save(clf_dir / f"mlp_es02_{feature_type}_test_probs.npy", y_prob)


# ── v2: PyTorch MLP with early stopping ───────────────────────────────────────

def _wd_tag(wd):
    if wd == 0.0:    return "wd0"
    if wd == 1e-4:   return "wd1e-4"
    if wd == 1e-3:   return "wd1e-3"
    return f"wd{wd:.0e}".replace("+", "")


def run_v2_es(data, run_dir, feature_type, label_set, seed, force):
    X_train, X_test, y_train, y_test, class_names, clf_dir, is_pure = data

    for hidden_spec in V2_HIDDEN_SPECS:
        hidden   = [int(x) for x in hidden_spec.split("x")]
        arch_tag = hidden_spec

        for wd in V2_WEIGHT_DECAYS:
            wt = _wd_tag(wd)
            out = clf_dir / f"mlp_v2_{arch_tag}_{wt}_es02_{feature_type}.json"
            if out.exists() and not force:
                print(f"  mlp_v2_{arch_tag}_{wt}_es02: cached — skip", flush=True)
                continue

            print(f"  mlp_v2_{arch_tag}_{wt}_es02: training...", flush=True)
            model, n_ep, best_ep, best_vl = train_mlp_es(
                X_train, y_train, hidden, float(wd), seed, is_pure=is_pure)
            print(f"    stopped ep={n_ep}  best_ep={best_ep}  best_val_loss={best_vl:.4f}",
                  flush=True)

            y_pred, y_prob = predict_mlp(model, X_test, is_pure=is_pure)
            metrics = evaluate_metrics(y_test, y_pred, y_prob, class_names, is_pure=is_pure)
            print(f"    F1={metrics['f1_macro']:.4f}  AUC={metrics['auc_macro']:.4f}",
                  flush=True)

            payload = {
                "run_dir": str(run_dir), "feature_type": feature_type,
                "label_set": label_set,
                "arch": hidden, "weight_decay": float(wd),
                "val_fraction": VAL_FRACTION, "patience": PATIENCE,
                "max_epochs": MAX_EPOCHS_V2,
                "epochs_trained": n_ep, "best_val_epoch": best_ep,
                "best_val_loss": best_vl,
                "early_stopping": True,
                "epoch_budget_note": "early stopping; val_fraction=0.2; patience=20",
                "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                **metrics,
            }
            with open(out, "w") as fh:
                json.dump(payload, fh, indent=2)
            np.save(clf_dir / f"mlp_v2_{arch_tag}_{wt}_es02_{feature_type}_test_preds.npy",
                    y_pred)
            np.save(clf_dir / f"mlp_v2_{arch_tag}_{wt}_es02_{feature_type}_test_probs.npy",
                    y_prob)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs/byol_runs")
    parser.add_argument("--run-glob",     default="pd128_*")
    parser.add_argument("--feature-type", default="projections",
                        choices=["projections", "encodings"])
    parser.add_argument("--label-set",    default="initial_pure")
    parser.add_argument("--seed",         type=int, default=2)
    parser.add_argument("--data-seed",    type=int, default=None)
    parser.add_argument("--cv-fold",      type=int, default=None)
    parser.add_argument("--force",        action="store_true")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs
                if re.search(r"_sw(?:cos|lin)?[\d.]+_f[\d.]+", rd.name)]
    if not run_dirs:
        print(f"No run dirs matching '{args.run_glob}' under {outputs_root}", file=sys.stderr)
        sys.exit(1)

    for rd in run_dirs:
        print(f"\n[{rd.name}  fold={args.cv_fold}]", flush=True)
        data = load_data(rd, args.feature_type, args.label_set,
                         args.data_seed, args.seed, args.cv_fold)
        if data is None:
            continue
        run_v1(data, rd, args.feature_type, args.label_set, args.seed, args.force)
        run_v2_es(data, rd, args.feature_type, args.label_set, args.seed, args.force)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
