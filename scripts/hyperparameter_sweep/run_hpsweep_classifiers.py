"""
run_hpsweep_classifiers.py — LR/RF/KNN classifiers for one hyperparameter sweep run.

Locates labels via the checkpoint's data_seed (not hardcoded to 42).
Handles f_label-suffixed label files (e.g. labelled_train_labels_f1.npy).

For each of LR, RF, KNN:
  1. Load labelled_train_projections + test_projections from data/byol/
  2. Load labels from data_splits/<data_seed>/ (respecting f_label suffix)
  3. Apply initial_pure filtering (sources with exactly one label in {FRI,FRII,Hybrid,Spiral,RD})
  4. StandardScaler → fit classifier → evaluate
  5. Save JSON to data/classifiers/lr_initial_pure_projections.json etc.

Output files (one per classifier):
    <run_dir>/data/classifiers/lr_initial_pure_projections.json
    <run_dir>/data/classifiers/rf_initial_pure_projections.json
    <run_dir>/data/classifiers/knn_initial_pure_projections.json

Usage:
    python scripts/run_hpsweep_classifiers.py --run-dir <path> [--force]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize


# ---------------------------------------------------------------------------
# Initial-pure label filtering
# 5 classes: FRI(0), FRII(1), Hybrids(2), Spirals(3), Relaxed doubles(4)
# Pure = exactly one label set in the initial group (cols 0-4)
# ---------------------------------------------------------------------------
CLASS_NAMES = ['FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles']

def apply_initial_pure(y_full: np.ndarray):
    """Return (y_1d, row_mask) for sources with exactly one initial-group label."""
    initial = y_full[:, 0:5]
    row_mask = initial.sum(axis=1) == 1
    y_sub = np.argmax(initial[row_mask], axis=1)
    return y_sub, row_mask


def evaluate(y_true, y_pred, y_prob) -> dict:
    n = len(CLASS_NAMES)
    y_true_bin = label_binarize(y_true, classes=list(range(n)))
    aucs = []
    for i in range(n):
        if len(np.unique(y_true_bin[:, i])) < 2:
            aucs.append(None)
        else:
            aucs.append(float(roc_auc_score(y_true_bin[:, i], y_prob[:, i])))
    return {
        "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro":     float(np.nanmean([a for a in aucs if a is not None])),
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "recall_macro":  float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class":  f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "auc_per_class": aucs,
        "class_names":   CLASS_NAMES,
    }


def main():
    ap = argparse.ArgumentParser(description="Run classifiers on one hpsweep BYOL run")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output files already exist")
    ap.add_argument("--seed", type=int, default=42, help="Classifier RNG seed (default: 42)")
    ap.add_argument("--n-estimators", type=int, default=200, help="RF trees (default: 200)")
    ap.add_argument("--n-neighbors", type=int, default=15, help="KNN k (default: 15)")
    ap.add_argument("--lr-c", type=float, default=1.0, help="LR regularisation C (default: 1.0)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    clf_dir = run_dir / "data" / "classifiers"
    clf_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "lr":  clf_dir / "lr_initial_pure_projections.json",
        "rf":  clf_dir / "rf_initial_pure_projections.json",
        "knn": clf_dir / "knn_initial_pure_projections.json",
    }

    if all(p.exists() for p in out_paths.values()) and not args.force:
        print(f"[{run_dir.name}] all classifier results cached — skipping (use --force to rerun)")
        for name, path in out_paths.items():
            with open(path) as fh:
                r = json.load(fh)
            print(f"  {name}: f1={r['f1_macro']:.4f}  auc={r['auc_macro']:.4f}  acc={r['accuracy']:.4f}")
        return

    # ── Locate data_seed from checkpoint ──────────────────────────────────────
    ckpt_path = run_dir / "byol_model_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data_seed = int(ckpt["config"]["data_seed"])
    f_label   = float(ckpt["config"].get("f_label", 1.0))
    f_str     = str(f_label).rstrip('0').rstrip('.') if '.' in str(f_label) else str(int(f_label))

    print(f"\n[{run_dir.name}]  data_seed={data_seed}  f_label={f_label}")

    # ── Locate splits directory ───────────────────────────────────────────────
    splits_dir = run_dir.parent.parent / "data_splits" / str(data_seed)
    if not splits_dir.exists():
        print(f"ERROR: splits dir not found: {splits_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load features ─────────────────────────────────────────────────────────
    byol_dir = run_dir / "data" / "byol"
    X_train = np.load(byol_dir / "labelled_train_projections.npy").astype(np.float32)
    X_test  = np.load(byol_dir / "test_projections.npy").astype(np.float32)
    print(f"  Features: train={X_train.shape}  test={X_test.shape}")

    # ── Load labels (handles f_label suffix) ─────────────────────────────────
    def _load_labels(name):
        # Try f_label-suffixed version first
        stem, ext = name.rsplit('.', 1)
        p_f = splits_dir / f"{stem}_f{f_str}.{ext}"
        if p_f.exists():
            return np.load(p_f)
        p = splits_dir / name
        if p.exists():
            return np.load(p)
        raise FileNotFoundError(f"Labels not found at {p_f} or {p}")

    y_train_full = _load_labels("labelled_train_labels.npy")
    y_test_full  = _load_labels("test_labels.npy")

    # ── Apply initial_pure filter ─────────────────────────────────────────────
    y_train, train_mask = apply_initial_pure(y_train_full)
    y_test,  test_mask  = apply_initial_pure(y_test_full)

    X_train_f = X_train[train_mask]
    X_test_f  = X_test[test_mask]

    print(f"  initial_pure: train={len(X_train_f)}  test={len(X_test_f)}  classes={len(CLASS_NAMES)}")

    if len(X_train_f) == 0:
        print("ERROR: no training samples after initial_pure filter", file=sys.stderr)
        sys.exit(1)

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s  = scaler.transform(X_test_f)

    # ── Fit and evaluate classifiers ──────────────────────────────────────────
    classifiers = {
        "lr":  LogisticRegression(C=args.lr_c, max_iter=1000, random_state=args.seed),
        "rf":  RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed,
                                      n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=args.n_neighbors, n_jobs=-1),
    }

    for name, clf in classifiers.items():
        out_path = out_paths[name]
        if out_path.exists() and not args.force:
            print(f"  {name}: cached — skipping")
            continue
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)
        metrics = evaluate(y_test, y_pred, y_prob)
        with open(out_path, 'w') as fh:
            json.dump(metrics, fh, indent=2)
        print(f"  {name}: f1={metrics['f1_macro']:.4f}  auc={metrics['auc_macro']:.4f}"
              f"  acc={metrics['accuracy']:.4f}  recall={metrics['recall_macro']:.4f}")

    print(f"  Saved to {clf_dir}/")


if __name__ == '__main__':
    main()
