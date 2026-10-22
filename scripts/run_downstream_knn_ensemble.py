"""
run_downstream_knn_ensemble.py — Three kNN variants on frozen BYOL projections.

Variants:
  A. Distance-weighted kNN (k=10, euclidean)
     → knn_dist_k10_{feature_type}.json
  B. k-averaging ensemble (k ∈ {5,10,15,20,30}, uniform weights, averaged probs)
     → knn_kavg_{feature_type}.json
  C. Gaussian-kernel classifier (cosine distance, bandwidth γ tuned by 3-fold CV)
     → knn_rbf_{feature_type}.json

Inputs are L2-normalised 128-d projections. StandardScaler is applied on train
features before all classifiers (same as existing LR/MLP scripts).

For variant C, the RBF kernel uses cosine distance:
    cos_dist(x, y) = 1 - x·y   (since inputs are L2-normalised)
    K(x, x_i; γ) = exp(-γ * cos_dist(x, x_i)²)
    p(c | x) ∝ Σ_{i: y_i = c} K(x, x_i; γ)
Bandwidth γ is tuned entirely within the training fold via 3-fold CV on macro F1.

Does NOT modify any existing classifier output files.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.data.label_sets import ALL_CLASS_NAMES, LABEL_SETS, apply_label_set


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


# ── Variant C: Gaussian-kernel classifier ────────────────────────────────────

def _rbf_predict_proba(X_test: np.ndarray, X_train: np.ndarray,
                       y_train: np.ndarray, gamma: float, n_classes: int) -> np.ndarray:
    """
    Compute class probabilities using RBF kernel on cosine distance.
    X_train and X_test should already be StandardScaler-normalised (not L2-norm).
    cos_dist approximated as 1 - dot product after re-normalising rows.
    """
    # Re-normalise rows for cosine distance (scaler changes the norms)
    X_tr_n  = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-12)
    X_te_n  = X_test  / (np.linalg.norm(X_test,  axis=1, keepdims=True) + 1e-12)

    # Cosine similarity: (N_test, N_train)
    sim = X_te_n @ X_tr_n.T
    sim = np.clip(sim, -1.0, 1.0)
    cos_dist = 1.0 - sim          # (N_test, N_train)

    # RBF kernel weights: (N_test, N_train)
    K = np.exp(-gamma * cos_dist ** 2)

    # Accumulate per class
    probs = np.zeros((len(X_test), n_classes), dtype=np.float64)
    for c in range(n_classes):
        mask_c = (y_train == c)
        if mask_c.any():
            probs[:, c] = K[:, mask_c].sum(axis=1)

    # Normalise to probabilities
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (probs / row_sums).astype(np.float32)


def run_knn_rbf(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                class_names: list, gamma_grid: list) -> tuple:
    """Returns (metrics_dict, gamma, cv_f1_per_gamma)."""
    n_classes = len(class_names)

    # Tune γ by 3-fold stratified CV on training set only
    cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cv_f1s = []
    for gamma in gamma_grid:
        fold_f1s = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            Xtr, Xva = X_train[tr_idx], X_train[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            probs_va  = _rbf_predict_proba(Xva, Xtr, ytr, gamma, n_classes)
            preds_va  = probs_va.argmax(axis=1)
            fold_f1s.append(f1_score(yva, preds_va, average="macro", zero_division=0))
        cv_f1s.append(float(np.mean(fold_f1s)))

    best_idx   = int(np.argmax(cv_f1s))
    best_gamma = gamma_grid[best_idx]

    # Evaluate on test set with best γ
    probs_test = _rbf_predict_proba(X_test, X_train, y_train, best_gamma, n_classes)
    preds_test = probs_test.argmax(axis=1)

    metrics = evaluate_metrics(y_test, preds_test, probs_test, class_names)
    metrics["gamma"]            = float(best_gamma)
    metrics["gamma_grid"]       = [float(g) for g in gamma_grid]
    metrics["cv_f1_per_gamma"]  = cv_f1s

    return metrics, probs_test, preds_test


# ── Per-run processing ────────────────────────────────────────────────────────

def process_run(run_dir: Path, feature_type: str, label_set: str,
                seed: int, data_seed: int, cv_fold: int, force: bool) -> None:

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
        print(f"  Empty train set — skipping.", flush=True)
        return

    _base_ls    = label_set[:-11] if label_set.endswith("_individual") else label_set
    class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[_base_ls]]
    n_classes   = len(class_names)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_filt)
    X_test  = scaler.transform(X_test_filt)

    y_train = y_train_raw.argmax(axis=1)
    y_test  = y_test_raw.argmax(axis=1)

    print(f"  train={len(X_train)}  test={len(X_test)}  classes={n_classes}", flush=True)
    clf_dir.mkdir(parents=True, exist_ok=True)

    _lbl_path = clf_dir / f"{feature_type}_test_labels.npy"
    if not _lbl_path.exists():
        np.save(_lbl_path, y_test)

    # ── Variant A: distance-weighted kNN ────────────────────────────────────
    out_a = clf_dir / f"knn_dist_k10_{feature_type}.json"
    if out_a.exists() and not force:
        print(f"  knn_dist_k10: cached — skip", flush=True)
    else:
        print(f"  knn_dist_k10: fitting...", flush=True)
        knn_a = KNeighborsClassifier(n_neighbors=10, weights="distance", metric="euclidean")
        knn_a.fit(X_train, y_train)
        prob_a = knn_a.predict_proba(X_test).astype(np.float32)
        pred_a = prob_a.argmax(axis=1)
        metrics_a = evaluate_metrics(y_test, pred_a, prob_a, class_names)
        payload_a = {"run_dir": str(run_dir), "feature_type": feature_type,
                     "label_set": label_set, "n_neighbors": 10, "weights": "distance",
                     "n_train": int(len(X_train)), "n_test": int(len(X_test)), **metrics_a}
        with open(out_a, "w") as fh:
            json.dump(payload_a, fh, indent=2)
        np.save(clf_dir / f"knn_dist_k10_{feature_type}_test_preds.npy", pred_a)
        np.save(clf_dir / f"knn_dist_k10_{feature_type}_test_probs.npy", prob_a)
        print(f"    F1={metrics_a['f1_macro']:.4f}  AUC={metrics_a['auc_macro']:.4f}", flush=True)

    # ── Variant B: k-averaging ensemble ─────────────────────────────────────
    out_b = clf_dir / f"knn_kavg_{feature_type}.json"
    if out_b.exists() and not force:
        print(f"  knn_kavg: cached — skip", flush=True)
    else:
        print(f"  knn_kavg: fitting ensemble...", flush=True)
        k_values  = [5, 10, 15, 20, 30]
        all_probs = []
        per_k     = {}
        for k in k_values:
            knn_k  = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
            knn_k.fit(X_train, y_train)
            prob_k = knn_k.predict_proba(X_test).astype(np.float32)
            all_probs.append(prob_k)
            pred_k = prob_k.argmax(axis=1)
            f1_k   = f1_score(y_test, pred_k, average="macro", zero_division=0)
            per_k[str(k)] = {"f1_macro": float(f1_k)}
        prob_b = np.mean(all_probs, axis=0).astype(np.float32)
        pred_b = prob_b.argmax(axis=1)
        metrics_b = evaluate_metrics(y_test, pred_b, prob_b, class_names)
        payload_b = {"run_dir": str(run_dir), "feature_type": feature_type,
                     "label_set": label_set, "k_values": k_values,
                     "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                     "per_k": per_k, **metrics_b}
        with open(out_b, "w") as fh:
            json.dump(payload_b, fh, indent=2)
        np.save(clf_dir / f"knn_kavg_{feature_type}_test_preds.npy", pred_b)
        np.save(clf_dir / f"knn_kavg_{feature_type}_test_probs.npy", prob_b)
        per_k_str = ["{:.3f}".format(per_k[str(k)]["f1_macro"]) for k in k_values]
        print(f"    F1={metrics_b['f1_macro']:.4f}  AUC={metrics_b['auc_macro']:.4f}  "
              f"per-k={per_k_str}", flush=True)

    # ── Variant C: Gaussian-kernel RBF ──────────────────────────────────────
    out_c = clf_dir / f"knn_rbf_{feature_type}.json"
    if out_c.exists() and not force:
        print(f"  knn_rbf: cached — skip", flush=True)
    else:
        print(f"  knn_rbf: tuning gamma and fitting...", flush=True)
        gamma_grid = np.logspace(-1, 2, 10).tolist()
        metrics_c, prob_c, pred_c = run_knn_rbf(
            X_train, y_train, X_test, y_test, class_names, gamma_grid)
        payload_c = {"run_dir": str(run_dir), "feature_type": feature_type,
                     "label_set": label_set,
                     "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                     **metrics_c}
        with open(out_c, "w") as fh:
            json.dump(payload_c, fh, indent=2)
        np.save(clf_dir / f"knn_rbf_{feature_type}_test_preds.npy", pred_c)
        np.save(clf_dir / f"knn_rbf_{feature_type}_test_probs.npy", prob_c)
        print(f"    F1={metrics_c['f1_macro']:.4f}  AUC={metrics_c['auc_macro']:.4f}  "
              f"gamma={metrics_c['gamma']:.3f}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="kNN ensemble classifiers on frozen BYOL projections.")
    parser.add_argument("--outputs-root", default="outputs/byol_runs")
    parser.add_argument("--run-glob",     default="pd128_*",
                        help="Glob pattern for run directories.")
    parser.add_argument("--feature-type", default="projections",
                        choices=["projections", "encodings"])
    parser.add_argument("--label-set",    default="initial_pure")
    parser.add_argument("--seed",         type=int, default=2)
    parser.add_argument("--data-seed",    type=int, default=None)
    parser.add_argument("--cv-fold",      type=int, default=None)
    parser.add_argument("--force",        action="store_true")
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
    print(f"Found {len(run_dirs)} run director(ies).", flush=True)

    for rd in run_dirs:
        process_run(rd, args.feature_type, args.label_set,
                    args.seed, args.data_seed, args.cv_fold, args.force)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
