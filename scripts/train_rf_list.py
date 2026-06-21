"""
train_rf_list.py — Sklearn classifier sweep over BYOL run directories.

For each run directory matching --run-glob under --outputs-root:
  1. Load labelled train and test feature vectors (projections or encodings).
  2. Load labels and apply the requested label set + pure-source filtering.
  3. Fit a StandardScaler on train, then train RF, KNN, and LR classifiers.
  4. Evaluate each on the test set (F1-macro, AUC-macro, accuracy).
  5. Save a JSON result per classifier to {run_dir}/data/classifiers/{clf}_{label_set}_{feature_type}.json.

After all runs, print a summary table sorted by RF F1-macro descending.
"""

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize


# ---------------------------------------------------------------------------
# Label-set definitions (copied from train_sklearn_classifiers.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers (copied from train_sklearn_classifiers.py)
# ---------------------------------------------------------------------------

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
    """Apply column selection and pure-source row filtering.

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
    n = len(class_names)

    if y_true.ndim == 1:
        # Multiclass (pure label sets): y_true and y_pred are 1-D argmax labels.
        y_true_bin = label_binarize(y_true, classes=list(range(n)))
        aucs = []
        for i in range(n):
            if len(np.unique(y_true_bin[:, i])) < 2:
                aucs.append(None)
            else:
                aucs.append(float(roc_auc_score(y_true_bin[:, i], y_prob[:, i])))
        auc_macro = float(np.nanmean([a for a in aucs if a is not None]))
        return {
            "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro":     auc_macro,
            "accuracy":      float(accuracy_score(y_true, y_pred)),
            "recall_macro":  float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_per_class":  f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "auc_per_class": aucs,
            "class_names":   class_names,
        }
    else:
        # Multi-label: y_true and y_pred are 2-D multi-hot arrays.
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
            "recall_macro":  float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_per_class":  f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "auc_per_class": aucs,
            "class_names":   class_names,
        }


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _fit_and_eval(clf, X_train, y_train, X_test, y_test, class_names,
                  label_set, is_multi_output=False):
    """Fit clf and return evaluate_metrics result dict."""
    clf.fit(X_train, y_train)
    if is_multi_output:
        y_pred = clf.predict(X_test)
        y_prob = np.stack(
            [est.predict_proba(X_test)[:, 1] for est in clf.estimators_], axis=1
        )
    else:
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
    return evaluate_metrics(y_test, y_pred, y_prob, class_names)


def process_run(run_dir: Path, feature_type: str, label_set: str,
                n_estimators: int, n_neighbors: int, lr_C: float,
                seed: int, force: bool):
    """Train RF, KNN, and LR for one run directory.

    Returns a result dict with keys rf/knn/lr (each with f1_macro, auc_macro, accuracy),
    or a failure dict (with key 'error' and 'detail').
    """
    clf_dir  = run_dir / "data" / "classifiers"
    rf_path  = clf_dir / f"rf_{label_set}_{feature_type}.json"
    knn_path = clf_dir / f"knn_{label_set}_{feature_type}.json"
    lr_path  = clf_dir / f"lr_{label_set}_{feature_type}.json"

    all_cached = rf_path.exists() and knn_path.exists() and lr_path.exists()
    if all_cached and not force:
        print(f"  [{run_dir.name}] skipping (all cached — use --force to rerun)", flush=True)
        out = dict(name=run_dir.name)
        for key, path in [("rf", rf_path), ("knn", knn_path), ("lr", lr_path)]:
            with open(path) as fh:
                saved = json.load(fh)
            out[key] = {
                "f1_macro":     saved.get("f1_macro",     float("nan")),
                "auc_macro":    saved.get("auc_macro",    float("nan")),
                "accuracy":     saved.get("accuracy",     float("nan")),
                "recall_macro": saved.get("recall_macro", float("nan")),
            }
        return out

    print(f"  [{run_dir.name}] processing...", flush=True)
    data_dir = run_dir / "data"

    # ── Load features ────────────────────────────────────────────────────────
    train_feat_path = data_dir / f"labelled_train_{feature_type}.npy"
    test_feat_path  = data_dir / f"test_{feature_type}.npy"

    if not train_feat_path.exists():
        return dict(name=run_dir.name, error="missing_data",
                    detail=f"Missing: {train_feat_path}")
    if not test_feat_path.exists():
        return dict(name=run_dir.name, error="missing_data",
                    detail=f"Missing: {test_feat_path}")

    X_train_raw = np.load(train_feat_path).astype(np.float32)
    X_test_raw  = np.load(test_feat_path).astype(np.float32)

    # ── Load labels ──────────────────────────────────────────────────────────
    lab_labels_path = data_dir / "labelled_train_labels.npy"
    if lab_labels_path.exists():
        y_train_full = np.load(lab_labels_path)
    else:
        train_labels_path = data_dir / "train_labels.npy"
        lab_idx_path      = data_dir / "labelled_train_idx.npy"
        if not train_labels_path.exists():
            return dict(name=run_dir.name, error="missing_data",
                        detail=f"Missing: {train_labels_path}")
        if not lab_idx_path.exists():
            return dict(name=run_dir.name, error="missing_data",
                        detail=f"Missing: {lab_idx_path}")
        all_train_labels = np.load(train_labels_path)
        lab_idx          = np.load(lab_idx_path)
        if len(all_train_labels) == len(X_train_raw):
            y_train_full = all_train_labels
        else:
            y_train_full = all_train_labels[lab_idx]

    test_labels_path = data_dir / "test_labels.npy"
    if not test_labels_path.exists():
        return dict(name=run_dir.name, error="missing_data",
                    detail=f"Missing: {test_labels_path}")
    y_test_full = np.load(test_labels_path)

    # ── Apply label set + pure filtering ─────────────────────────────────────
    y_train_raw, train_mask = apply_label_set(y_train_full, label_set)
    y_test_raw,  test_mask  = apply_label_set(y_test_full,  label_set)

    X_train = X_train_raw[train_mask]
    X_test  = X_test_raw[test_mask]

    if len(X_train) == 0:
        return dict(name=run_dir.name, error="empty_train",
                    detail=f"No labelled train samples after applying label_set='{label_set}' "
                           f"(raw N_lab={len(X_train_raw)}, pure mask kept 0 rows)")

    if label_set == "derived":
        class_names = DERIVED_CLASS_NAMES
    else:
        class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[label_set]]

    print(f"    train={len(X_train)}  test={len(X_test)}  "
          f"classes={len(class_names)}", flush=True)

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Multiclass vs multi-label ─────────────────────────────────────────────
    _pure_sets    = {"classical_pure", "initial_pure", "morphology_pure"}
    is_multiclass = label_set in _pure_sets
    if is_multiclass:
        y_train = y_train_raw.argmax(axis=1)
        y_test  = y_test_raw.argmax(axis=1)
    else:
        y_train = y_train_raw
        y_test  = y_test_raw

    clf_dir.mkdir(parents=True, exist_ok=True)
    out = dict(name=run_dir.name)

    # ── Classifiers ───────────────────────────────────────────────────────────
    _specs = [
        ("rf",  rf_path,
         RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
         if is_multiclass else
         MultiOutputClassifier(RandomForestClassifier(n_estimators=n_estimators,
                                                      random_state=seed, n_jobs=-1))),
        ("knn", knn_path,
         KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)),
        ("lr",  lr_path,
         LogisticRegression(max_iter=1000, C=lr_C, random_state=seed)),
    ]

    for clf_name, path, clf in _specs:
        if path.exists() and not force:
            print(f"    {clf_name.upper()}: cached", flush=True)
            with open(path) as fh:
                saved = json.load(fh)
            out[clf_name] = {
                "f1_macro":     saved.get("f1_macro",     float("nan")),
                "auc_macro":    saved.get("auc_macro",    float("nan")),
                "accuracy":     saved.get("accuracy",     float("nan")),
                "recall_macro": saved.get("recall_macro", float("nan")),
            }
            continue

        is_mo = not is_multiclass and clf_name == "rf"
        metrics = _fit_and_eval(clf, X_train, y_train, X_test, y_test,
                                class_names, label_set, is_multi_output=is_mo)
        print(f"    {clf_name.upper()}: F1={metrics['f1_macro']:.4f}  "
              f"AUC={metrics['auc_macro']:.4f}  Acc={metrics['accuracy']:.4f}", flush=True)

        payload = {
            "run_dir":      str(run_dir),
            "feature_type": feature_type,
            "label_set":    label_set,
            "n_train":      int(len(X_train)),
            "n_test":       int(len(X_test)),
            "class_names":  class_names,
            **metrics,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

        out[clf_name] = {
            "f1_macro":     metrics["f1_macro"],
            "auc_macro":    metrics["auc_macro"],
            "accuracy":     metrics["accuracy"],
            "recall_macro": metrics["recall_macro"],
        }

    return out


# ---------------------------------------------------------------------------
# Multiprocessing worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _worker(args):
    run_dir, feature_type, label_set, n_estimators, n_neighbors, lr_C, seed, force = args
    try:
        return process_run(run_dir, feature_type, label_set,
                           n_estimators, n_neighbors, lr_C, seed, force)
    except Exception as exc:
        import traceback
        print(f"  ERROR in {run_dir.name}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return dict(name=run_dir.name, error="other", detail=str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sklearn classifier sweep (RF / KNN / LR) over BYOL run directories."
    )
    parser.add_argument("--outputs-root", default="outputs",
                        help="Root directory containing run subdirectories (default: outputs).")
    parser.add_argument("--run-glob",     default="run_enb0_*",
                        help="Glob pattern for run directories (default: run_enb0_*).")
    parser.add_argument("--feature-type", default="projections",
                        choices=["projections", "encodings"],
                        help="Feature vectors to use (default: projections).")
    parser.add_argument("--label-set",    default="classical_pure",
                        choices=list(LABEL_SETS.keys()),
                        help="Classification scheme (default: classical_pure).")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of RF trees (default: 200).")
    parser.add_argument("--n-neighbors",  type=int, default=15,
                        help="KNN neighbours (default: 15).")
    parser.add_argument("--lr-c",         type=float, default=1.0,
                        help="Logistic regression inverse regularisation strength (default: 1.0).")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--force",        action="store_true",
                        help="Re-run even if result already saved.")
    parser.add_argument("--workers",      type=int, default=1,
                        help="Number of parallel worker processes (default: 1).")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)

    # ── Discover run directories ───────────────────────────────────────────────
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs if re.search(r'_f[\d.]+_sw[\d.]+', rd.name)]
    if not run_dirs:
        print(f"No run directories found matching '{args.run_glob}' under {outputs_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs)} run directories.\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    worker_args = [
        (rd, args.feature_type, args.label_set, args.n_estimators,
         args.n_neighbors, args.lr_c, args.seed, args.force)
        for rd in run_dirs
    ]

    n_workers = min(args.workers, len(worker_args))
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            all_results = pool.map(_worker, worker_args)
    else:
        all_results = [_worker(a) for a in worker_args]

    # ── Split successes / failures ────────────────────────────────────────────
    results  = [r for r in all_results if r is not None and "error" not in r]
    failures = [r for r in all_results if r is not None and "error" in r]

    # ── Ranked summary ────────────────────────────────────────────────────────
    if results:
        results.sort(
            key=lambda r: r.get("rf", {}).get("f1_macro", -1.0),
            reverse=True,
        )

        def _fmt(r, clf, metric, width):
            v = r.get(clf, {}).get(metric, float("nan"))
            return f"{v:>{width}.4f}" if v == v else " " * (width - 3) + "N/A"

        w   = max(len(r["name"]) for r in results)
        hdr = (f"{'Rank':>4}  {'RF F1':>7}  {'KNN F1':>7}  {'LR F1':>7}  "
               f"{'RF AUC':>7}  {'KNN AUC':>8}  {'LR AUC':>7}  {'Run':<{w}}")
        sep = "=" * len(hdr)
        print(f"\n{sep}")
        print(f"RF / KNN / LR ({args.label_set} / {args.feature_type}) — ranked by RF F1-macro")
        print(sep)
        print(hdr)
        print("-" * len(hdr))
        for i, r in enumerate(results, 1):
            print(
                f"{i:>4}  "
                f"{_fmt(r, 'rf',  'f1_macro',  7)}  "
                f"{_fmt(r, 'knn', 'f1_macro',  7)}  "
                f"{_fmt(r, 'lr',  'f1_macro',  7)}  "
                f"{_fmt(r, 'rf',  'auc_macro',  7)}  "
                f"{_fmt(r, 'knn', 'auc_macro',  8)}  "
                f"{_fmt(r, 'lr',  'auc_macro',  7)}  "
                f"{r['name']:<{w}}"
            )
        print(sep)

    # ── Failure summary ───────────────────────────────────────────────────────
    if failures:
        by_type = {}
        for f in failures:
            by_type.setdefault(f["error"], []).append(f)
        print(f"\nFailures ({len(failures)}):")
        for err_type, group in by_type.items():
            label = {
                "missing_data": "Missing data files",
                "empty_train":  "No labelled train samples after label-set filtering",
                "other":        "Other error",
            }.get(err_type, err_type)
            print(f"\n  {label} ({len(group)}):")
            for f in group:
                print(f"    {f['name']}")
                print(f"      {f['detail']}")


if __name__ == "__main__":
    main()
