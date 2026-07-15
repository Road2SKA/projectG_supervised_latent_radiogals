"""
train_byol_classifier_many_times.py — multi-run RF/KNN/LR on a single BYOL run.

Trains RF, KNN, and LR classifiers on one BYOL run's features N times with
consecutive seeds (seed, seed+1, ...). Prints mean±std and top-1/top-5/top-all
ensemble scores per classifier in the same format as train_baseline_classifiers.py.

Usage:
    python scripts/train_byol_classifier_many_times.py \\
        --byol-run-dir outputs/byol_runs/<run> \\
        --label-set initial_pure \\
        --n-runs 10 \\
        --feature-type projections
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from suplat.utils.class_weights import compute_sample_weights


# ---------------------------------------------------------------------------
# Label-set definitions (same as train_byol_classifiers.py)
# ---------------------------------------------------------------------------

ALL_CLASS_NAMES = [
    'FRI', 'FRII', 'Hybrids', 'Spirals', 'Relaxed doubles',
    'C-curv', 'S-curv', 'Misalign', 'Wings', 'X-shaped',
    'Straight jets', 'Multi hotspots', 'Cont. jets', 'Banding',
    'One-sided', 'Restarted', 'Cluster', 'Merger', 'Diffuse', 'Unknown',
]

LABEL_SETS = {
    "classical":       [0, 1],
    "classical_pure":  [0, 1],
    "initial":         list(range(0, 5)),
    "initial_pure":    list(range(0, 5)),
    "morphology":      list(range(5, 16)),
    "morphology_pure": list(range(5, 16)),
    "environment":     list(range(16, 20)),
    "full":            list(range(0, 20)),
}

_PURE_SETS = {"classical_pure", "initial_pure", "morphology_pure"}


def _apply_label_set(labels_20: np.ndarray, label_set: str):
    n = len(labels_20)
    row_mask = np.ones(n, dtype=bool)

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(byol_run_dir: Path, feature_type: str, label_set: str):
    """Load and preprocess features + labels. Returns preprocessed arrays."""
    feat_dir = byol_run_dir / "data" / "byol"

    train_path = feat_dir / f"labelled_train_{feature_type}.npy"
    test_path  = feat_dir / f"test_{feature_type}.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path}")

    X_train_raw = np.load(train_path).astype(np.float32)
    X_test_raw  = np.load(test_path).astype(np.float32)

    # Locate data_splits/ — first try walking upward from run dir, then fall
    # back to the project root (parent of scripts/) which is always known.
    _project_root = Path(__file__).resolve().parent.parent
    _search = byol_run_dir.parent
    for _ in range(5):
        if (_search / "data_splits").is_dir():
            break
        _search = _search.parent
    else:
        _search = _project_root / "outputs"
    if not (_search / "data_splits").is_dir():
        _search = _project_root / "outputs"
    splits_dir = _search / "data_splits" / "42"

    # Labels — prefer per-run copy saved alongside projections
    run_lab_path = feat_dir / "labelled_train_labels.npy"
    lab_path     = splits_dir / "labelled_train_labels.npy"
    if run_lab_path.exists():
        y_train_full = np.load(run_lab_path)
    elif lab_path.exists():
        y_train_full = np.load(lab_path)
        if len(y_train_full) != len(X_train_raw):
            # Feature file covers full train set (labelled+unlabelled); reconstruct
            # labels for all train samples so the label-set filter can exclude
            # unlabelled ones (which have all-zero labels).
            train_idx_path    = splits_dir / "train_idx.npy"
            lab_idx_path      = splits_dir / "labelled_train_idx.npy"
            unlab_idx_path    = splits_dir / "unlabelled_train_idx.npy"
            unlab_labels_path = splits_dir / "unlabelled_train_labels.npy"
            if (train_idx_path.exists() and lab_idx_path.exists()
                    and unlab_idx_path.exists() and unlab_labels_path.exists()):
                train_idx    = np.load(train_idx_path)
                lab_idx      = np.load(lab_idx_path)
                unlab_idx    = np.load(unlab_idx_path)
                unlab_labels = np.load(unlab_labels_path)
                lab_labels   = y_train_full
                if len(X_train_raw) == len(train_idx):
                    idx_map = {}
                    for i, idx in enumerate(lab_idx):
                        idx_map[idx] = lab_labels[i]
                    for i, idx in enumerate(unlab_idx):
                        idx_map[idx] = unlab_labels[i]
                    y_train_full = np.stack([idx_map[idx] for idx in train_idx])
                else:
                    raise RuntimeError(
                        f"Cannot reconstruct labels: X_train has {len(X_train_raw)} rows "
                        f"but train_idx has {len(train_idx)}."
                    )
            else:
                raise RuntimeError(
                    f"Feature/label size mismatch ({len(X_train_raw)} vs {len(y_train_full)}) "
                    "and split index files are missing for reconstruction."
                )
    else:
        raise FileNotFoundError(f"Missing: {lab_path}")

    test_labels_path = splits_dir / "test_labels.npy"
    if not test_labels_path.exists():
        raise FileNotFoundError(f"Missing: {test_labels_path}")
    y_test_full = np.load(test_labels_path)

    y_train_raw, train_mask = _apply_label_set(y_train_full, label_set)
    y_test_raw,  test_mask  = _apply_label_set(y_test_full,  label_set)

    X_train = X_train_raw[train_mask]
    X_test  = X_test_raw[test_mask]

    if len(X_train) == 0:
        raise RuntimeError(f"No training samples after applying label_set='{label_set}'")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    is_multiclass = label_set in _PURE_SETS
    if is_multiclass:
        y_train = y_train_raw.argmax(axis=1)
        y_test  = y_test_raw.argmax(axis=1)
    else:
        y_train = y_train_raw
        y_test  = y_test_raw

    class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[label_set]]

    print(f"  train={len(X_train)}  test={len(X_test)}  classes={len(class_names)}")

    return X_train, X_test, y_train, y_test, y_train_full[train_mask], class_names, is_multiclass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _evaluate(y_true, y_pred, y_prob, class_names):
    n = len(class_names)
    if y_true.ndim == 1:
        y_true_bin = label_binarize(y_true, classes=list(range(n)))
        aucs = []
        for i in range(n):
            if len(np.unique(y_true_bin[:, i])) < 2:
                aucs.append(None)
            else:
                aucs.append(float(roc_auc_score(y_true_bin[:, i], y_prob[:, i])))
        auc_macro = float(np.nanmean([a for a in aucs if a is not None]))
        return {
            "f1_macro":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro":    auc_macro,
            "accuracy":     float(accuracy_score(y_true, y_pred)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    else:
        aucs = []
        for i in range(n):
            if len(np.unique(y_true[:, i])) < 2:
                aucs.append(None)
            else:
                aucs.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))
        return {
            "f1_macro":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro":    float(np.nanmean([a for a in aucs if a is not None])),
            "accuracy":     float(accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }


# ---------------------------------------------------------------------------
# Single run training
# ---------------------------------------------------------------------------

def train_one_run(run_i, seed, X_train, X_test, y_train, y_test,
                  y_train_full_masked, class_names, is_multiclass,
                  n_estimators, n_neighbors, lr_C,
                  out_dir: Path, force: bool,
                  class_weight_mode, class_weight_strength):
    """Train RF, KNN, LR for one seed. Returns dict of metrics + probs per clf."""
    run_dir = out_dir / f"run{run_i}"

    # KNN and LR are deterministic — only train them on run 1.
    clfs_to_train = ["rf"] if run_i > 1 else ["rf", "knn", "lr"]

    cached = {clf: (run_dir / f"{clf}.json").exists() for clf in clfs_to_train}
    if all(cached.values()) and not force:
        print(f"  Run {run_i}: all cached", flush=True)
        result = {}
        for clf in clfs_to_train:
            with open(run_dir / f"{clf}.json") as fh:
                d = json.load(fh)
            result[clf] = {k: d[k] for k in ("f1_macro", "auc_macro", "accuracy", "recall_macro")}
            prob_path = run_dir / f"{clf}_probs.npy"
            result[clf]["probs"] = np.load(prob_path) if prob_path.exists() else None
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run {run_i} (seed={seed}):", flush=True)

    y_full_cols = y_train_full_masked[:, :20] if y_train_full_masked.shape[1] >= 20 else y_train_full_masked
    sample_weights = compute_sample_weights(y_full_cols, class_weight_mode, class_weight_strength)

    rf_clf = (
        RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        if is_multiclass else
        MultiOutputClassifier(RandomForestClassifier(n_estimators=n_estimators,
                                                     random_state=seed, n_jobs=-1))
    )
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    lr_clf  = LogisticRegression(max_iter=1000, C=lr_C, random_state=seed)

    all_specs = [
        ("rf",  rf_clf,  sample_weights),
        ("knn", knn_clf, None),
        ("lr",  lr_clf,  sample_weights),
    ]
    specs = [(name, clf, sw) for name, clf, sw in all_specs if name in clfs_to_train]

    result = {}
    for clf_name, clf, sw in specs:
        if cached.get(clf_name) and not force:
            with open(run_dir / f"{clf_name}.json") as fh:
                d = json.load(fh)
            result[clf_name] = {k: d[k] for k in ("f1_macro", "auc_macro", "accuracy", "recall_macro")}
            prob_path = run_dir / f"{clf_name}_probs.npy"
            result[clf_name]["probs"] = np.load(prob_path) if prob_path.exists() else None
            print(f"    {clf_name.upper()}: cached", flush=True)
            continue

        if sw is not None:
            clf.fit(X_train, y_train, sample_weight=sw)
        else:
            clf.fit(X_train, y_train)

        if not is_multiclass and clf_name == "rf":
            y_pred = clf.predict(X_test)
            y_prob = np.stack(
                [est.predict_proba(X_test)[:, 1] for est in clf.estimators_], axis=1
            ).astype(np.float32)
        else:
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test).astype(np.float32)

        metrics = _evaluate(y_test, y_pred, y_prob, class_names)
        print(f"    {clf_name.upper()}: F1={metrics['f1_macro']:.4f}  "
              f"AUC={metrics['auc_macro']:.4f}  Acc={metrics['accuracy']:.4f}", flush=True)

        with open(run_dir / f"{clf_name}.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        np.save(run_dir / f"{clf_name}_probs.npy", y_prob)
        labels_path = run_dir / "test_labels.npy"
        if not labels_path.exists():
            np.save(labels_path, y_test)

        result[clf_name] = {**metrics, "probs": y_prob}

    return result


# ---------------------------------------------------------------------------
# Multi-run summary (mirrors train_baseline_classifiers.py format)
# ---------------------------------------------------------------------------

def _ensemble_metrics(ranked_probs, y_test, class_names, is_multiclass, k):
    avg_probs = np.mean(ranked_probs[:k], axis=0)
    if is_multiclass:
        y_pred = avg_probs.argmax(axis=1)
    else:
        y_pred = (avg_probs >= 0.5).astype(np.int64)
    return _evaluate(y_test, y_pred, avg_probs, class_names)


def print_multirun_summary(all_run_results, n_runs, label_set, y_test, class_names,
                           is_multiclass, out_dir: Path):
    summary = {}

    # RF: multi-run mean±std + ensemble
    rf_results = [r["rf"] for r in all_run_results if "rf" in r]
    if rf_results:
        f1s  = np.array([r["f1_macro"]    for r in rf_results])
        aucs = np.array([r["auc_macro"]   for r in rf_results])
        recs = np.array([r["recall_macro"] for r in rf_results])
        accs = np.array([r["accuracy"]    for r in rf_results])

        print(f"\n{'='*60}")
        print(f"RF — {n_runs} runs  ({label_set})")
        print(f"{'='*60}")
        print(f"  Macro F1    : {f1s.mean():.4f} ± {f1s.std():.4f}"
              f"  (min={f1s.min():.4f}, max={f1s.max():.4f})")
        print(f"  Macro AUC   : {aucs.mean():.4f} ± {aucs.std():.4f}"
              f"  (min={aucs.min():.4f}, max={aucs.max():.4f})")
        print(f"  Macro Recall: {recs.mean():.4f} ± {recs.std():.4f}"
              f"  (min={recs.min():.4f}, max={recs.max():.4f})")
        print(f"  Accuracy    : {accs.mean():.4f} ± {accs.std():.4f}"
              f"  (min={accs.min():.4f}, max={accs.max():.4f})")

        probs_list = [r["probs"] for r in rf_results if r.get("probs") is not None]
        if probs_list:
            ranked_idx   = np.argsort(f1s)[::-1]
            ranked_probs = [probs_list[i] for i in ranked_idx]
            print(f"\n  Ensemble scores (runs ranked by F1, threshold 0.5):")
            ensemble_summary = {}
            for k, tag in [(1, 'top-1'), (5, 'top-5'), (n_runs, 'top-all')]:
                k = min(k, len(ranked_probs))
                m = _ensemble_metrics(ranked_probs, y_test, class_names, is_multiclass, k=k)
                label = 'top-all' if k == len(ranked_probs) and tag == 'top-all' else tag
                print(f"  {label:<10}F1={m['f1_macro']:.4f}  "
                      f"AUC={m['auc_macro']:.4f}  "
                      f"Recall={m['recall_macro']:.4f}  "
                      f"Acc={m['accuracy']:.4f}")
                ensemble_summary[label] = m
            summary["rf"] = {"ensemble": ensemble_summary}
        summary.setdefault("rf", {}).update({
            "f1_macro_mean": float(f1s.mean()), "f1_macro_std": float(f1s.std()),
            "auc_macro_mean": float(aucs.mean()), "auc_macro_std": float(aucs.std()),
            "recall_macro_mean": float(recs.mean()), "recall_macro_std": float(recs.std()),
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        })

    # KNN and LR: deterministic, single result from run 1
    print(f"\n{'='*60}")
    print(f"KNN / LR — single run  ({label_set})")
    print(f"{'='*60}")
    for clf in ("knn", "lr"):
        r = next((run[clf] for run in all_run_results if clf in run), None)
        if r is None:
            continue
        print(f"  {clf.upper():<4}  F1={r['f1_macro']:.4f}  AUC={r['auc_macro']:.4f}"
              f"  Recall={r['recall_macro']:.4f}  Acc={r['accuracy']:.4f}")
        summary[clf] = {k: r[k] for k in ("f1_macro", "auc_macro", "recall_macro", "accuracy")}

    out_path = out_dir / "run_summary.json"
    with open(out_path, "w") as fh:
        json.dump({"n_runs": n_runs, "label_set": label_set, **summary}, fh, indent=2)
    print(f"\nSummary saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-run RF/KNN/LR on a single BYOL run's features."
    )
    parser.add_argument("--byol-run-dir", type=Path, required=True,
                        help="Path to a single BYOL run directory.")
    parser.add_argument("--label-set",    default="initial_pure",
                        choices=list(LABEL_SETS.keys()))
    parser.add_argument("--feature-type", default="projections",
                        choices=["projections", "encodings"])
    parser.add_argument("--n-runs",       type=int, default=10,
                        help="Number of training runs with consecutive seeds. Default: 10.")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Base random seed. Run i uses seed+(i-1). Default: 42.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-neighbors",  type=int, default=15)
    parser.add_argument("--lr-c",         type=float, default=1.0)
    parser.add_argument("--class-weight-mode", type=str, default=None,
                        choices=["score", "initial", "morphology", "environment", "classical", "all"])
    parser.add_argument("--class-weight-strength", type=float, default=0.0)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if cached results exist.")
    args = parser.parse_args()

    run_dir = args.byol_run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    out_dir = run_dir / "data" / "classifiers" / f"multirun_{args.label_set}_{args.feature_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"BYOL run : {run_dir.name}")
    print(f"Label set: {args.label_set}  Feature: {args.feature_type}  N runs: {args.n_runs}")
    print(f"Output   : {out_dir}\n")

    X_train, X_test, y_train, y_test, y_train_full_masked, class_names, is_multiclass = \
        load_data(run_dir, args.feature_type, args.label_set)

    all_run_results = []
    for run_i in range(1, args.n_runs + 1):
        seed_i = args.seed + (run_i - 1)
        result = train_one_run(
            run_i, seed_i, X_train, X_test, y_train, y_test,
            y_train_full_masked, class_names, is_multiclass,
            args.n_estimators, args.n_neighbors, args.lr_c,
            out_dir, args.force,
            args.class_weight_mode, args.class_weight_strength,
        )
        all_run_results.append(result)

    print_multirun_summary(
        all_run_results, args.n_runs, args.label_set,
        y_test, class_names, is_multiclass, out_dir,
    )


if __name__ == "__main__":
    main()
