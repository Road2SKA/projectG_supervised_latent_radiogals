"""
run_downstream_mlp_classifier.py — MLP classifier on frozen BYOL projections.

For each run directory matching --run-glob under --outputs-root:
  1. Load labelled train / test projections from data/byol/.
  2. Apply the requested label set (default: initial_pure) + pure-source filtering.
  3. Fit a StandardScaler on train features, then train an sklearn MLP.
  4. Evaluate on the test set (F1-macro, AUC-macro, Accuracy, Recall-macro).
  5. Save JSON + prediction .npy files to:
       data/classifiers/simple_downstream/{label_set}_{cw_tag}/mlp_{feature_type}.json

Results are saved in exactly the same format as run_downstream_classifiers.py so the
notebook can pick them up alongside LR/RF/KNN/GP without any extra changes.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from suplat.data.label_sets import ALL_CLASS_NAMES, LABEL_SETS, apply_label_set


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, class_names: list) -> dict:
    """Multiclass metrics (pure label sets only)."""
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
        "f1_per_class":       f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class":   recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "accuracy_per_class": [float(accuracy_score((y_true == c).astype(int),
                                                    (y_pred == c).astype(int)))
                               for c in range(n)],
        "auc_per_class":      aucs,
        "class_names":        class_names,
    }


# ── Per-run processing ────────────────────────────────────────────────────────

def process_run(run_dir: Path, feature_type: str, label_set: str,
                hidden_layer_sizes: tuple, max_iter: int, learning_rate_init: float,
                seed: int, force: bool,
                data_seed: int = None, cv_fold: int = None) -> dict:

    _eff_data_seed = data_seed if data_seed is not None else seed
    _seed_dir      = run_dir / f"data_seed_{_eff_data_seed}" / f"training_seed_{seed}"
    _fold_dir      = _seed_dir / f"cross_val_{cv_fold}" if cv_fold is not None else _seed_dir
    clf_dir        = _fold_dir / "data" / "classifiers" / "simple_downstream" / f"{label_set}_cwNone"
    out_path       = clf_dir / f"mlp_{feature_type}.json"

    if out_path.exists() and not force:
        print(f"  [{run_dir.name}] cached — skipping (use --force to rerun)", flush=True)
        with open(out_path) as fh:
            saved = json.load(fh)
        return {
            "name":         run_dir.name,
            "f1_macro":     saved.get("f1_macro",     float("nan")),
            "auc_macro":    saved.get("auc_macro",    float("nan")),
            "accuracy":     saved.get("accuracy",     float("nan")),
            "recall_macro": saved.get("recall_macro", float("nan")),
        }

    print(f"  [{run_dir.name}] processing...", flush=True)

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
    if not train_feat_path.exists():
        return dict(name=run_dir.name, error="missing_data", detail=str(train_feat_path))
    if not test_feat_path.exists():
        return dict(name=run_dir.name, error="missing_data", detail=str(test_feat_path))

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
            return dict(name=run_dir.name, error="label_size_mismatch",
                        detail=f"X_train {len(X_train_raw)} vs labels {len(y_train_full)}")
    else:
        return dict(name=run_dir.name, error="missing_data", detail=str(lab_path))

    test_labels_path = splits_dir / "test_labels.npy"
    if not test_labels_path.exists():
        return dict(name=run_dir.name, error="missing_data", detail=str(test_labels_path))
    y_test_full = np.load(test_labels_path)

    # ── Apply label set (pure → multiclass) ──────────────────────────────────
    y_train_raw, train_mask = apply_label_set(y_train_full, label_set)
    y_test_raw,  test_mask  = apply_label_set(y_test_full,  label_set)

    X_train = X_train_raw[train_mask]
    X_test  = X_test_raw[test_mask]

    if len(X_train) == 0:
        return dict(name=run_dir.name, error="empty_train",
                    detail=f"label_set='{label_set}' kept 0 rows")

    _base_ls    = label_set[:-11] if label_set.endswith("_individual") else label_set
    class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[_base_ls]]

    print(f"    train={len(X_train)}  test={len(X_test)}  classes={len(class_names)}",
          flush=True)

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # pure label set → multiclass integer targets
    y_train = y_train_raw.argmax(axis=1)
    y_test  = y_test_raw.argmax(axis=1)

    # ── Train MLP ─────────────────────────────────────────────────────────────
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(X_train, y_train)
    _loss_str = f"{mlp.best_loss_:.4f}" if mlp.best_loss_ is not None else "N/A"
    print(f"    MLP: converged in {mlp.n_iter_} iter(s)  "
          f"best val loss={_loss_str}", flush=True)

    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)   # (N_test, n_classes)

    metrics = evaluate_metrics(y_test, y_pred, y_prob, class_names)
    print(f"    MLP: F1={metrics['f1_macro']:.4f}  "
          f"AUC={metrics['auc_macro']:.4f}  Acc={metrics['accuracy']:.4f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    clf_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir":              str(run_dir),
        "feature_type":         feature_type,
        "label_set":            label_set,
        "hidden_layer_sizes":   list(hidden_layer_sizes),
        "max_iter":             max_iter,
        "learning_rate_init":   learning_rate_init,
        "n_iter":               int(mlp.n_iter_),
        "n_train":              int(len(X_train)),
        "n_test":               int(len(X_test)),
        **metrics,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    np.save(clf_dir / f"mlp_{feature_type}_test_preds.npy", y_pred)
    np.save(clf_dir / f"mlp_{feature_type}_test_probs.npy", y_prob)
    _lbl_path = clf_dir / f"{feature_type}_test_labels.npy"
    if not _lbl_path.exists():
        np.save(_lbl_path, y_test)

    return {
        "name":         run_dir.name,
        "f1_macro":     metrics["f1_macro"],
        "auc_macro":    metrics["auc_macro"],
        "accuracy":     metrics["accuracy"],
        "recall_macro": metrics["recall_macro"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP on frozen BYOL projections (initial_pure by default).")
    parser.add_argument("--outputs-root",    default="outputs/byol_runs",
                        help="Root containing run subdirs (default: outputs/byol_runs).")
    parser.add_argument("--run-glob",        default="pd128_*",
                        help="Glob pattern for run directories (default: pd128_*).")
    parser.add_argument("--feature-type",    default="projections",
                        choices=["projections", "encodings"])
    parser.add_argument("--label-set",       default="initial_pure",
                        help="Must be a *_pure label set (default: initial_pure).")
    parser.add_argument("--hidden-layers",   default="256,128",
                        help="Hidden layer sizes as comma-separated ints (default: 256,128).")
    parser.add_argument("--max-iter",        type=int, default=500,
                        help="Max training epochs (default: 500; early stopping active).")
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Adam initial learning rate (default: 1e-3).")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--data-seed",       type=int, default=None,
                        help="Seed for data_splits/<seed>/ (defaults to --seed).")
    parser.add_argument("--cv-fold",         type=int, default=None,
                        help="Cross-val fold index (0-based).")
    parser.add_argument("--force",           action="store_true",
                        help="Re-run even if output JSON already exists.")
    args = parser.parse_args()

    if not args.label_set.endswith("_pure"):
        parser.error("--label-set must be a *_pure label set (e.g. initial_pure).")

    hidden_layer_sizes = tuple(int(x) for x in args.hidden_layers.split(","))

    outputs_root = Path(args.outputs_root)
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs
                if re.search(r"_sw(?:cos|lin)?[\d.]+_f[\d.]+", rd.name)]
    if not run_dirs:
        print(f"No run directories found matching '{args.run_glob}' under {outputs_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs)} run directory/directories.\n")

    results, errors = [], []
    for rd in run_dirs:
        r = process_run(
            rd, args.feature_type, args.label_set,
            hidden_layer_sizes, args.max_iter, args.lr,
            args.seed, args.force,
            data_seed=args.data_seed, cv_fold=args.cv_fold,
        )
        if "error" in r:
            errors.append(r)
        else:
            results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        results.sort(key=lambda r: r["f1_macro"], reverse=True)
        print(f"\n{'Run':<55} {'F1':>6} {'AUC':>6} {'Acc':>6} {'Rec':>6}")
        print("-" * 75)
        for r in results:
            print(f"{r['name']:<55} {r['f1_macro']:6.4f} {r['auc_macro']:6.4f} "
                  f"{r['accuracy']:6.4f} {r['recall_macro']:6.4f}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e['name']}: {e['error']} — {e.get('detail','')}")


if __name__ == "__main__":
    main()
