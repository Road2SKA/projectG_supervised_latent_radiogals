"""
train_rf_list.py — Sklearn classifier sweep over BYOL run directories.

For each run directory matching --run-glob under --outputs-root:
  1. Load labelled train and test feature vectors (projections or encodings).
  2. Load labels and apply the requested label set + pure-source filtering.
  3. Fit a StandardScaler on train, then train RF, KNN, and LR classifiers.
  4. Evaluate each on the test set (F1-macro, AUC-macro, accuracy).
  5. Save a JSON result per classifier to {run_dir}/data/classifiers/{clf}_{label_set}_{feature_type}.json.

After all runs, print a summary table sorted by best F1-macro across classifiers descending.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from suplat.utils.class_weights import compute_class_weights, compute_sample_weights


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
    "full_pure":       list(range(0, 20)),
    # _binary variants: same columns as base set; evaluated element-wise (not exact-match)
    "classical_binary":   [0, 1],
    "initial_binary":     list(range(0, 5)),
    "morphology_binary":  list(range(5, 16)),
    "environment_binary": list(range(16, 20)),
    "full_binary":        list(range(0, 20)),
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

    _base      = label_set[:-7] if label_set.endswith('_binary') else label_set
    cols       = LABEL_SETS[_base]
    labels_sub = labels_20[row_mask][:, cols]
    return labels_sub.astype(np.int64), row_mask


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, class_names: list,
                     is_binary: bool = False) -> dict:
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
        _acc = float((y_true == y_pred).mean()) if is_binary else float(accuracy_score(y_true, y_pred))
        return {
            "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "auc_macro":     float(np.nanmean([a for a in aucs if a is not None])),
            "accuracy":      _acc,
            "recall_macro":  float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_per_class":  f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            "auc_per_class": aucs,
            "class_names":   class_names,
        }


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _fit_and_eval(clf, X_train, y_train, X_test, y_test, class_names,
                  label_set, is_multi_output=False, sample_weight=None):
    """Fit clf and return evaluate_metrics result dict."""
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
    if is_multi_output:
        y_pred = clf.predict(X_test)
        y_prob = np.stack(
            [est.predict_proba(X_test)[:, 1] for est in clf.estimators_], axis=1
        )
    else:
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
    return evaluate_metrics(y_test, y_pred, y_prob, class_names,
                            is_binary=label_set.endswith('_binary')), y_pred, y_prob



def _do_ip_eval(out: dict, clf_dir: Path, feature_type: str, run_name: str,
                y_test_full, test_mask):
    """Evaluate already-fitted classifiers on the initial_pure (5-class) subset.

    Loads per-classifier test_probs.npy from clf_dir, applies the initial_pure
    mask (rows with exactly one active class among the first 5), renormalises
    the first-5 probability columns, and computes multiclass metrics.
    When y_test_full / test_mask are None (all-cached path), the probs are
    loaded but the mask must already be embedded in the saved label file.
    """
    _ip_class_names = [ALL_CLASS_NAMES[i] for i in range(5)]
    _n_ip = None

    for _cn in ("rf", "knn", "lr"):
        _prob_path = clf_dir / f"{_cn}_{feature_type}_test_probs.npy"
        if not _prob_path.exists():
            continue
        _probs = np.load(_prob_path)   # (N_test, 20) for full label set

        # Determine initial_pure mask
        if y_test_full is not None and test_mask is not None:
            _y20 = y_test_full[test_mask]
            _ip_mask = _y20[:, :5].sum(axis=1) == 1
            _y_ip    = _y20[_ip_mask][:, :5].argmax(axis=1)
        else:
            # Fall back to stored test labels if available
            _lbl_path = clf_dir / f"{feature_type}_test_labels.npy"
            if not _lbl_path.exists():
                continue
            _y20     = np.load(_lbl_path)
            _ip_mask = _y20[:, :5].sum(axis=1) == 1
            _y_ip    = _y20[_ip_mask][:, :5].argmax(axis=1)

        if _ip_mask.sum() == 0:
            continue

        _p5 = _probs[_ip_mask, :5]
        _row_sum = _p5.sum(axis=1, keepdims=True).clip(min=1e-9)
        _p5_norm = _p5 / _row_sum
        _yp_ip   = _p5_norm.argmax(axis=1)

        _ip_metrics = evaluate_metrics(_y_ip, _yp_ip, _p5_norm, _ip_class_names)
        if _n_ip is None:
            _n_ip = int(_ip_mask.sum())
            print(f"    ── initial_pure subset ({_n_ip} samples) ──", flush=True)
        print(f"    {_cn.upper()} (ip): F1={_ip_metrics['f1_macro']:.4f}  "
              f"AUC={_ip_metrics['auc_macro']:.4f}  "
              f"Acc={_ip_metrics['accuracy']:.4f}", flush=True)
        out[_cn]["ip_f1_macro"]     = _ip_metrics["f1_macro"]
        out[_cn]["ip_auc_macro"]    = _ip_metrics["auc_macro"]
        out[_cn]["ip_accuracy"]     = _ip_metrics["accuracy"]
        out[_cn]["ip_recall_macro"] = _ip_metrics["recall_macro"]


def process_run(run_dir: Path, feature_type: str, label_set: str,
                n_estimators: int, n_neighbors: int, lr_C: float,
                seed: int, force: bool,
                class_weight_mode: str = None, class_weight_strength: float = 0.0):
    """Train RF, KNN, and LR for one run directory.

    Returns a result dict with keys rf/knn/lr (each with f1_macro, auc_macro, accuracy),
    or a failure dict (with key 'error' and 'detail').
    """
    _cw_base         = f"cw{class_weight_mode}" if class_weight_mode else "cwNone"
    _cw_tag          = ("cwNone" if not class_weight_mode
                        else _cw_base if class_weight_strength == 1.0
                        else f"{_cw_base}{class_weight_strength}")
    clf_dir          = run_dir / "data" / "classifiers" / "simple_downstream" / f"{label_set}_{_cw_tag}"
    rf_path          = clf_dir / f"rf_{feature_type}.json"
    knn_path         = clf_dir / f"knn_{feature_type}.json"
    lr_path          = clf_dir / f"lr_{feature_type}.json"
    _frac_cache_path    = clf_dir / "label_fraction_metrics.json"
    _need_ip_frac       = not label_set.endswith("_pure")
    _frac_ip_cache_path = clf_dir / "label_fraction_metrics_initial_pure_eval.json"

    all_cached = (rf_path.exists() and knn_path.exists() and lr_path.exists()
                  and _frac_cache_path.exists()
                  and (not _need_ip_frac or _frac_ip_cache_path.exists()))
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
        if label_set == "full":
            _do_ip_eval(out, clf_dir, feature_type, run_dir.name,
                        y_test_full=None, test_mask=None)
        return out

    print(f"  [{run_dir.name}] processing...", flush=True)
    # Locate data_splits/ by walking upward — robust to any nesting depth
    _search = run_dir.parent
    for _ in range(5):
        if (_search / "data_splits").is_dir():
            break
        _search = _search.parent
    splits_dir = _search / "data_splits" / "42"
    feat_dir   = run_dir / "data" / "byol"

    # ── Load features ────────────────────────────────────────────────────────
    train_feat_path = feat_dir / f"labelled_train_{feature_type}.npy"
    test_feat_path  = feat_dir / f"test_{feature_type}.npy"

    if not train_feat_path.exists():
        return dict(name=run_dir.name, error="missing_data",
                    detail=f"Missing: {train_feat_path}")
    if not test_feat_path.exists():
        return dict(name=run_dir.name, error="missing_data",
                    detail=f"Missing: {test_feat_path}")

    X_train_raw = np.load(train_feat_path).astype(np.float32)
    X_test_raw  = np.load(test_feat_path).astype(np.float32)

    # ── Load labels ──────────────────────────────────────────────────────────
    # Prefer per-run labels saved alongside projections (avoids shared splits_dir
    # being overwritten by a different run with a different f_label).
    run_lab_labels_path = feat_dir / "labelled_train_labels.npy"
    lab_labels_path     = splits_dir / "labelled_train_labels.npy"

    if run_lab_labels_path.exists():
        y_train_full = np.load(run_lab_labels_path)
    elif lab_labels_path.exists():
        y_train_full = np.load(lab_labels_path)
        # If sizes don't match, try to reconstruct full train labels from splits.
        if len(y_train_full) != len(X_train_raw):
            train_idx_path   = splits_dir / "train_idx.npy"
            lab_idx_path     = splits_dir / "labelled_train_idx.npy"
            unlab_idx_path   = splits_dir / "unlabelled_train_idx.npy"
            unlab_labels_path = splits_dir / "unlabelled_train_labels.npy"
            if (train_idx_path.exists() and lab_idx_path.exists()
                    and unlab_idx_path.exists() and unlab_labels_path.exists()):
                train_idx   = np.load(train_idx_path)
                lab_idx     = np.load(lab_idx_path)
                unlab_idx   = np.load(unlab_idx_path)
                unlab_labels = np.load(unlab_labels_path)
                lab_labels   = y_train_full  # already loaded (from splits_dir)
                # Reconstruct labels ordered by train_idx
                if len(X_train_raw) == len(train_idx):
                    idx_map = {}
                    for i, idx in enumerate(lab_idx):
                        idx_map[idx] = lab_labels[i]
                    for i, idx in enumerate(unlab_idx):
                        idx_map[idx] = unlab_labels[i]
                    y_train_full = np.stack([idx_map[idx] for idx in train_idx])
                else:
                    return dict(name=run_dir.name, error="label_size_mismatch",
                                detail=f"X_train has {len(X_train_raw)} rows but labels "
                                       f"have {len(y_train_full)}; cannot reconstruct "
                                       f"without per-run labelled_train_labels.npy")
            else:
                return dict(name=run_dir.name, error="label_size_mismatch",
                            detail=f"X_train has {len(X_train_raw)} rows but labels "
                                   f"have {len(y_train_full)}; missing splits files for reconstruction")
    else:
        return dict(name=run_dir.name, error="missing_data",
                    detail=f"Missing: {lab_labels_path}")

    test_labels_path = splits_dir / "test_labels.npy"
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
        _base_ls    = label_set[:-7] if label_set.endswith('_binary') else label_set
        class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[_base_ls]]

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

    # ── Per-sample weights ────────────────────────────────────────────────────
    _pure_to_base = {"initial_pure": "initial", "classical_pure": "classical",
                     "morphology_pure": "morphology"}

    if is_multiclass:
        # Pure label sets: each row has exactly one positive → per-sample weighting is valid.
        _eff_cw_mode = class_weight_mode
        if class_weight_mode not in (None, "score"):
            _eff_cw_mode = _pure_to_base.get(label_set, class_weight_mode)
        sample_weights = compute_sample_weights(y_train_full[train_mask, :20],
                                                _eff_cw_mode, class_weight_strength)
    else:
        # Multi-label label sets: mirror train_baseline_classifier.py.
        # compute_class_weights → (20,) alpha → slice to label_set columns
        # → per-sample weight = mean alpha over each sample's positive classes.
        if (class_weight_mode is not None and class_weight_mode != "score"
                and class_weight_strength > 0.0
                and label_set != "derived"):
            _alpha_full = compute_class_weights(
                y_train_full[train_mask, :20],
                class_weight_mode,
                class_weight_strength,
            )                                           # (20,)
            _label_cols = LABEL_SETS[label_set[:-7] if label_set.endswith('_binary') else label_set]  # list of int indices
            _alpha_arr  = _alpha_full[_label_cols]      # (n_classes,)
            # y_train_raw: (N_train, n_classes), already filtered to label_set columns
            _row_sums = y_train_raw.sum(axis=1)         # positives per sample
            _pos_alpha = np.where(
                _row_sums > 0,
                (y_train_raw * _alpha_arr).sum(axis=1) / np.maximum(_row_sums, 1),
                1.0,
            )
            sample_weights = (_pos_alpha / _pos_alpha.mean()).astype(np.float32)
        elif class_weight_mode == "score":
            sample_weights = compute_sample_weights(y_train_full[train_mask, :20],
                                                    "score", class_weight_strength)
        else:
            sample_weights = compute_sample_weights(y_train_full[train_mask, :20],
                                                    None, class_weight_strength)

    clf_dir.mkdir(parents=True, exist_ok=True)

    hparams_path = clf_dir / "hparams.txt"
    if not hparams_path.exists() or force:
        with open(hparams_path, "w") as _fh:
            _fh.write(f"run_dir:               {run_dir}\n")
            _fh.write(f"label_set:             {label_set}\n")
            _fh.write(f"feature_type:          {feature_type}\n")
            _fh.write(f"n_estimators:          {n_estimators}\n")
            _fh.write(f"n_neighbors:           {n_neighbors}\n")
            _fh.write(f"lr_C:                  {lr_C}\n")
            _fh.write(f"seed:                  {seed}\n")
            _fh.write(f"class_weight_mode:     {class_weight_mode}\n")
            _fh.write(f"class_weight_strength: {class_weight_strength}\n")

    out = dict(name=run_dir.name)

    # ── Classifiers ───────────────────────────────────────────────────────────
    _specs = [
        ("rf",  rf_path,
         RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
         if is_multiclass else
         MultiOutputClassifier(RandomForestClassifier(n_estimators=n_estimators,
                                                      random_state=seed, n_jobs=-1))),
        ("knn", knn_path,
         KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
         if is_multiclass else
         MultiOutputClassifier(KNeighborsClassifier(n_neighbors=n_neighbors,
                                                    metric="euclidean", n_jobs=-1))),
        ("lr",  lr_path,
         LogisticRegression(max_iter=1000, C=lr_C, random_state=seed)
         if is_multiclass else
         MultiOutputClassifier(LogisticRegression(max_iter=1000, C=lr_C,
                                                  random_state=seed))),
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

        is_mo = not is_multiclass
        # KNN does not support sample_weight; RF and LR do
        sw = None if clf_name == "knn" else sample_weights
        metrics, y_pred_test, y_prob_test = _fit_and_eval(
            clf, X_train, y_train, X_test, y_test,
            class_names, label_set, is_multi_output=is_mo, sample_weight=sw)
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

        np.save(clf_dir / f"{clf_name}_{feature_type}_test_preds.npy", y_pred_test)
        np.save(clf_dir / f"{clf_name}_{feature_type}_test_probs.npy", y_prob_test)
        _lbl_path = clf_dir / f"{feature_type}_test_labels.npy"
        if not _lbl_path.exists():
            np.save(_lbl_path, y_test)

        out[clf_name] = {
            "f1_macro":     metrics["f1_macro"],
            "auc_macro":    metrics["auc_macro"],
            "accuracy":     metrics["accuracy"],
            "recall_macro": metrics["recall_macro"],
        }

    # ── initial_pure evaluation (when not training on a pure label set) ─────────
    if not label_set.endswith("_pure"):
        _do_ip_eval(out, clf_dir, feature_type, run_dir.name,
                    y_test_full=y_test_full, test_mask=test_mask)

    # ── Label-fraction sweep (all label sets except "derived") ───────────────
    _do_frac_sweep = label_set != "derived"
    _do_ip_frac    = _need_ip_frac and (not _frac_ip_cache_path.exists() or force)
    if _do_frac_sweep and (not _frac_cache_path.exists() or _do_ip_frac or force):
        _FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
        _rng   = np.random.default_rng(seed)
        _frac_out: dict    = {}
        # Pre-initialise all fraction keys so the notebook's exact-key check always passes
        _frac_ip_out: dict = {str(_frac): {} for _frac in _FRACS}
        for _frac in _FRACS:
            _n   = max(len(class_names) * 2, int(_frac * len(X_train)))
            _idx = (_rng.choice(len(X_train), size=_n, replace=False)
                    if _frac < 1.0 else np.arange(len(X_train)))
            _Xf, _yf = X_train[_idx], y_train[_idx]
            _Xf_fit, _yf_fit = _Xf, _yf
            _sw_frac = sample_weights[_idx] if sample_weights is not None else None
            _Xt_fit, _yt_fit = X_test, y_test
            _ncls = len(class_names)
            _frac_out[str(_frac)] = {}
            _knn_k = min(n_neighbors, max(1, len(_Xf_fit)))
            # Pre-compute ip mask for this fraction (constant across classifiers)
            if _do_ip_frac:
                _ip_mask_frac = (_yt_fit[:, :5].sum(axis=1) == 1)
                _y_ip_frac    = (_yt_fit[_ip_mask_frac, :5].argmax(axis=1)
                                 if _ip_mask_frac.any() else None)
            for _cname, _clf_base in [
                ("LogReg", LogisticRegression(max_iter=1000, C=lr_C,
                                              random_state=seed)),
                ("KNN",    KNeighborsClassifier(n_neighbors=_knn_k,
                                                metric="euclidean", n_jobs=-1)),
                ("RF",     RandomForestClassifier(n_estimators=50,
                                                  random_state=seed, n_jobs=-1)),
            ]:
                if len(_Xf_fit) == 0:
                    continue
                _clf_frac = (MultiOutputClassifier(_clf_base) if not is_multiclass
                             else _clf_base)
                _fit_sw = None if _cname == "KNN" else _sw_frac
                try:
                    if _fit_sw is not None:
                        _clf_frac.fit(_Xf_fit, _yf_fit, sample_weight=_fit_sw)
                    else:
                        _clf_frac.fit(_Xf_fit, _yf_fit)
                except ValueError:
                    # Some label columns may be all-zero in small fractions
                    continue
                _yp = _clf_frac.predict(_Xt_fit)
                if not is_multiclass:
                    def _pos_prob(est, X):
                        p = est.predict_proba(X)
                        # If only one class seen during training, proba has shape (N,1)
                        return p[:, 1] if p.shape[1] > 1 else p[:, 0]
                    _prob = np.stack(
                        [_pos_prob(est, _Xt_fit)
                         for est in _clf_frac.estimators_], axis=1)
                    _aucs = []
                    for _i in range(_ncls):
                        if len(np.unique(_yt_fit[:, _i])) < 2:
                            _aucs.append(None)
                        else:
                            _aucs.append(float(roc_auc_score(
                                _yt_fit[:, _i], _prob[:, _i])))
                    _auc = float(np.nanmean([a for a in _aucs if a is not None]))
                else:
                    _prob = _clf_frac.predict_proba(_Xt_fit)
                    if _prob.shape[1] != _ncls:
                        _full_p = np.zeros((len(_Xt_fit), _ncls), dtype=np.float64)
                        for _j, _c in enumerate(_clf_frac.classes_):
                            _full_p[:, int(_c)] = _prob[:, _j]
                        _prob = _full_p
                    _y_bin = label_binarize(_yt_fit, classes=list(range(_ncls)))
                    try:
                        _auc = float(roc_auc_score(_y_bin, _prob,
                                                   multi_class="ovr", average="macro"))
                    except Exception:
                        _auc = float("nan")
                _frac_out[str(_frac)][_cname] = {
                    "f1":       float(f1_score(_yt_fit, _yp, average="macro",
                                               zero_division=0)),
                    "auc":      _auc,
                    "accuracy": (float((_yt_fit == _yp).mean()) if label_set.endswith('_binary')
                                 else float(accuracy_score(_yt_fit, _yp))),
                    "recall":   float(recall_score(_yt_fit, _yp, average="macro",
                                                   zero_division=0)),
                }
                # Cross-eval: evaluate on initial_pure (5-class) subset
                if _do_ip_frac and _y_ip_frac is not None and not is_multiclass:
                    _p5_f  = _prob[_ip_mask_frac, :5]
                    _p5_fn = _p5_f / _p5_f.sum(axis=1, keepdims=True).clip(min=1e-9)
                    _yp_ip_f  = _p5_fn.argmax(axis=1)
                    _y_ip_bin = label_binarize(_y_ip_frac, classes=list(range(5)))
                    try:
                        _ip_auc = float(roc_auc_score(
                            _y_ip_bin, _p5_fn, multi_class="ovr", average="macro"))
                    except Exception:
                        _ip_auc = float("nan")
                    _frac_ip_out[str(_frac)][_cname] = {
                        "f1":       float(f1_score(_y_ip_frac, _yp_ip_f,
                                                   average="macro", zero_division=0)),
                        "auc":      _ip_auc,
                        "accuracy": float(accuracy_score(_y_ip_frac, _yp_ip_f)),
                        "recall":   float(recall_score(_y_ip_frac, _yp_ip_f,
                                                       average="macro", zero_division=0)),
                    }
        if not _frac_cache_path.exists() or force:
            with open(_frac_cache_path, "w") as _fh:
                json.dump(_frac_out, _fh, indent=2)
            print(f"    Saved label-fraction sweep → {_frac_cache_path.name}",
                  flush=True)
        if _do_ip_frac and _frac_ip_out:
            with open(_frac_ip_cache_path, "w") as _fh:
                json.dump(_frac_ip_out, _fh, indent=2)
            print(f"    Saved IP cross-eval sweep → {_frac_ip_cache_path.name}",
                  flush=True)

    return out


# ---------------------------------------------------------------------------
# Multiprocessing worker (top-level for pickling)
# ---------------------------------------------------------------------------

def _worker(args):
    run_dir, feature_type, label_set, n_estimators, n_neighbors, lr_C, seed, force, cw_mode, cw_strength = args
    try:
        return process_run(run_dir, feature_type, label_set,
                           n_estimators, n_neighbors, lr_C, seed, force,
                           class_weight_mode=cw_mode, class_weight_strength=cw_strength)
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
    parser.add_argument("--outputs-root", default="outputs/byol_runs",
                        help="Root directory containing run subdirectories (default: outputs/byol_runs).")
    parser.add_argument("--run-glob",     default="enb0_*",
                        help="Glob pattern for run directories (default: enb0_*).")
    parser.add_argument("--feature-type", default="projections",
                        choices=["projections", "encodings"],
                        help="Feature vectors to use (default: projections).")
    parser.add_argument("--label-set",    default="classical_pure",
                        help="Classification scheme (default: classical_pure). "
                             "Append '_binary' for element-wise accuracy (e.g. initial_binary).")
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
    parser.add_argument("--class-weight-mode", type=str, default=None,
                        choices=["score", "initial", "initial_pure", "morphology", "morphology_pure",
                                 "environment", "environment_pure", "classical", "classical_pure",
                                 "all", "all_pure", "full", "None"],
                        help="Upweight rare samples: 'score' (interest tier 1-4) or a label-set name. "
                             "'full' is a synonym for 'all'. Pass 'None' or omit to use uniform weights. "
                             "Label-set modes (e.g. initial, morphology) require a pure label set — "
                             "each training sample must have exactly one positive in the selected columns. "
                             "Pass a *_pure label_set (e.g. initial_pure). "
                             "RF and LR accept sample_weight; KNN does not and always receives uniform weights. "
                             "Default: None (uniform).")
    parser.add_argument("--class-weight-strength", type=float, default=0.0,
                        help="Magnitude of class upweighting (0=uniform, default). "
                             "w = clip(1 + strength*(raw_norm - 1), min=0).")
    args = parser.parse_args()

    # Validate label_set (allow _binary suffix on any known base label set)
    _ls_base = args.label_set[:-7] if args.label_set.endswith('_binary') else args.label_set
    if _ls_base not in LABEL_SETS:
        parser.error(f"Unknown label set: {args.label_set!r}")

    # Normalise synonyms: "None" → None, "full" → "all"
    if args.class_weight_mode in ("None", "none"):
        args.class_weight_mode = None
    elif args.class_weight_mode == "full":
        args.class_weight_mode = "all"

    outputs_root = Path(args.outputs_root)

    # ── Discover run directories ───────────────────────────────────────────────
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs if re.search(r'_f[\d.]+_sw(?:cos|lin)?[\d.]+', rd.name)]
    if not run_dirs:
        print(f"No run directories found matching '{args.run_glob}' under {outputs_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs)} run directories.\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    worker_args = [
        (rd, args.feature_type, args.label_set, args.n_estimators,
         args.n_neighbors, args.lr_c, args.seed, args.force,
         args.class_weight_mode, args.class_weight_strength)
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
            key=lambda r: max(
                r.get("rf",  {}).get("f1_macro", -1.0),
                r.get("knn", {}).get("f1_macro", -1.0),
                r.get("lr",  {}).get("f1_macro", -1.0),
            ),
            reverse=True,
        )

        w = max(len(r["name"]) for r in results)
        _cols = [
            ("RF",  "rf",  "f1_macro"),
            ("KNN", "knn", "f1_macro"),
            ("LR",  "lr",  "f1_macro"),
            ("RF",  "rf",  "auc_macro"),
            ("KNN", "knn", "auc_macro"),
            ("LR",  "lr",  "auc_macro"),
            ("RF",  "rf",  "accuracy"),
            ("KNN", "knn", "accuracy"),
            ("LR",  "lr",  "accuracy"),
            ("RF",  "rf",  "recall_macro"),
            ("KNN", "knn", "recall_macro"),
            ("LR",  "lr",  "recall_macro"),
        ]
        _short = {"f1_macro": "F1", "auc_macro": "AUC",
                  "accuracy": "Acc", "recall_macro": "Rec"}
        col_w = 10  # wider to accommodate *x.xxxx*

        # Pre-compute per-column best values for highlighting
        _col_best = {}
        for _, clf, metric in _cols:
            vals = [r.get(clf, {}).get(metric, float("nan")) for r in results]
            valid = [v for v in vals if v == v]
            _col_best[(clf, metric)] = max(valid) if valid else float("nan")

        def _fmt(r, clf, metric, width):
            v = r.get(clf, {}).get(metric, float("nan"))
            if v != v:
                return " " * (width - 3) + "N/A"
            s = f"{v:.4f}"
            if v == _col_best[(clf, metric)]:
                s = f"*{s}*"
            return f"{s:>{width}}"

        hdr = f"{'Rank':>4}  " + "  ".join(
            f"{c[0]+' '+_short[c[2]]:>{col_w}}" for c in _cols
        ) + f"  {'Run':<{w}}"
        sep = "=" * len(hdr)
        print(f"\n{sep}")
        print(f"RF / KNN / LR ({args.label_set} / {args.feature_type}) — ranked by best F1-macro")
        print(sep)
        print(hdr)
        print("-" * len(hdr))
        for i, r in enumerate(results, 1):
            row = f"{i:>4}  " + "  ".join(
                _fmt(r, clf, metric, col_w) for _, clf, metric in _cols
            ) + f"  {r['name']:<{w}}"
            print(row)
        print(sep)

    if results:
        print_statistical_summary(
            results,
            title=f"RF / KNN / LR ({args.label_set} / {args.feature_type})",
        )

    # ── initial_pure summary (when not training on a pure label set) ──────────
    if not args.label_set.endswith("_pure") and results:
        _ip_results = [r for r in results
                       if any("ip_f1_macro" in r.get(c, {}) for c in ("rf", "knn", "lr"))]
        if _ip_results:
            _ip_cols = [
                ("RF",  "rf",  "ip_f1_macro"),
                ("KNN", "knn", "ip_f1_macro"),
                ("LR",  "lr",  "ip_f1_macro"),
                ("RF",  "rf",  "ip_auc_macro"),
                ("KNN", "knn", "ip_auc_macro"),
                ("LR",  "lr",  "ip_auc_macro"),
                ("RF",  "rf",  "ip_accuracy"),
                ("KNN", "knn", "ip_accuracy"),
                ("LR",  "lr",  "ip_accuracy"),
                ("RF",  "rf",  "ip_recall_macro"),
                ("KNN", "knn", "ip_recall_macro"),
                ("LR",  "lr",  "ip_recall_macro"),
            ]
            _ip_short = {
                "ip_f1_macro": "F1", "ip_auc_macro": "AUC",
                "ip_accuracy": "Acc", "ip_recall_macro": "Rec",
            }
            _ip_best = {}
            for _, clf, metric in _ip_cols:
                vals = [r.get(clf, {}).get(metric, float("nan")) for r in _ip_results]
                valid = [v for v in vals if v == v]
                _ip_best[(clf, metric)] = max(valid) if valid else float("nan")

            def _fmt_ip(r, clf, metric, width):
                v = r.get(clf, {}).get(metric, float("nan"))
                if v != v:
                    return " " * (width - 3) + "N/A"
                s = f"{v:.4f}"
                if v == _ip_best[(clf, metric)]:
                    s = f"*{s}*"
                return f"{s:>{width}}"

            _ww = max(len(r["name"]) for r in _ip_results)
            _cw = 10
            _ih = (f"{'Rank':>4}  " +
                   "  ".join(f"{c[0]+' '+_ip_short[c[2]]:>{_cw}}" for c in _ip_cols) +
                   f"  {'Run':<{_ww}}")
            _is = "=" * len(_ih)
            print("\n" + _is)
            print(f"RF / KNN / LR (initial_pure eval / {args.feature_type}) — ranked by best F1-macro")
            print(_is)
            print(_ih)
            print("-" * len(_ih))
            for i, r in enumerate(
                sorted(_ip_results,
                       key=lambda r: max(r.get("rf", {}).get("ip_f1_macro", -1.0),
                                         r.get("knn", {}).get("ip_f1_macro", -1.0),
                                         r.get("lr",  {}).get("ip_f1_macro", -1.0)),
                       reverse=True), 1
            ):
                _row = (f"{i:>4}  " +
                        "  ".join(_fmt_ip(r, clf, metric, _cw)
                                   for _, clf, metric in _ip_cols) +
                        f"  {r['name']:<{_ww}}")
                print(_row)
            print(_is)
            print_statistical_summary(
                _ip_results,
                metrics=("ip_f1_macro", "ip_auc_macro", "ip_accuracy", "ip_recall_macro"),
                title=f"RF / KNN / LR (initial_pure eval / {args.feature_type})",
            )

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


# ---------------------------------------------------------------------------
# Statistical summary — average performance per hyperparameter value
# ---------------------------------------------------------------------------

# Patterns to extract hyperparameter values from run directory names.
_HPARAM_RE = {
    "ema":       r"_ema([\d.]+)",
    "vicregvar": r"_vicregvar([\d.]+)",
    "cov":       r"_cov([\d.]+)",
    "gamma":     r"_gamma([\d.]+)",
    "f":         r"_f([\d.]+)_sw",
    "sw":        r"_sw((?:cos|lin)?[\d.]+(?:to[\d.]+)?)(?:_|$)",
    "pd":        r"_pd(\d+)",
}


def _parse_hparams(name: str) -> dict:
    out = {}
    for param, pat in _HPARAM_RE.items():
        m = re.search(pat, name)
        if m:
            out[param] = m.group(1)
    return out


def _parse_aug_type(name: str) -> str:
    """Return the augmentation tag from a run name, or 'standard' if absent.

    Tags like 'augextended', 'augquart', 'augquart_ext' sit between the sw
    value and the trailing _YYYYMMDD_HHMM timestamp.
    """
    name_no_ts = re.sub(r'_\d{8}_\d{4}$', '', name)
    m = re.search(r'_(aug\S+)$', name_no_ts)
    return m.group(1) if m else 'standard'


def print_statistical_summary(results: list,
                               clfs=("rf", "knn", "lr"),
                               metrics=("f1_macro", "auc_macro", "accuracy", "recall_macro"),
                               title: str = None):
    """For each hyperparameter, group runs by that param's value and print mean±std + N."""

    metric_labels = {
        "f1_macro":        "F1",
        "auc_macro":       "AUC",
        "accuracy":        "Acc",
        "recall_macro":    "Rec",
        "ip_f1_macro":     "F1",
        "ip_auc_macro":    "AUC",
        "ip_accuracy":     "Acc",
        "ip_recall_macro": "Rec",
    }

    parsed = [(r, _parse_hparams(r["name"])) for r in results]
    clf_metric_cols = [(c, m) for c in clfs for m in metrics]
    col_headers = [f"{c.upper()} {metric_labels[m]}" for c, m in clf_metric_cols]
    col_w = 12
    _first_group = True

    for param in sorted(_HPARAM_RE.keys()):
        groups: dict = {}
        for r, hp in parsed:
            if param in hp:
                groups.setdefault(hp[param], []).append(r)

        if len(groups) < 2:
            continue

        name_w = max(len(str(v)) for v in groups)
        hdr = (f"  {param:<{name_w}}  {'N':>3}  " +
               "  ".join(f"{h:^{col_w}}" for h in col_headers))
        sep = "-" * len(hdr)

        if _first_group and title:
            print(f"\n{'#' * (len(hdr) + 2)}")
            print(f"# Statistical summary — {title}")
            print(f"{'#' * (len(hdr) + 2)}")
            _first_group = False
        print(f"\n{'=' * len(hdr)}")
        print(f"Grouped by: {param}")
        print(f"{'=' * len(hdr)}")
        print(hdr)
        print(sep)

        # Pre-compute per-column best mean across groups
        _group_means = {}
        for val, group in groups.items():
            for clf, metric in clf_metric_cols:
                vals = [r.get(clf, {}).get(metric, float("nan")) for r in group]
                vals = [v for v in vals if v == v]
                _group_means.setdefault((clf, metric), {})[val] = (
                    float(np.mean(vals)) if vals else float("nan")
                )
        _col_best_mean = {}
        for (clf, metric), val_means in _group_means.items():
            valid = [v for v in val_means.values() if v == v]
            _col_best_mean[(clf, metric)] = max(valid) if valid else float("nan")

        def _sort_key(v):
            try:
                return (0, float(v), "")
            except (ValueError, TypeError):
                return (1, 0.0, str(v))

        for val in sorted(groups, key=_sort_key):
            group = groups[val]
            n = len(group)
            cells = []
            for clf, metric in clf_metric_cols:
                vals = [r.get(clf, {}).get(metric, float("nan")) for r in group]
                vals = [v for v in vals if v == v]
                if not vals:
                    cells.append(f"{'N/A':^{col_w}}")
                else:
                    mu = float(np.mean(vals))
                    best = _col_best_mean[(clf, metric)]
                    is_best = (mu == mu) and (best == best) and (mu == best)
                    if len(vals) == 1:
                        s = f"{mu:.3f}"
                    else:
                        s = f"{mu:.3f}±{float(np.std(vals)):.3f}"
                    if is_best:
                        s = f"*{s}*"
                    cells.append(f"{s:^{col_w}}")
            print(f"  {val:<{name_w}}  {n:>3}  " + "  ".join(cells))

        print(sep)

    # ── Augmentation grouping ──────────────────────────────────────────────────
    aug_groups: dict = {}
    for r, _ in parsed:
        aug = _parse_aug_type(r["name"])
        aug_groups.setdefault(aug, []).append(r)

    if len(aug_groups) >= 2:
        name_w = max(len(str(v)) for v in aug_groups)
        hdr = (f"  {'augmentation':<{name_w}}  {'N':>3}  " +
               "  ".join(f"{h:^{col_w}}" for h in col_headers))
        sep = "-" * len(hdr)

        if _first_group and title:
            print(f"\n{'#' * (len(hdr) + 2)}")
            print(f"# Statistical summary — {title}")
            print(f"{'#' * (len(hdr) + 2)}")
            _first_group = False
        print(f"\n{'=' * len(hdr)}")
        print("Grouped by: augmentation")
        print(f"{'=' * len(hdr)}")
        print(hdr)
        print(sep)

        _aug_means = {}
        for val, group in aug_groups.items():
            for clf, metric in clf_metric_cols:
                vals = [r.get(clf, {}).get(metric, float("nan")) for r in group]
                vals = [v for v in vals if v == v]
                _aug_means.setdefault((clf, metric), {})[val] = (
                    float(np.mean(vals)) if vals else float("nan")
                )
        _aug_col_best = {}
        for (clf, metric), val_means in _aug_means.items():
            valid = [v for v in val_means.values() if v == v]
            _aug_col_best[(clf, metric)] = max(valid) if valid else float("nan")

        for val in sorted(aug_groups):
            group = aug_groups[val]
            n = len(group)
            cells = []
            for clf, metric in clf_metric_cols:
                vals = [r.get(clf, {}).get(metric, float("nan")) for r in group]
                vals = [v for v in vals if v == v]
                if not vals:
                    cells.append(f"{'N/A':^{col_w}}")
                else:
                    mu = float(np.mean(vals))
                    best = _aug_col_best[(clf, metric)]
                    is_best = (mu == mu) and (best == best) and (mu == best)
                    if len(vals) == 1:
                        s = f"{mu:.3f}"
                    else:
                        s = f"{mu:.3f}±{float(np.std(vals)):.3f}"
                    if is_best:
                        s = f"*{s}*"
                    cells.append(f"{s:^{col_w}}")
            print(f"  {val:<{name_w}}  {n:>3}  " + "  ".join(cells))

        print(sep)


if __name__ == "__main__":
    main()
