"""Noise robustness sweep for downstream classifiers.

Feature noise: train on clean projections, evaluate on test projections
with additive isotropic Gaussian noise (σ sweep). Follows the "train
clean / test noisy" paradigm — the classifier/features are never touched,
only what happens on the test side.

Outputs
-------
  <out_dir>/feat_noise/sw{sw}_ls{ls}_fold{k}_sigma{s}.json

Each JSON: sw, label_set, fold, noise_type, noise_level,
           classifier, f1_macro, f1_per_class, accuracy, class_names.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from suplat.data.label_sets import (
    ALL_CLASS_NAMES,
    LABEL_SETS,
    apply_label_set,
)


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def _make_clf(name, seed=42):
    if name == "lr":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, class_names):
    return {
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Per-run-dir / per-fold processing
# ---------------------------------------------------------------------------

def _process_fold(
    fold_dir: Path,
    run_name: str,
    sw: float,
    label_set: str,
    feature_type: str,
    feat_sigmas,
    classifiers,
    seed: int,
    out_dir: Path,
    data_seed,
    force: bool = False,
):
    # ── Locate splits_dir ─────────────────────────────────────────────────
    _search = fold_dir.parent
    for _ in range(6):
        if (_search / "data_splits").is_dir():
            break
        _search = _search.parent
    _splits_base = _search / "data_splits" / str(data_seed)

    # Support both cross_val and plain seed dirs
    _fold_m = re.match(r"cross_val_(\d+)", fold_dir.name)
    if _fold_m:
        fold_idx   = int(_fold_m.group(1))
        splits_dir = _splits_base / f"cross_val_{fold_idx}"
    else:
        fold_idx   = 0
        splits_dir = _splits_base

    # ── Load features ─────────────────────────────────────────────────────
    feat_dir = fold_dir / "data" / "byol"
    tr_path  = feat_dir / f"labelled_train_{feature_type}.npy"
    te_path  = feat_dir / f"test_{feature_type}.npy"
    if not tr_path.exists() or not te_path.exists():
        print(f"  [skip] missing features in {fold_dir}", flush=True)
        return

    X_tr_raw = np.load(tr_path).astype(np.float32)
    X_te_raw = np.load(te_path).astype(np.float32)

    # ── Load labels ───────────────────────────────────────────────────────
    _f_m  = re.search(r'_f([\d.]+)', run_name)
    _f_tag = ""
    if _f_m:
        _fv    = float(_f_m.group(1))
        _f_tag = f"_f{int(_fv) if _fv == int(_fv) else _fv}"

    run_lab_path = feat_dir / "labelled_train_labels.npy"
    lab_path     = splits_dir / f"labelled_train_labels{_f_tag}.npy"
    if run_lab_path.exists():
        y_tr_full = np.load(run_lab_path)
    elif lab_path.exists():
        y_tr_full = np.load(lab_path)
    else:
        print(f"  [skip] missing train labels for {fold_dir}", flush=True)
        return

    test_lab_path = splits_dir / "test_labels.npy"
    if not test_lab_path.exists():
        print(f"  [skip] missing test labels for {fold_dir}", flush=True)
        return
    y_te_full = np.load(test_lab_path)

    # ── Apply label set ───────────────────────────────────────────────────
    y_tr_sub, tr_mask = apply_label_set(y_tr_full, label_set)
    y_te_sub, te_mask = apply_label_set(y_te_full, label_set)

    X_tr = X_tr_raw[tr_mask]
    X_te = X_te_raw[te_mask]

    if len(X_tr) == 0:
        print(f"  [skip] empty train after label set '{label_set}'", flush=True)
        return

    # Class names
    _pure_sets = {"classical_pure", "initial_pure", "morphology_pure",
                  "environment_pure", "full_pure"}
    is_multiclass = label_set in _pure_sets
    _base_ls = (label_set[:-11] if label_set.endswith("_individual") else label_set)
    class_names = [ALL_CLASS_NAMES[i] for i in LABEL_SETS[_base_ls]]

    if is_multiclass:
        y_tr_int = y_tr_sub.argmax(axis=1)
        y_te_int = y_te_sub.argmax(axis=1)
    else:
        # Multi-label sets aren't supported by the single-output LR/MLP
        # classifiers here, so they're silently skipped (feature noise loop
        # below only runs when is_multiclass).
        y_tr_int = None
        y_te_int = None

    # ── Fit scaler on clean train ─────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # Per-feature std of TRAIN (used for sigma parameterisation)
    feat_std = scaler.scale_.astype(np.float32)   # shape (n_feat,)

    # ── Output dirs ───────────────────────────────────────────────────────
    fn_out  = out_dir / "feat_noise"
    fn_out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── Feature noise loop ────────────────────────────────────────────────
    # Train classifiers ONCE on clean scaled train, then evaluate on noisy test
    if is_multiclass:
        for clf_name in classifiers:
            clf = _make_clf(clf_name, seed=seed)
            try:
                clf.fit(X_tr_sc, y_tr_int)
            except Exception as e:
                print(f"  [warn] clf={clf_name} fit failed: {e}", flush=True)
                continue
            for sigma in feat_sigmas:
                _out_path = fn_out / f"sw{sw}_ls{label_set}_fold{fold_idx}_sigma{sigma}_{clf_name}.json"
                if _out_path.exists() and not force:
                    continue
                if sigma == 0.0:
                    X_te_noisy = X_te_sc
                else:
                    noise = rng.standard_normal(X_te_sc.shape).astype(np.float32)
                    # sigma as fraction of per-feature std — but X_te_sc is already
                    # scaled by StandardScaler, so feat_std in scaled space is 1.0.
                    # We parameterise as "sigma std units" in the SCALED space.
                    X_te_noisy = X_te_sc + sigma * noise

                y_pred = clf.predict(X_te_noisy)
                result = {
                    "sw":          sw,
                    "label_set":   label_set,
                    "fold":        fold_idx,
                    "noise_type":  "feature",
                    "noise_level": sigma,
                    "classifier":  clf_name,
                    **_metrics(y_te_int, y_pred, class_names),
                }
                with open(_out_path, "w") as fh:
                    json.dump(result, fh)
            print(f"  [fn] sw={sw} ls={label_set} fold={fold_idx} clf={clf_name} done",
                  flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness sweep for downstream classifiers."
    )
    parser.add_argument("--outputs_root",  required=True,
                        help="Path to outputs/ directory.")
    parser.add_argument("--run_glob",      default="byol_runs/pd128_*_f1",
                        help="Glob pattern under outputs_root for run name dirs.")
    parser.add_argument("--sw_vals",       nargs="+", type=float,
                        default=[0.0, 0.1, 1.0, 10.0])
    parser.add_argument("--label_sets",    nargs="+",
                        default=["initial_pure", "initial_binary"])
    parser.add_argument("--feature_type",  default="projections")
    parser.add_argument("--n_folds",       type=int, default=5)
    parser.add_argument("--feat_sigmas",   nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 1.0, 2.0],
                        help="Gaussian sigma as std units in scaled feature space.")
    parser.add_argument("--out_dir",       required=True,
                        help="Where to write result JSON files.")
    parser.add_argument("--classifiers",   nargs="+", default=["lr", "mlp"])
    parser.add_argument("--data_seed",     type=int, default=2)
    parser.add_argument("--training_seed", type=int, default=2)
    parser.add_argument("--seed",          type=int, default=42,
                        help="RNG seed for noise and classifier init.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing result JSONs instead of skipping them.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect matching run name dirs
    run_name_dirs = sorted(outputs_root.glob(args.run_glob))
    print(f"Found {len(run_name_dirs)} run name dirs matching '{args.run_glob}'")

    for rnd in run_name_dirs:
        _sw_m = re.search(r'_sw([\d.]+)', rnd.name)
        if not _sw_m:
            continue
        sw = float(_sw_m.group(1))
        if sw not in args.sw_vals:
            continue

        # Locate cross-val fold dirs
        _sd = rnd / f"data_seed_{args.data_seed}" / f"training_seed_{args.training_seed}"
        if not _sd.is_dir():
            print(f"  [skip] seed dir not found: {_sd}", flush=True)
            continue

        fold_dirs = sorted(_sd.glob("cross_val_*"))
        if not fold_dirs:
            fold_dirs = [_sd]   # single non-CV run

        for fold_dir in fold_dirs:
            for label_set in args.label_sets:
                print(f"\n── {rnd.name} | fold={fold_dir.name} | ls={label_set} ──",
                      flush=True)
                _process_fold(
                    fold_dir     = fold_dir,
                    run_name     = rnd.name,
                    sw           = sw,
                    label_set    = label_set,
                    feature_type = args.feature_type,
                    feat_sigmas  = args.feat_sigmas,
                    classifiers  = args.classifiers,
                    seed         = args.seed,
                    out_dir      = out_dir,
                    data_seed    = args.data_seed,
                    force        = args.force,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
