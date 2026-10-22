"""
threshold_tuning_analysis.py — Diagnostic for threshold-vs-class-weighting comparison.

IMPORTANT: All threshold tuning is strictly leave-one-fold-out. For each fold k,
thresholds are derived exclusively from the 4 remaining folds. Fold k's test data
is NEVER used to select thresholds applied to fold k. This ensures no data leakage.

Part 1: Per-class AUC and F1 table comparing cwNone vs cwinitial_pure.
Part 2: LOFO threshold tuning on cwNone LR probabilities; reports macro-F1 gain.

Usage:
    source .venv/bin/activate
    python scripts/threshold_tuning_analysis.py \\
        --outputs-root outputs/byol_runs \\
        --run-name pd128_qext_v1_wd1e-3_lrconst_sw0.1_f1 \\
        --data-seed 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fold_dir(run_dir: Path, data_seed: int, seed: int, fold: int) -> Path:
    return (run_dir / f"data_seed_{data_seed}" / f"training_seed_{seed}"
            / f"cross_val_{fold}" / "data" / "downstream_classifiers")


def _load_json_metrics(fold_dir: Path, cw_tag: str, field: str) -> list:
    p = fold_dir / f"initial_pure_{cw_tag}" / "lr_projections.json"
    if not p.exists():
        return None
    with open(p) as fh:
        d = json.load(fh)
    return d.get(field)


def _load_probs_labels(fold_dir: Path, cw_tag: str):
    d = fold_dir / f"initial_pure_{cw_tag}"
    probs  = np.load(d / "lr_projections_test_probs.npy")
    labels = np.load(d / "projections_test_labels.npy")
    return probs, labels


# ── Part 1: AUC / F1 table ────────────────────────────────────────────────────

def part1_auc_f1_table(run_dir: Path, data_seed: int, seed: int, n_folds: int):
    print("=" * 70)
    print("PART 1 — Per-class AUC and F1: cwNone vs cwinitial_pure")
    print("         (mean ± std across 5 folds, ds=2, LR on projections)")
    print("=" * 70)

    class_names = ["FRI", "FRII", "Hybrids", "Spirals", "Relaxed"]
    n_classes = len(class_names)

    auc_none   = np.full((n_folds, n_classes), np.nan)
    auc_cw     = np.full((n_folds, n_classes), np.nan)
    f1_none    = np.full((n_folds, n_classes), np.nan)
    f1_cw      = np.full((n_folds, n_classes), np.nan)

    for k in range(n_folds):
        fd = _fold_dir(run_dir, data_seed, seed, k)
        a_n = _load_json_metrics(fd, "cwNone",        "auc_per_class")
        a_c = _load_json_metrics(fd, "cwinitial_pure","auc_per_class")
        f_n = _load_json_metrics(fd, "cwNone",        "f1_per_class")
        f_c = _load_json_metrics(fd, "cwinitial_pure","f1_per_class")
        if a_n:
            auc_none[k] = [v if v is not None else np.nan for v in a_n]
        if a_c:
            auc_cw[k]   = [v if v is not None else np.nan for v in a_c]
        if f_n:
            f1_none[k]  = f_n
        if f_c:
            f1_cw[k]    = f_c

    # Mean across folds
    mn_auc_none = np.nanmean(auc_none, axis=0)
    mn_auc_cw   = np.nanmean(auc_cw,   axis=0)
    mn_f1_none  = np.nanmean(f1_none,  axis=0)
    mn_f1_cw    = np.nanmean(f1_cw,    axis=0)
    d_auc       = mn_auc_cw  - mn_auc_none
    d_f1        = mn_f1_cw   - mn_f1_none

    col = "{:<17} {:>8} {:>8} {:>7} {:>8} {:>8} {:>7}"
    print(col.format("Class", "cw=None AUC", "cw=init AUC", "ΔAUC",
                     "cw=None F1",  "cw=init F1",  "ΔF1"))
    print("-" * 70)
    for i, cn in enumerate(class_names):
        flag = ""
        if d_auc[i] < 0.01 and d_f1[i] < -0.05:
            flag = " <-- THRESHOLD PROBLEM"
        print(col.format(cn,
              f"{mn_auc_none[i]:.3f}", f"{mn_auc_cw[i]:.3f}", f"{d_auc[i]:+.3f}",
              f"{mn_f1_none[i]:.3f}",  f"{mn_f1_cw[i]:.3f}",  f"{d_f1[i]:+.3f}")
              + flag)

    # Also print macro averages
    print("-" * 70)
    print(col.format("MACRO",
          f"{np.nanmean(mn_auc_none):.3f}", f"{np.nanmean(mn_auc_cw):.3f}",
          f"{np.nanmean(d_auc):+.3f}",
          f"{np.nanmean(mn_f1_none):.3f}",  f"{np.nanmean(mn_f1_cw):.3f}",
          f"{np.nanmean(d_f1):+.3f}"))

    # Conclusion
    spirals_idx = 3
    if d_auc[spirals_idx] < 0.01 and d_f1[spirals_idx] < -0.05:
        print("\n*** THRESHOLD PROBLEM CONFIRMED ***")
        print(f"    Spirals: ΔAUC={d_auc[spirals_idx]:+.3f} (< 0.01)  ΔF1={d_f1[spirals_idx]:+.3f} (< -0.05)")
        print("    Class weighting hurts F1 without improving AUC → wrong threshold at p=0.5.")
    else:
        print("\nNote: Spirals threshold problem NOT confirmed by this criterion.")
    print()


# ── Part 2: LOFO threshold tuning ────────────────────────────────────────────

def part2_threshold_tuning(run_dir: Path, data_seed: int, seed: int, n_folds: int):
    print("=" * 70)
    print("PART 2 — Leave-one-fold-out threshold tuning (cwNone LR probs)")
    print("         Tuned rule: pred = argmax_c( prob[c] / threshold[c] )")
    print("         Thresholds for fold k derived ONLY from folds j ≠ k.")
    print("=" * 70)

    n_classes = 5
    threshold_grid = np.linspace(0.05, 0.95, 50)

    # Load all folds
    all_probs  = []
    all_labels = []
    for k in range(n_folds):
        fd = _fold_dir(run_dir, data_seed, seed, k)
        probs, labels = _load_probs_labels(fd, "cwNone")
        all_probs.append(probs)
        all_labels.append(labels)

    fold_results = []
    all_tuned_thresholds = []

    for k in range(n_folds):
        # Build validation set from folds j ≠ k
        val_probs  = np.concatenate([all_probs[j]  for j in range(n_folds) if j != k], axis=0)
        val_labels = np.concatenate([all_labels[j] for j in range(n_folds) if j != k], axis=0)

        # Tune threshold per class on validation set
        thresholds = np.zeros(n_classes)
        for c in range(n_classes):
            best_t, best_f1 = 0.5, -1.0
            for t in threshold_grid:
                pred_c = (val_probs[:, c] >= t).astype(int)
                true_c = (val_labels == c).astype(int)
                # Handle edge case: all-zero predictions
                if pred_c.sum() == 0:
                    continue
                f1_c = f1_score(true_c, pred_c, zero_division=0)
                if f1_c > best_f1:
                    best_f1 = f1_c
                    best_t  = t
            thresholds[c] = best_t
        all_tuned_thresholds.append(thresholds)

        # Evaluate on fold k
        eval_probs  = all_probs[k]
        eval_labels = all_labels[k]

        pred_default = eval_probs.argmax(axis=1)
        # Scaled argmax: normalise each class prob by its threshold
        scaled = eval_probs / thresholds[np.newaxis, :]
        pred_tuned = scaled.argmax(axis=1)

        f1_default = f1_score(eval_labels, pred_default, average="macro", zero_division=0)
        f1_tuned   = f1_score(eval_labels, pred_tuned,   average="macro", zero_division=0)

        f1_default_per = f1_score(eval_labels, pred_default, average=None,
                                  labels=list(range(n_classes)), zero_division=0)
        f1_tuned_per   = f1_score(eval_labels, pred_tuned,   average=None,
                                  labels=list(range(n_classes)), zero_division=0)

        # Prediction count per class
        counts_default = np.bincount(pred_default, minlength=n_classes)
        counts_tuned   = np.bincount(pred_tuned,   minlength=n_classes)

        fold_results.append({
            "fold":               k,
            "f1_default":         f1_default,
            "f1_tuned":           f1_tuned,
            "delta_f1":           f1_tuned - f1_default,
            "spirals_f1_default": float(f1_default_per[3]),
            "spirals_f1_tuned":   float(f1_tuned_per[3]),
            "f1_default_per":     f1_default_per.tolist(),
            "f1_tuned_per":       f1_tuned_per.tolist(),
            "counts_default":     counts_default.tolist(),
            "counts_tuned":       counts_tuned.tolist(),
        })

    # Print fold-level table
    print(f"\n{'Fold':<6} {'F1_default':>10} {'F1_tuned':>10} {'ΔF1':>8} "
          f"{'Spirals_def':>12} {'Spirals_tuned':>14}")
    print("-" * 64)
    for r in fold_results:
        print(f"{r['fold']:<6} {r['f1_default']:10.4f} {r['f1_tuned']:10.4f} "
              f"{r['delta_f1']:+8.4f} {r['spirals_f1_default']:12.4f} "
              f"{r['spirals_f1_tuned']:14.4f}")

    f1_def_arr  = np.array([r["f1_default"]  for r in fold_results])
    f1_tun_arr  = np.array([r["f1_tuned"]    for r in fold_results])
    delta_arr   = np.array([r["delta_f1"]    for r in fold_results])
    sp_def_arr  = np.array([r["spirals_f1_default"] for r in fold_results])
    sp_tun_arr  = np.array([r["spirals_f1_tuned"]   for r in fold_results])

    print("-" * 64)
    print(f"{'Mean':<6} {f1_def_arr.mean():10.4f} {f1_tun_arr.mean():10.4f} "
          f"{delta_arr.mean():+8.4f} {sp_def_arr.mean():12.4f} {sp_tun_arr.mean():14.4f}")
    print(f"{'Std':<6} {f1_def_arr.std():10.4f} {f1_tun_arr.std():10.4f} "
          f"{delta_arr.std():+8.4f} {sp_def_arr.std():12.4f} {sp_tun_arr.std():14.4f}")

    # Per-class F1 summary
    class_names = ["FRI", "FRII", "Hybrids", "Spirals", "Relaxed"]
    print("\nPer-class F1 (mean across folds):  default → tuned  (Δ)")
    for c, cn in enumerate(class_names):
        def_vals = np.array([r["f1_default_per"][c] for r in fold_results])
        tun_vals = np.array([r["f1_tuned_per"][c]   for r in fold_results])
        delta    = tun_vals.mean() - def_vals.mean()
        print(f"  {cn:<17} {def_vals.mean():.4f} → {tun_vals.mean():.4f}  ({delta:+.4f})")

    # Prediction count comparison
    print("\nPrediction counts (mean across folds):  default | tuned")
    for c, cn in enumerate(class_names):
        cnt_def = np.mean([r["counts_default"][c] for r in fold_results])
        cnt_tun = np.mean([r["counts_tuned"][c]   for r in fold_results])
        print(f"  {cn:<17} {cnt_def:6.1f} | {cnt_tun:6.1f}")

    # Tuned thresholds per fold
    print("\nTuned thresholds per fold:")
    print(f"  {'Fold':<6} " + " ".join(f"{cn:>9}" for cn in class_names))
    print("  " + "-" * (6 + 10 * n_classes))
    for k, thr in enumerate(all_tuned_thresholds):
        print(f"  {k:<6} " + " ".join(f"{thr[c]:9.3f}" for c in range(n_classes)))

    # Final conclusion
    NOISE_FLOOR = 0.029
    mean_delta  = delta_arr.mean()
    print("\n" + "=" * 70)
    if mean_delta >= NOISE_FLOOR:
        print(f"THRESHOLD TUNING SUFFICIENT — mean ΔF1 = {mean_delta:+.4f} >= {NOISE_FLOOR}")
        print("Consider stopping here (no new classifiers needed for F1 improvement).")
    else:
        print(f"Threshold tuning within noise floor: mean ΔF1 = {mean_delta:+.4f} < {NOISE_FLOOR}")
        print("Effect is not reliably above noise; additional classifiers may be needed.")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Threshold diagnostic: cwNone vs cwinitial_pure and LOFO threshold tuning.")
    parser.add_argument("--outputs-root", default="outputs/byol_runs",
                        help="Root containing run subdirs.")
    parser.add_argument("--run-name",     required=True,
                        help="Run directory name, e.g. pd128_qext_v1_wd1e-3_lrconst_sw0.1_f1.")
    parser.add_argument("--data-seed",    type=int, default=2,
                        help="Data split seed (default: 2).")
    parser.add_argument("--seed",         type=int, default=2,
                        help="Training seed / run seed (default: 2).")
    parser.add_argument("--n-folds",      type=int, default=5,
                        help="Number of CV folds (default: 5).")
    args = parser.parse_args()

    run_dir = Path(args.outputs_root) / args.run_name
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    part1_auc_f1_table(run_dir, args.data_seed, args.seed, args.n_folds)
    part2_threshold_tuning(run_dir, args.data_seed, args.seed, args.n_folds)


if __name__ == "__main__":
    main()
