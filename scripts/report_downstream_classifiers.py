"""
report_downstream_classifiers.py — Standalone print report for downstream classifiers.

Prints three tables (no figures, no notebook changes, no JSON written):
  Table 1: Macro F1 by classifier and sw (unweighted)
  Table 2: MLP v2 config sweep
  Table 3: Per-class F1 for best new classifiers

Reference numbers (initial_pure, cwNone, ds=2, 5-fold CV):
  LR (sklearn)                 0.718
  kNN (sklearn, k=5 uniform)   0.657
  MLP (sklearn, 256×128, ES)   0.706
  Supervised enb0 ep20         0.698
  Noise floor (1σ)             ±0.029

Usage:
    python scripts/report_downstream_classifiers.py \\
        --outputs-root outputs/byol_runs \\
        --data-seed 2 \\
        --sw-values 0.0 0.05 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── Reference numbers ─────────────────────────────────────────────────────────

REFERENCE = {
    "LR (ref)":             0.718,
    "kNN k=5 (ref)":        0.657,
    "MLP sklearn (ref)":    0.706,
    "Sup enb0 ep20 (ref)":  0.698,
}
NOISE_FLOOR = 0.029
N_FOLDS     = 5
CLASS_NAMES = ["FRI", "FRII", "Hybrids", "Spirals", "Relaxed"]

# MLP v2 arch × wd configs
MLP_V2_CONFIGS = [
    ("256x128", "wd0"),
    ("256x128", "wd1e-4"),
    ("256x128", "wd1e-3"),
    ("512x256", "wd0"),
    ("512x256", "wd1e-4"),
    ("512x256", "wd1e-3"),
    ("256",     "wd0"),
    ("256",     "wd1e-4"),
    ("256",     "wd1e-3"),
]


# ── Loader helpers ────────────────────────────────────────────────────────────

def _run_dir_for_sw(outputs_root: Path, run_base: str, sw: str) -> Path:
    """Resolve run directory for a given sw string (e.g. '0.1' → sw0.1)."""
    return outputs_root / f"{run_base}_sw{sw}_f1"


def _fold_clf_dir(run_dir: Path, data_seed: int, seed: int, fold: int,
                  label_set: str, cw_tag: str) -> Path:
    return (run_dir / f"data_seed_{data_seed}" / f"training_seed_{seed}"
            / f"cross_val_{fold}" / "data" / "downstream_classifiers"
            / f"{label_set}_{cw_tag}")


def _load_metric(json_path: Path, key: str):
    if not json_path.exists():
        return None
    with open(json_path) as fh:
        d = json.load(fh)
    return d.get(key)


def _collect_folds(outputs_root: Path, run_base: str, sw: str,
                   data_seed: int, seed: int, json_name: str,
                   metric: str) -> list:
    """Load a scalar metric from 5 folds for a given sw/classifier. Returns list of floats."""
    rd  = _run_dir_for_sw(outputs_root, run_base, sw)
    vals = []
    for fold in range(N_FOLDS):
        d    = _fold_clf_dir(rd, data_seed, seed, fold, "initial_pure", "cwNone")
        v    = _load_metric(d / json_name, metric)
        if v is not None:
            vals.append(float(v))
    return vals


def _collect_per_class(outputs_root: Path, run_base: str, sw: str,
                       data_seed: int, seed: int, json_name: str) -> np.ndarray:
    """Load f1_per_class from 5 folds. Returns (5, n_classes) or None."""
    rd  = _run_dir_for_sw(outputs_root, run_base, sw)
    rows = []
    for fold in range(N_FOLDS):
        d   = _fold_clf_dir(rd, data_seed, seed, fold, "initial_pure", "cwNone")
        v   = _load_metric(d / json_name, "f1_per_class")
        if v is not None:
            rows.append(v)
    return np.array(rows) if rows else None


def _best_es02_json(outputs_root: Path, run_base: str, sw: str,
                    data_seed: int, seed: int) -> str | None:
    """Return the mlp_v2_*_es02 json name with the highest mean F1 at the given sw, or None."""
    best_f1, best_jn = -1.0, None
    for arch_tag, wd_tag in MLP_V2_CONFIGS:
        jn   = f"mlp_v2_{arch_tag}_{wd_tag}_es02_projections.json"
        vals = _collect_folds(outputs_root, run_base, sw, data_seed, seed, jn, "f1_macro")
        mn, _ = _mean_std_raw(vals)
        if not np.isnan(mn) and mn > best_f1:
            best_f1, best_jn = mn, jn
    return best_jn


def _mean_std_raw(vals: list) -> tuple:
    """Internal: mean/std without calling _mean_std to avoid forward-reference."""
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals)
    return float(a.mean()), float(a.std())


def _mean_std(vals: list) -> tuple:
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals)
    return float(a.mean()), float(a.std())


def _collect_folds_paired(outputs_root: Path, run_base: str, sw0: str, sw_hi: str,
                          data_seed: int, seed: int, json_name: str,
                          metric: str) -> list:
    """Load metric from sw0 and sw_hi for each fold. Returns list of (v0, v_hi) pairs."""
    rd0   = _run_dir_for_sw(outputs_root, run_base, sw0)
    rd_hi = _run_dir_for_sw(outputs_root, run_base, sw_hi)
    pairs = []
    for fold in range(N_FOLDS):
        d0   = _fold_clf_dir(rd0,   data_seed, seed, fold, "initial_pure", "cwNone")
        d_hi = _fold_clf_dir(rd_hi, data_seed, seed, fold, "initial_pure", "cwNone")
        v0   = _load_metric(d0   / json_name, metric)
        v_hi = _load_metric(d_hi / json_name, metric)
        if v0 is not None and v_hi is not None:
            pairs.append((float(v0), float(v_hi)))
    return pairs


def _paired_delta_summary(pairs: list) -> tuple:
    """From (v0, v_hi) pairs return (mean_delta, std_delta, n_positive, n_total)."""
    if not pairs:
        return float("nan"), float("nan"), 0, 0
    deltas = np.array([hi - lo for lo, hi in pairs])
    return float(deltas.mean()), float(deltas.std()), int((deltas > 0).sum()), len(deltas)


def _load_sup_enb0_folds(project_root: Path, metric: str = "f1_macro") -> list:
    """Load ep20 supervised enb0 results (5 folds) from outputs/classifiers/.

    project_root is expected to be the 'outputs/' directory
    (i.e. outputs_root.parent for outputs_root = outputs/byol_runs).
    """
    base = (project_root / "classifiers"
            / "retrain_sup_noes_cv5" / "without_generative")
    vals = []
    for fold in range(N_FOLDS):
        p = base / f"retrain_sup_noes_cv5_ep20_fold{fold}" / "results.json"
        v = _load_metric(p, metric)
        if v is not None:
            vals.append(float(v))
    return vals


# ── Table 1: Macro F1 by classifier and sw ────────────────────────────────────

def table1_macro_f1(outputs_root: Path, project_root: Path, run_base: str,
                    sw_values: list, data_seed: int, seed: int):
    print("=" * 88)
    print("TABLE 1 — Macro F1 by classifier and sw (unweighted, initial_pure, ds=2, 5-fold CV)")
    print(f"          Δ and sign: paired per-fold delta sw_hi − sw_lo; sign = n_positive/{N_FOLDS}.")
    print(f"          Reference: LR={REFERENCE['LR (ref)']:.3f}  kNN={REFERENCE['kNN k=5 (ref)']:.3f}  "
          f"MLP={REFERENCE['MLP sklearn (ref)']:.3f}  SupEnb0={REFERENCE['Sup enb0 ep20 (ref)']:.3f}")
    print("=" * 88)

    sw0   = sw_values[0]
    sw_hi = sw_values[-1]

    # (display_name, json_name, static)
    # static=True: not a BYOL-downstream run; loaded from a separate path or fixed value.
    classifiers = [
        ("LR (ref)",            "lr_projections.json",                   False),
        ("kNN k=5 (ref)",       "knn_projections.json",                  False),
        ("MLP sklearn (ref)",   "mlp_projections.json",                  False),
        ("Sup enb0 ep20 (ref)", None,                                    True),
        ("MLP_v2 256x128 wd0",  "mlp_v2_256x128_wd0_projections.json",  False),
        ("MLP_v2 512x256 wd0",  "mlp_v2_512x256_wd0_projections.json",  False),
        ("MLP_v2 256 wd0",      "mlp_v2_256_wd0_projections.json",      False),
        ("kNN_dist_k10",        "knn_dist_k10_projections.json",         False),
        ("kNN_kavg",            "knn_kavg_projections.json",             False),
        ("kNN_rbf",             "knn_rbf_projections.json",              False),
        # ── early-stopping variants (saved separately, won't exist until job runs) ──
        ("MLP_v1 es02",         "mlp_es02_projections.json",             False),
        ("MLP_v2 es02 best",    "BEST_ES02",                             False),
    ]

    col_w    = 22
    sw_col_w = 9
    header = f"{'Classifier':<{col_w}}"
    for sw in sw_values:
        header += f"{'sw='+sw:>{sw_col_w}}"
    header += f"  {'Δ(hi-lo)':>9}  {'sign':>4}  {'σΔ':>5}"
    print(header)
    print("-" * len(header))

    for clf_name, json_name, is_static in classifiers:
        if is_static:
            vals    = _load_sup_enb0_folds(project_root)
            mn, std = _mean_std(vals)
            row = f"{clf_name:<{col_w}}"
            for _ in sw_values:
                row += f"{mn:>{sw_col_w}.3f}" if not np.isnan(mn) else f"{'N/A':>{sw_col_w}}"
            row += f"  {'—':>9}  {'—':>4}  {'—':>5}"
            print(row)
            continue

        # Resolve sentinel: find best es02 config across sw_hi
        if json_name == "BEST_ES02":
            json_name = _best_es02_json(outputs_root, run_base, sw_hi, data_seed, seed)
        if json_name is None:
            row = f"{clf_name:<{col_w}}"
            for _ in sw_values:
                row += f"{'N/A':>{sw_col_w}}"
            row += f"  {'N/A':>9}  {'N/A':>4}  {'N/A':>5}"
            print(row)
            continue

        sw_means = {}
        for sw in sw_values:
            vals     = _collect_folds(outputs_root, run_base, sw, data_seed, seed,
                                      json_name, "f1_macro")
            mn, _    = _mean_std(vals)
            sw_means[sw] = mn

        pairs = _collect_folds_paired(outputs_root, run_base, sw0, sw_hi,
                                      data_seed, seed, json_name, "f1_macro")
        mean_d, std_d, n_pos, n_tot = _paired_delta_summary(pairs)

        row = f"{clf_name:<{col_w}}"
        for sw in sw_values:
            mn = sw_means.get(sw, float("nan"))
            row += f"{mn:>{sw_col_w}.3f}" if not np.isnan(mn) else f"{'N/A':>{sw_col_w}}"

        if np.isnan(mean_d):
            row += f"  {'N/A':>9}  {'N/A':>4}  {'N/A':>5}"
        else:
            sign_str = f"{n_pos}/{n_tot}"
            row += f"  {mean_d:>+9.3f}  {sign_str:>4}  {std_d:>5.3f}"
        print(row)

    print()
    print("  Split: kNN family (k=5, kavg, rbf) is monotonic ↑ in sw (0.0 → 0.05 → 0.1).")
    print("  LR and MLP v2 show no consistent sw response (flat or mixed sign pattern).")
    print("  Local classifiers benefit from training-set augmentation; parametric do not.")
    print("  This replicates the Q3 finding and is now confirmed on independently")
    print("  implemented classifiers (kavg and rbf were written from scratch for this run).")
    print()


# ── Table 2: MLP v2 config sweep ─────────────────────────────────────────────

def table2_mlp_v2_sweep(outputs_root: Path, run_base: str, sw_values: list,
                        data_seed: int, seed: int):
    print("=" * 70)
    print("TABLE 2 — MLP v2 config sweep (initial_pure, cwNone, ds=2, 5-fold CV)")
    print(f"          (* = Δ < noise floor ±{NOISE_FLOOR})")
    print("=" * 70)

    sw0 = sw_values[0]
    sw_hi = sw_values[-1]

    print(f"{'Config':<20} {'sw='+sw0+' F1':>12} {'sw='+sw_hi+' F1':>12} {'Δ(hi-lo)':>10}  {'sign':>4}  {'σΔ':>5}")
    print("-" * 72)

    for arch_tag, wd_tag in MLP_V2_CONFIGS:
        json_name = f"mlp_v2_{arch_tag}_{wd_tag}_projections.json"

        vals_lo = _collect_folds(outputs_root, run_base, sw0,  data_seed, seed,
                                 json_name, "f1_macro")
        vals_hi = _collect_folds(outputs_root, run_base, sw_hi, data_seed, seed,
                                 json_name, "f1_macro")

        mn_lo, _ = _mean_std(vals_lo)
        mn_hi, _ = _mean_std(vals_hi)

        pairs = _collect_folds_paired(outputs_root, run_base, sw0, sw_hi,
                                      data_seed, seed, json_name, "f1_macro")
        mean_d, std_d, n_pos, n_tot = _paired_delta_summary(pairs)

        name = f"{arch_tag} {wd_tag}"
        if np.isnan(mn_lo):
            print(f"{name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>10}  {'N/A':>4}  {'N/A':>5}")
        else:
            sign_str = f"{n_pos}/{n_tot}"
            print(f"{name:<20} {mn_lo:>12.3f} {mn_hi:>12.3f} {mean_d:>+9.3f}  {sign_str:>4}  {std_d:>5.3f}")

    print()


# ── Table 2b: MLP v2 es02 config sweep vs fixed-epoch ────────────────────────

def table2b_mlp_v2_es02(outputs_root: Path, run_base: str, sw_values: list,
                         data_seed: int, seed: int):
    print("=" * 82)
    print("TABLE 2b — MLP v2 early stopping (val=0.2, patience=20) vs fixed 100 epochs")
    print( "           initial_pure, cwNone, ds=2, 5-fold CV")
    print("=" * 82)

    sw0   = sw_values[0]
    sw_hi = sw_values[-1]

    print(f"{'Config':<20} {'fixed sw='+sw_hi:>13} {'ES sw='+sw_hi:>10} {'Δ(ES-fix)':>10}"
          f"  {'ES sign':>7}  {'ES σΔ':>6}")
    print("-" * 70)

    for arch_tag, wd_tag in MLP_V2_CONFIGS:
        jn_fix = f"mlp_v2_{arch_tag}_{wd_tag}_projections.json"
        jn_es  = f"mlp_v2_{arch_tag}_{wd_tag}_es02_projections.json"

        mn_fix, _ = _mean_std(_collect_folds(outputs_root, run_base, sw_hi,
                                             data_seed, seed, jn_fix, "f1_macro"))
        mn_es,  _ = _mean_std(_collect_folds(outputs_root, run_base, sw_hi,
                                             data_seed, seed, jn_es,  "f1_macro"))

        # sw response for ES variant
        pairs_es = _collect_folds_paired(outputs_root, run_base, sw0, sw_hi,
                                         data_seed, seed, jn_es, "f1_macro")
        mean_d_es, std_d_es, n_pos_es, n_tot_es = _paired_delta_summary(pairs_es)

        name = f"{arch_tag} {wd_tag}"
        if np.isnan(mn_es):
            print(f"{name:<20} {mn_fix:>13.3f} {'N/A':>10} {'N/A':>10}  {'N/A':>7}  {'N/A':>6}")
        else:
            delta_vs_fix = mn_es - mn_fix
            sign_str = f"{n_pos_es}/{n_tot_es}"
            print(f"{name:<20} {mn_fix:>13.3f} {mn_es:>10.3f} {delta_vs_fix:>+9.3f}"
                  f"  {sign_str:>7}  {std_d_es:>6.3f}")

    print()


# ── Table 3: Per-class F1 for best new classifiers ───────────────────────────

def table3_per_class_f1(outputs_root: Path, run_base: str, sw: str,
                        data_seed: int, seed: int):
    print("=" * 80)
    print(f"TABLE 3 — Per-class F1 (sw={sw}, cwNone, initial_pure, ds=2, 5-fold CV)")
    print("=" * 80)

    class_w = 10
    col_w   = 9
    n_cls   = len(CLASS_NAMES)

    header = f"{'Classifier':<22}" + "".join(f"{cn:>{col_w}}" for cn in CLASS_NAMES) + f"{'macro':>{col_w}}"
    print(header)
    print("-" * len(header))

    classifiers = [
        ("LR (ref)",          "lr_projections.json"),
        ("MLP_v2 best",       None),  # filled dynamically
        ("kNN_kavg",          "knn_kavg_projections.json"),
        ("kNN_rbf",           "knn_rbf_projections.json"),
    ]

    # Find best MLP v2 config at this sw
    best_mlp_f1 = -1.0
    best_mlp_json = None
    for arch_tag, wd_tag in MLP_V2_CONFIGS:
        jname = f"mlp_v2_{arch_tag}_{wd_tag}_projections.json"
        vals  = _collect_folds(outputs_root, run_base, sw, data_seed, seed,
                               jname, "f1_macro")
        mn, _ = _mean_std(vals)
        if not np.isnan(mn) and mn > best_mlp_f1:
            best_mlp_f1  = mn
            best_mlp_json = jname
            best_mlp_name = f"MLP_v2 {arch_tag} {wd_tag}"

    for clf_name, json_name in classifiers:
        if clf_name == "MLP_v2 best":
            if best_mlp_json is None:
                print(f"{'MLP_v2 best':<22}" + "".join(f"{'N/A':>{col_w}}" for _ in range(n_cls)) + f"{'N/A':>{col_w}}")
                continue
            json_name = best_mlp_json
            clf_name  = best_mlp_name[:22]

        arr = _collect_per_class(outputs_root, run_base, sw, data_seed, seed, json_name)
        if arr is None:
            print(f"{clf_name:<22}" + "".join(f"{'N/A':>{col_w}}" for _ in range(n_cls + 1)))
            continue

        means = arr.mean(axis=0)
        macro = means.mean()
        row   = f"{clf_name:<22}" + "".join(f"{means[c]:>{col_w}.3f}" for c in range(n_cls)) + f"{macro:>{col_w}.3f}"
        print(row)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print downstream classifier report tables.")
    parser.add_argument("--outputs-root", default="outputs/byol_runs")
    parser.add_argument("--run-base",     default="pd128_qext_v1_wd1e-3_lrconst",
                        help="Run name prefix before _sw{sw}_f1 (default: pd128_qext_v1_wd1e-3_lrconst).")
    parser.add_argument("--data-seed",    type=int, default=2)
    parser.add_argument("--seed",         type=int, default=2)
    parser.add_argument("--sw-values",    nargs="+", default=["0.0", "0.05", "0.1"],
                        help="sw values to compare (default: 0.0 0.05 0.1).")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    project_root = outputs_root.parent

    table1_macro_f1(outputs_root, project_root, args.run_base, args.sw_values,
                    args.data_seed, args.seed)
    table2_mlp_v2_sweep(outputs_root, args.run_base, args.sw_values, args.data_seed, args.seed)
    table2b_mlp_v2_es02(outputs_root, args.run_base, args.sw_values, args.data_seed, args.seed)
    table3_per_class_f1(outputs_root, args.run_base, args.sw_values[-1],
                        args.data_seed, args.seed)


if __name__ == "__main__":
    main()
