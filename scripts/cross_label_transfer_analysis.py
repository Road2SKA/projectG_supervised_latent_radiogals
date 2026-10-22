"""
cross_label_transfer_analysis.py — Cross-label retrieval transfer test.

Tests whether sup-initial supervision generalises beyond the label axis it
optimised (cols 0-4: initial) to the axes it did NOT (cols 5-19: morphology +
environment).

Metric: nDCG@10 on test-set retrieval.  For each query in the test set its 10
nearest neighbours are retrieved by cosine similarity (dot product on
L2-normalised projections) and nDCG is computed with shared-label count as the
relevance score.  Queries with no labels in the evaluated subset are excluded.

Comparison: sup-initial sw ∈ {0.05, 0.1, 0.5} vs BYOL sw=0.0, paired per data
seed (ds=2..6).  The per-seed value is the mean nDCG@10 over the 5 CV folds.
Paired delta = sup_value − sw0_value per seed.

Both families read test_projections.npy from the same cross_val_K directory and
use the same test_idx.npy from data_splits/<ds>/cross_val_K/.
n_test is printed per fold to confirm the split is shared.

KEY CONSTRAINT: if the positive control (Table 2, cols 0-4) shows no
improvement for any sw value, the cross-label result (Table 1) is stated as
uninterpretable and no further inference is drawn.

Usage:
    source .venv/bin/activate
    python scripts/cross_label_transfer_analysis.py [--outputs-root ...] [--splits-root ...]
"""

import argparse
from pathlib import Path

import numpy as np


# ── constants ─────────────────────────────────────────────────────────────────

DATA_SEEDS   = [2, 3, 4, 5, 6]
N_FOLDS      = 5
K            = 10

SW0_RUN_BASE = "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1"
SUP_SW_VALUES = ["0.05", "0.1", "0.5"]
SUP_RUN_BASE  = "pd128_qext_v1_wd1e-3_lrconst_sup-initial_sw{sw}_f1"

INITIAL_COLS = list(range(0, 5))   # cols 0-4: positive control
CROSS_COLS   = list(range(5, 20))  # cols 5-19: morphology + environment (transfer axis)


# ── retrieval metric ─────────────────────────────────────────────────────────

def ndcg_at_k_mean(proj: np.ndarray, labels_subset: np.ndarray, k: int = 10):
    """
    Mean nDCG@K over valid queries (those with ≥1 positive label in subset).

    proj: (N, D) raw BYOL projections (NOT assumed to be L2-normalised;
          this function normalises them internally before computing cosine sim).
    labels_subset: (N, C) binary multi-label for the subset being evaluated.
    Returns (mean_ndcg, n_valid_queries).
    """
    N = len(proj)
    # L2-normalise so dot product = cosine similarity
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    proj_n = proj / norms

    # Cosine similarity
    sim = proj_n @ proj_n.T          # (N, N)
    np.fill_diagonal(sim, -np.inf)   # exclude self-retrieval

    # Shared-label relevance matrix (integer counts)
    rel_matrix = (labels_subset.astype(np.float32) @
                  labels_subset.astype(np.float32).T).astype(np.int32)  # (N, N)

    # Top-K retrieved indices for all queries at once
    top_k_idx = np.argsort(-sim, axis=1)[:, :k]   # (N, K)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # 1/log2(2), 1/log2(3), …
    ndcg_vals = []

    for i in range(N):
        if labels_subset[i].sum() == 0:
            continue  # no labels in this subset for this query

        # DCG@K
        retrieved_rel = rel_matrix[i, top_k_idx[i]].astype(np.float64)
        dcg = float(np.dot(retrieved_rel, discounts))

        # IDCG@K: ideal ordering over the gallery (self excluded)
        all_rel = rel_matrix[i].astype(np.float64)
        all_rel[i] = 0
        ideal_k   = min(k, N - 1)
        # Use partition for speed on large N
        if ideal_k < len(all_rel):
            ideal_rel = np.partition(all_rel, -ideal_k)[-ideal_k:]
        else:
            ideal_rel = all_rel
        ideal_rel = np.sort(ideal_rel)[::-1]
        idcg = float(np.dot(ideal_rel, discounts[:len(ideal_rel)]))

        if idcg == 0:
            continue  # no relevant item exists in gallery for this query

        ndcg_vals.append(dcg / idcg)

    if not ndcg_vals:
        return float("nan"), 0
    return float(np.mean(ndcg_vals)), len(ndcg_vals)


# ── data loading ─────────────────────────────────────────────────────────────

def load_proj(outputs_root: Path, run_name: str, ds: int, fold: int) -> np.ndarray:
    p = (outputs_root / run_name / f"data_seed_{ds}" / f"training_seed_{ds}"
         / f"cross_val_{fold}" / "data" / "byol" / "test_projections.npy")
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return np.load(p)


def load_test_labels_and_idx(splits_root: Path, ds: int, fold: int):
    d = splits_root / str(ds) / f"cross_val_{fold}"
    labels  = np.load(d / "test_labels.npy")
    idx     = np.load(d / "test_idx.npy")
    return labels, idx


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs/byol_runs",
                        help="Root containing byol run directories.")
    parser.add_argument("--splits-root",  default="outputs/data_splits",
                        help="Root containing data_splits/<ds>/cross_val_K/.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    splits_root  = Path(args.splits_root)

    # ── accumulate nDCG per (run, ds, fold, col_tag) ──────────────────────────
    # structure: results[run_label][(ds, fold)][col_tag] = ndcg_float
    run_labels = ["sw=0.0"] + [f"sw={sw}" for sw in SUP_SW_VALUES]
    run_names  = {
        "sw=0.0": SW0_RUN_BASE,
        **{f"sw={sw}": SUP_RUN_BASE.format(sw=sw) for sw in SUP_SW_VALUES},
    }

    results = {rl: {} for rl in run_labels}
    split_ok_printed = False  # print split-match confirmation once

    print("Verifying test split alignment and computing nDCG@10 …")
    print()

    for ds in DATA_SEEDS:
        for fold in range(N_FOLDS):
            labels, test_idx = load_test_labels_and_idx(splits_root, ds, fold)
            n_test = len(test_idx)

            # Print first occurrence of each (ds, fold) to confirm n_test
            if not split_ok_printed or fold == 0:
                n_cross   = int((labels[:, CROSS_COLS].sum(axis=1) > 0).sum())
                n_initial = int((labels[:, INITIAL_COLS].sum(axis=1) > 0).sum())
                print(f"  ds={ds} fold={fold}: n_test={n_test:4d}  "
                      f"cross_queries={n_cross}  initial_queries={n_initial}")
            split_ok_printed = True

            for rl in run_labels:
                proj = load_proj(outputs_root, run_names[rl], ds, fold)
                assert len(proj) == n_test, (
                    f"Projection length {len(proj)} ≠ n_test {n_test} "
                    f"for {rl} ds={ds} fold={fold}")

                ndcg_cross,   _ = ndcg_at_k_mean(proj, labels[:, CROSS_COLS],   K)
                ndcg_initial, _ = ndcg_at_k_mean(proj, labels[:, INITIAL_COLS], K)

                results[rl][(ds, fold)] = {
                    "cross":   ndcg_cross,
                    "initial": ndcg_initial,
                }

    print()
    print("=" * 72)
    print("Split alignment confirmed: both families read from the same")
    print("data_splits/<ds>/cross_val_K/ test_idx.npy.  n_test printed above.")
    print("=" * 72)
    print()

    # ── aggregate per (run, ds): mean over 5 folds ────────────────────────────
    # ds_vals[rl][ds][col_tag] = mean nDCG over folds
    def ds_mean(rl, ds, col_tag):
        vals = [results[rl][(ds, fold)][col_tag]
                for fold in range(N_FOLDS)
                if (ds, fold) in results[rl]]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    # ── paired delta summary ──────────────────────────────────────────────────
    def paired_summary(sw_label, col_tag):
        """Returns (per_seed_sup, per_seed_sw0, deltas, mean_d, n_pos)."""
        sup_vals  = [ds_mean(sw_label, ds, col_tag) for ds in DATA_SEEDS]
        sw0_vals  = [ds_mean("sw=0.0",  ds, col_tag) for ds in DATA_SEEDS]
        deltas    = [s - b for s, b in zip(sup_vals, sw0_vals)
                     if not (np.isnan(s) or np.isnan(b))]
        mean_d    = float(np.mean(deltas)) if deltas else float("nan")
        n_pos     = sum(1 for d in deltas if d > 0)
        return sup_vals, sw0_vals, deltas, mean_d, n_pos

    # ── print tables ─────────────────────────────────────────────────────────

    def print_table(col_tag, col_desc, col_range_str):
        ds_header = "  ".join(f"ds={ds}" for ds in DATA_SEEDS)
        print("=" * 78)
        print(f"TABLE — nDCG@{K} retrieval  |  {col_desc}  ({col_range_str})")
        print(f"        Paired per data seed; each seed value = mean over 5 folds.")
        print("=" * 78)
        print(f"{'Classifier':<18}  {ds_header}    mean    Δ(sup-sw0)  sign   σΔ")
        print("-" * 78)

        # sw=0.0 baseline row
        sw0_vals = [ds_mean("sw=0.0", ds, col_tag) for ds in DATA_SEEDS]
        row = f"{'sw=0.0 (base)':<18}  "
        row += "  ".join(f"{v:.4f}" for v in sw0_vals)
        row += f"  {np.nanmean(sw0_vals):.4f}   {'—':>10}  {'—':>4}  {'—':>5}"
        print(row)

        for sw in SUP_SW_VALUES:
            rl = f"sw={sw}"
            sup_vals, _, deltas, mean_d, n_pos = paired_summary(rl, col_tag)
            std_d = float(np.std(deltas)) if deltas else float("nan")
            mean_sup = float(np.nanmean(sup_vals))

            row = f"{'sup-init ' + rl:<18}  "
            row += "  ".join(f"{v:.4f}" for v in sup_vals)
            sign_str = f"{n_pos}/{len(deltas)}"
            row += f"  {mean_sup:.4f}  {mean_d:>+10.4f}  {sign_str:>4}  {std_d:>5.4f}"
            print(row)

        print()

    print_table(
        col_tag="cross",
        col_desc="Cross-label transfer axis",
        col_range_str="cols 5–19: morphology + environment",
    )
    print_table(
        col_tag="initial",
        col_desc="Positive control (optimised axis)",
        col_range_str="cols 0–4: FRI/FRII/Hybrids/Spirals/Relaxed",
    )

    # ── interpret positive control ────────────────────────────────────────────
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    ctrl_any_positive = False
    for sw in SUP_SW_VALUES:
        _, _, _, mean_d, n_pos = paired_summary(f"sw={sw}", "initial")
        if n_pos >= 4 or mean_d > 0.005:
            ctrl_any_positive = True

    if not ctrl_any_positive:
        print("WARNING: Positive control FAILED — supervision did not improve")
        print("retrieval on the initial-label axis (cols 0-4) for any sw value.")
        print("Cross-label result (Table 1) is UNINTERPRETABLE: if the runs")
        print("did not learn the supervised axis, transfer cannot be assessed.")
    else:
        print("Positive control PASSED for at least one sw value.")
        print("Cross-label result is interpretable.")
        cross_any = False
        for sw in SUP_SW_VALUES:
            _, _, _, mean_d, n_pos = paired_summary(f"sw={sw}", "cross")
            rl = f"sw={sw}"
            if n_pos >= 4 or mean_d > 0.005:
                cross_any = True
                print(f"  sw={sw}: cross-label nDCG@10 Δ={mean_d:+.4f}  sign={n_pos}/{len(DATA_SEEDS)}")
        if not cross_any:
            print("Cross-label transfer: no consistent improvement detected.")
            print("Supervision improved only the optimised axis (no positive transfer).")
    print()


if __name__ == "__main__":
    main()
