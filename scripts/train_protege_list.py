"""
run_protege_sweep.py — Protege GP active-learning sweep over BYOL run directories.

For each run directory matching --run-glob under --outputs-root:
  1. Load labelled + unlabelled train projections and indices.
  2. Build source names, labels (tier-scored human_label 1–5), and PCA features.
  3. Seed anomaly_scores from the labelled training set, run ScoreConverter.
  4. Run GP active learning (Protege) querying the entire unlabelled pool.
  5. Compute recall-curve AUC over the unqueried eval set.
  6. Save recall_curve, protege_scores.parquet, and protege_summary.json.

After all runs, print a summary table sorted by AUC descending.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from astronomaly.anomaly_detection import human_loop_learning, protege

# ---------------------------------------------------------------------------
# Hardcoded paths and constants
# ---------------------------------------------------------------------------
CSV_PATH    = Path("/users/mbredber/p3_SUPLAT/data/metadata/lotss_classifications_horton_et_al_2025_filtered.csv")
LABELS_PATH = Path("/users/mbredber/p3_SUPLAT/data/preprocessed/lotss/labels_filtered.npy")

LABEL_COLS = [
    "fri", "frii", "hybrid", "spiral", "relaxed",
    "cshaped", "sshaped", "misaligned", "wings", "xshaped",
    "straight", "multihotspots", "continuous", "banding", "onesided",
    "restarted", "cluster", "merger", "diffuse", "unknown",
]

SCORE_5 = ["xshaped", "unknown"]
SCORE_4 = ["cluster", "merger", "diffuse", "sshaped"]
SCORE_3 = ["restarted", "onesided", "banding", "cshaped", "wings", "misaligned", "spiral", "relaxed"]
SCORE_2 = ["fri", "frii", "hybrid", "straight", "multihotspots", "continuous"]
TIERS   = [(5, SCORE_5), (4, SCORE_4), (3, SCORE_3), (2, SCORE_2)]


# ---------------------------------------------------------------------------
# Active-learning function (verbatim from notebook)
# ---------------------------------------------------------------------------
def run_GP_active_learning(features, labels, input_anomaly_scores, output_dir,
                           steps=10, initial_steps=None, N_labels=100, epsilon=0.5):
    anomaly_scores = input_anomaly_scores.copy()
    anomaly_scores['human_label'] = [-1] * len(anomaly_scores)
    pipeline_active_learning = protege.GaussianProcess(
        features, force_rerun=True, output_dir=output_dir, ei_tradeoff=epsilon)
    if initial_steps is not None:
        anomaly_scores.sort_values('score', ascending=False, inplace=True)
        inds = anomaly_scores[anomaly_scores.human_label == -1].index[:initial_steps]
        anomaly_scores.loc[inds, 'human_label'] = labels.loc[inds, 'human_label']
        features_with_labels = pipeline_active_learning.combine_data_frames(features, anomaly_scores)
        active_output = pipeline_active_learning.run(features_with_labels)
        anomaly_scores['trained_score'] = active_output['trained_score']
        anomaly_scores['acquisition']   = active_output['acquisition']
    else:
        initial_steps = 0
    for i in range(initial_steps // steps, N_labels // steps):
        anomaly_scores.sort_values('acquisition', ascending=False, inplace=True)
        inds = anomaly_scores[anomaly_scores.human_label == -1].index[:steps]
        anomaly_scores.loc[inds, 'human_label'] = labels.loc[inds, 'human_label']
        features_with_labels = pipeline_active_learning.combine_data_frames(features, anomaly_scores)
        active_output = pipeline_active_learning.run(features_with_labels)
        anomaly_scores['trained_score'] = active_output['trained_score']
        anomaly_scores['acquisition']   = active_output['acquisition']
    return anomaly_scores


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------
def process_run(run_dir: Path, epsilon: float,
                steps: int, suffix: str, csv_df: pd.DataFrame, labels_all: np.ndarray,
                use_pca: bool = False):
    # --- Load data_seed from BYOL checkpoint ---
    ckpt = torch.load(run_dir / "byol_model_best.pt", map_location="cpu", weights_only=False)
    data_seed = int(ckpt["config"]["data_seed"])
    np.random.seed(data_seed)
    print(f"  data_seed={data_seed}", flush=True)

    data_dir     = run_dir / "data"
    protege_dir  = run_dir / "protege"
    protege_dir.mkdir(parents=True, exist_ok=True)

    # --- Load projections and indices ---
    lab_proj  = np.load(data_dir / "train_projections.npy")
    lab_idx   = np.load(data_dir / "labelled_train_idx.npy")
    unlab_idx = np.load(data_dir / "unlabelled_train_idx.npy")

    unlab_proj_path = data_dir / "unlabelled_train_projections.npy"
    if unlab_proj_path.exists() and len(lab_idx) > 0 and len(unlab_idx) > 0:
        unlab_proj = np.load(unlab_proj_path)
        all_proj   = np.concatenate([lab_proj, unlab_proj], axis=0)
        all_idx    = np.concatenate([lab_idx, unlab_idx])
    elif len(lab_idx) == 0:
        # f=0: train_projections.npy is the full (unlabelled) set
        all_proj = lab_proj
        all_idx  = unlab_idx
    else:
        # f=1: train_projections.npy is the full (labelled) set
        all_proj = lab_proj
        all_idx  = lab_idx

    # n_query_additional is the full unlabelled pool — ensures total budget = train_size
    n_query_additional = len(unlab_idx)

    # --- Source names (CSV row order aligned with labels_filtered.npy) ---
    source_names = csv_df.iloc[all_idx]["Source_Name"].values

    # --- Labels ---
    labels_split  = labels_all[all_idx]
    labels_npy_df = pd.DataFrame(labels_split.astype(bool), columns=LABEL_COLS)
    human_labels  = np.ones(len(labels_npy_df), dtype=int)
    for score_val, cols in reversed(TIERS):
        mask = labels_npy_df[cols].any(axis=1).values
        human_labels[mask] = score_val
    labels_df = pd.DataFrame({"human_label": human_labels}, index=source_names)

    # --- Features: StandardScaler (+ optional PCA at 0.95 variance) ---
    scaler = StandardScaler()
    proj_scaled = scaler.fit_transform(all_proj)
    if use_pca:
        pca = PCA(n_components=0.95, svd_solver="full")
        proj_final = pca.fit_transform(proj_scaled)
        print(f"  PCA: {all_proj.shape[1]}D -> {proj_final.shape[1]}D  "
              f"(explained var >= 0.95)", flush=True)
    else:
        proj_final = proj_scaled
        print(f"  Features: {proj_final.shape[1]}D (no PCA)", flush=True)
    features_pca = pd.DataFrame(proj_final, index=source_names)

    # --- Labelled seed names ---
    if len(lab_idx) > 0 and not unlab_proj_path.exists():
        # f=1 case: all sources are labelled seeds
        lab_names = source_names
    elif len(lab_idx) > 0:
        # 0 < f < 1: lab_proj / lab_idx are the labelled subset
        lab_names = csv_df.iloc[lab_idx]["Source_Name"].values
    else:
        # f=0: no labelled seeds
        lab_names = np.array([], dtype=str)

    # --- Seed anomaly_scores ---
    anomaly_scores = pd.DataFrame(
        {"score": np.zeros(len(features_pca))}, index=features_pca.index
    )
    if len(lab_names) > 0:
        anomaly_scores.loc[lab_names, "score"] = (
            labels_df.loc[lab_names, "human_label"].values.astype(float)
        )

    # --- ScoreConverter ---
    score_converter = human_loop_learning.ScoreConverter(
        force_rerun=True, output_dir=str(protege_dir)
    )
    anomaly_scores = score_converter.run(anomaly_scores)

    # --- Active learning ---
    n_labelled_seed = int(len(lab_names))
    N_labels        = n_labelled_seed + n_query_additional

    print(f"  GP active learning: initial_steps={n_labelled_seed}, "
          f"N_labels={N_labels}, steps={steps}, epsilon={epsilon}", flush=True)

    active_output = run_GP_active_learning(
        features_pca, labels_df, anomaly_scores,
        output_dir=str(protege_dir),
        steps=steps,
        initial_steps=n_labelled_seed,
        N_labels=N_labels,
        epsilon=epsilon,
    )

    # --- Recall curve (eval = unqueried sources) ---
    eval_mask    = active_output["human_label"] == -1
    eval_sources = active_output[eval_mask].index
    n_eval       = int(eval_mask.sum())

    if n_eval == 0:
        print(f"  WARNING: no eval sources (all sources were queried). "
              f"AUC cannot be computed for this run.", flush=True)
        auc   = float("nan")
        n_pos = 0
    else:
        true_labels = labels_df.loc[eval_sources, "human_label"]
        true_pos    = (true_labels >= 4).astype(int)
        n_pos       = int(true_pos.sum())

        eval_scores = active_output.loc[eval_sources, "trained_score"]
        sorted_idx  = eval_scores.sort_values(ascending=False).index
        sorted_pos  = true_pos.loc[sorted_idx].values
        cum_found   = np.cumsum(sorted_pos)
        x           = np.arange(1, n_eval + 1)

        auc = float(np.trapz(cum_found, x) / (n_eval * n_pos)) if n_pos > 0 else 0.0

        # --- Save recall curve plot ---
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, cum_found, label="Protege", linewidth=2)
        ax.plot(x, x * (n_pos / n_eval), "k--", label="Random baseline",
                linewidth=1.5, alpha=0.7)
        ax.axvline(x=n_query_additional, color="grey", linestyle=":", linewidth=1.2,
                   alpha=0.7, label=f"n_query_additional={n_query_additional}")
        ax.set_xlabel("Sources inspected (ranked by Protege score)")
        ax.set_ylabel("Interesting sources found (label >= 4)")
        ax.set_title(f"{run_dir.name}  (AUC={auc:.4f}, {n_pos} positives in eval set of {n_eval})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(protege_dir / f"recall_curve{suffix}.png", dpi=120)
        plt.close(fig)

    # --- Save scores and summary ---
    active_output.to_parquet(protege_dir / f"protege_scores{suffix}.parquet")

    summary = {
        "run_dir":            str(run_dir),
        "data_seed":          data_seed,
        "n_labelled_seed":    n_labelled_seed,
        "n_query_additional": n_query_additional,
        "steps":              steps,
        "epsilon":            epsilon,
        "n_eval":             n_eval,
        "n_eval_positives":   n_pos,
        "auc":                None if (isinstance(auc, float) and auc != auc) else auc,
        "pca_components":     int(proj_final.shape[1]),
    }
    with open(protege_dir / f"protege_summary{suffix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    auc_str = f"{auc:.4f}" if (isinstance(auc, float) and auc == auc) else "N/A"
    print(f"  -> AUC={auc_str}  eval={n_eval}  positives={n_pos}", flush=True)
    return auc, n_eval, n_pos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Protege GP active-learning sweep over BYOL run directories."
    )
    parser.add_argument("--outputs-root",      default="outputs",
                        help="Root directory containing run subdirectories.")
    parser.add_argument("--run-glob",          default="run_no_val_run_f*_sw*",
                        help="Glob pattern for run directories.")
    parser.add_argument("--epsilon",           type=float, default=0.0,
                        help="GP acquisition epsilon: exploration-exploitation trade-off (0=exploit, 3=paper default).")
    parser.add_argument("--steps",             type=int, default=10,
                        help="Sources labelled per GP iteration.")
    parser.add_argument("--output-suffix",     default="",
                        help="Suffix appended to output filenames (e.g. 'ei05').")
    parser.add_argument("--pca",               action="store_true",
                        help="Apply PCA (0.95 variance) after StandardScaler (default: off).")
    parser.add_argument("--force",             action="store_true",
                        help="Re-run even if protege_summary already exists.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    suffix       = f"_{args.output_suffix}" if args.output_suffix else ""

    # Pre-load shared data once
    csv_df     = pd.read_csv(CSV_PATH)
    labels_all = np.load(LABELS_PATH)

    # Discover run directories
    run_dirs = sorted(outputs_root.glob(args.run_glob))
    run_dirs = [rd for rd in run_dirs if re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)]
    if not run_dirs:
        print(f"No run directories found matching '{args.run_glob}' under {outputs_root}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_dirs)} run directories.\n")

    results = []
    for rd in run_dirs:
        m      = re.search(r'_f([\d.]+)_sw([\d.]+)_', rd.name)
        f_val  = float(m.group(1))
        sw_val = float(m.group(2))
        if f_val == 1.0:
            print(f"[{rd.name}]  skipping (f=1, no unlabelled eval set)", flush=True)
            print()
            continue

        summary_path = rd / "protege" / f"protege_summary{suffix}.json"
        if summary_path.exists() and not args.force:
            print(f"[{rd.name}]  skipping (already done — use --force to rerun)", flush=True)
            with open(summary_path) as fh:
                s = json.load(fh)
            results.append(dict(name=rd.name, f=f_val, sw=sw_val,
                                auc=s.get("auc"), n_eval=s.get("n_eval", 0),
                                n_pos=s.get("n_eval_positives", 0)))
            print()
            continue

        print(f"[{rd.name}]  f={f_val}  sw={sw_val}", flush=True)
        try:
            auc, n_eval, n_pos = process_run(
                rd, args.epsilon,
                args.steps, suffix, csv_df, labels_all,
                use_pca=args.pca,
            )
            results.append(dict(name=rd.name, f=f_val, sw=sw_val,
                                auc=auc, n_eval=n_eval, n_pos=n_pos))
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
        print()

    # Summary table
    if results:
        results.sort(key=lambda r: r["auc"] if (isinstance(r["auc"], float) and r["auc"] == r["auc"]) else -1.0, reverse=True)
        print("\n" + "=" * 70)
        print(f"{'Run':<45}  {'f':>5}  {'sw':>6}  {'AUC':>7}  {'n_eval':>7}  {'n_pos':>6}")
        print("-" * 70)
        for r in results:
            auc_s = f"{r['auc']:>7.4f}" if (isinstance(r['auc'], float) and r['auc'] == r['auc']) else "    N/A"
            print(f"{r['name']:<45}  {r['f']:>5}  {r['sw']:>6}  "
                  f"{auc_s}  {r['n_eval']:>7}  {r['n_pos']:>6}")
        print("=" * 70)


if __name__ == "__main__":
    main()
