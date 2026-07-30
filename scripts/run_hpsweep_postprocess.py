"""
run_hpsweep_postprocess.py — Post-processing for a single hyperparameter sweep run.

For an already-trained run directory:
  1. Runs Protege GP on projections (calls process_run from train_byol_proteges.py).
  2. Writes status.json to signal completion.
  3. Updates data/row.json via collate_hpsweep.py --mode row.

Usage:
    python scripts/run_hpsweep_postprocess.py --run-dir <path>

The protege summary is saved by process_run to:
    <run_dir>/data/protege/protege_summary_proj_nopca.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import process_run functions from existing scripts (bypasses their regex glob filters)
from train_byol_proteges import process_run as protege_run, CSV_PATH, LABELS_PATH
from train_byol_classifiers import process_run as clf_run


def main():
    ap = argparse.ArgumentParser(description="Post-process one hpsweep run: protege + status + row")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to a completed BYOL run directory")
    ap.add_argument("--epsilon", type=float, default=2.0,
                    help="GP acquisition epsilon (default: 2.0)")
    ap.add_argument("--steps", type=int, default=100,
                    help="Sources labelled per GP iteration (default: 100)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run GP even if results already exist")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: run dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Post-processing: {run_dir.name}")
    print(f"{'='*60}\n")

    # Load shared data
    csv_df     = pd.read_csv(CSV_PATH)
    labels_all = np.load(LABELS_PATH)

    # Run Protege GP on projections (no PCA, default epsilon)
    print("Running Protege GP (projections, no PCA)...")
    try:
        auc, train_auc, n_eval, n_pos = protege_run(
            run_dir,
            epsilon=args.epsilon,
            steps=args.steps,
            suffix="",           # → saved as protege_summary_proj_nopca.json
            csv_df=csv_df,
            labels_all=labels_all,
            latent="proj",
            use_pca=False,
            force=args.force,
        )
        print(f"Protege done: test_AUC={auc:.4f}  eval={n_eval}  positives={n_pos}")
    except Exception as exc:
        print(f"ERROR in Protege GP: {exc}", file=sys.stderr)
        auc = None

    # Run LR classifier (initial_pure label set, projections)
    print("\nRunning LR classifier probe (initial_pure, projections)...")
    try:
        clf_run(
            run_dir,
            feature_type="projections",
            label_set="initial_pure",
            n_estimators=200,
            n_neighbors=15,
            lr_C=1.0,
            seed=42,
            force=args.force,
            class_weight_mode=None,
            class_weight_strength=0.0,
        )
        print("Classifier done.")
    except Exception as exc:
        print(f"ERROR in classifier: {exc}", file=sys.stderr)

    # Write status.json (covers runs that completed training before status.json was added)
    status_path = run_dir / 'status.json'
    if not status_path.exists():
        with open(status_path, 'w') as fh:
            json.dump({'status': 'complete',
                       'finished_at': datetime.now().isoformat(),
                       'note': 'written by run_hpsweep_postprocess'}, fh)
        print(f"Written status.json")

    # Update row.json
    print("Updating row.json...")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'collate_hpsweep.py'),
         '--mode', 'row', '--run-dir', str(run_dir)],
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"WARNING: collate_hpsweep row mode returned {result.returncode}", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)

    print(f"\nDone: {run_dir.name}")


if __name__ == '__main__':
    main()
