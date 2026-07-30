"""
collate_hpsweep.py — Assemble per-run row files into a sweep summary CSV.

Two modes:

  --mode row --run-dir <path>
      Read metrics from a single completed run and write <run_dir>/data/row.json.
      Called inline at the end of each sbatch task.

  --mode aggregate --sweep-root <dir>
      Glob all */data/row.json under sweep-root, build a DataFrame, and write
      <sweep_root>/../sweep_results.csv  (i.e. hyperparameter_tuning/sweep_results.csv).
      Tolerates missing / malformed row files.
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_config(config_path: Path) -> dict:
    """Parse logs/configuration_log.txt → dict of key: value strings."""
    cfg = {}
    if not config_path.exists():
        return cfg
    with open(config_path) as fh:
        for line in fh:
            line = line.strip()
            if ': ' in line and not line.startswith('='):
                key, _, val = line.partition(': ')
                cfg[key.strip()] = val.strip()
    return cfg


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _argmin(lst):
    """Return index of minimum ignoring None values."""
    valid = [(i, v) for i, v in enumerate(lst) if v is not None]
    if not valid:
        return None
    return min(valid, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Row mode: read one run, write data/row.json
# ---------------------------------------------------------------------------

def build_row(run_dir: Path) -> dict:
    row = {}

    # ── config log ────────────────────────────────────────────────────────
    cfg = _parse_config(run_dir / 'logs' / 'configuration_log.txt')
    # The first line "Run: <run_id>" is parsed as key "Run"
    row['run_name']      = cfg.get('Run', run_dir.name)
    row['projection_dim'] = _safe_int(cfg.get('projection_dim'))
    row['weight_decay']   = _safe_float(cfg.get('weight_decay'))
    row['lr_schedule']    = cfg.get('lr_schedule')
    row['augmentation']   = cfg.get('augmentation')
    row['vicreg']         = 'on' if _safe_float(cfg.get('vicreg_var_weight', '0')) else 'off'

    # ── completion status ─────────────────────────────────────────────────
    status_path = run_dir / 'status.json'
    if status_path.exists():
        with open(status_path) as fh:
            status_data = json.load(fh)
        row['status'] = status_data.get('status', 'unknown')
    else:
        row['status'] = 'incomplete'

    # ── training history ──────────────────────────────────────────────────
    history_path = run_dir / 'data' / 'byol' / 'training_history_partial.json'
    if history_path.exists():
        with open(history_path) as fh:
            history = json.load(fh)

        train_loss = history.get('train_loss', [])
        if train_loss:
            row['train_loss_final']      = train_loss[-1]
            row['train_loss_best']       = min(v for v in train_loss if v is not None)
            row['train_loss_best_epoch'] = _argmin(train_loss)
        else:
            row['train_loss_final']      = None
            row['train_loss_best']       = None
            row['train_loss_best_epoch'] = None

        # val_friend_loss only exists when supervision_weight > 0 (a val set is evaluated).
        # Fall back to train_friend_loss (supervised pairing loss during training), which
        # is always populated regardless of supervision_weight.
        val_friend = history.get('val_friend_loss') or history.get('monitor_val_loss', [])
        valid_vf = [v for v in val_friend if v is not None]
        if valid_vf:
            row['val_friend_loss_final']      = val_friend[-1]
            row['val_friend_loss_best']       = min(valid_vf)
            row['val_friend_loss_best_epoch'] = _argmin(val_friend)
        else:
            row['val_friend_loss_final']      = None
            row['val_friend_loss_best']       = None
            row['val_friend_loss_best_epoch'] = None

        train_friend = history.get('train_friend_loss', [])
        valid_tf = [v for v in train_friend if v is not None]
        if valid_tf:
            row['train_friend_loss_final']      = train_friend[-1]
            row['train_friend_loss_best']       = min(valid_tf)
            row['train_friend_loss_best_epoch'] = _argmin(train_friend)
        else:
            row['train_friend_loss_final']      = None
            row['train_friend_loss_best']       = None
            row['train_friend_loss_best_epoch'] = None
    else:
        for col in ['train_loss_final', 'train_loss_best', 'train_loss_best_epoch',
                    'val_friend_loss_final', 'val_friend_loss_best', 'val_friend_loss_best_epoch',
                    'train_friend_loss_final', 'train_friend_loss_best', 'train_friend_loss_best_epoch']:
            row[col] = None

    # ── protege summary ───────────────────────────────────────────────────
    # train_byol_proteges.py saves to data/protege/protege_summary_proj_nopca.json (key: test_auc)
    # Legacy path: protege/protege_summary.json (key: auc)
    protege_path = run_dir / 'data' / 'protege' / 'protege_summary_proj_nopca.json'
    if not protege_path.exists():
        protege_path = run_dir / 'protege' / 'protege_summary.json'
    if protege_path.exists():
        with open(protege_path) as fh:
            protege_data = json.load(fh)
        row['protege_proj_recall_auc'] = _safe_float(
            protege_data.get('test_auc', protege_data.get('auc'))
        )
    else:
        row['protege_proj_recall_auc'] = None

    # ── LR classifier probe ───────────────────────────────────────────────
    # run_hpsweep_classifiers.py saves to data/classifiers/lr_initial_pure_projections.json
    clf_path = run_dir / 'data' / 'classifiers' / 'lr_initial_pure_projections.json'
    if clf_path.exists():
        with open(clf_path) as fh:
            clf_data = json.load(fh)
        row['lr_f1_macro']     = _safe_float(clf_data.get('f1_macro'))
        row['lr_auc_macro']    = _safe_float(clf_data.get('auc_macro'))
        row['lr_accuracy']     = _safe_float(clf_data.get('accuracy'))
        row['lr_recall_macro'] = _safe_float(clf_data.get('recall_macro'))
    else:
        row['lr_f1_macro']     = None
        row['lr_auc_macro']    = None
        row['lr_accuracy']     = None
        row['lr_recall_macro'] = None

    return row


def mode_row(run_dir: Path):
    run_dir = run_dir.resolve()
    row = build_row(run_dir)
    out_path = run_dir / 'data' / 'row.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(row, fh, indent=2)
    print(f"Row written to {out_path}")
    print(f"  status={row['status']}  protege_auc={row['protege_proj_recall_auc']}  lr_f1={row['lr_f1_macro']}")


# ---------------------------------------------------------------------------
# Aggregate mode: collect all row.json → CSV
# ---------------------------------------------------------------------------

def mode_aggregate(sweep_root: Path):
    import pandas as pd

    sweep_root = sweep_root.resolve()
    row_files = sorted(sweep_root.glob('*/data/row.json'))
    print(f"Found {len(row_files)} row files under {sweep_root}")

    rows = []
    for rf in row_files:
        try:
            with open(rf) as fh:
                row = json.load(fh)
            row.setdefault('run_dir', str(rf.parent.parent))
            rows.append(row)
        except Exception as exc:
            print(f"  WARNING: could not read {rf}: {exc}")
            rows.append({'run_dir': str(rf.parent.parent), 'status': 'malformed'})

    # Also catch run dirs that have no row.json yet (crashed before collate)
    all_run_dirs = {p.parent for p in sweep_root.glob('*/data/')}
    seen = {Path(r.get('run_dir', '')).resolve() for r in rows}
    for rd in all_run_dirs:
        if rd.resolve() not in seen:
            print(f"  WARNING: no row.json found for {rd.name}")
            rows.append({'run_dir': str(rd), 'status': 'missing'})

    df = pd.DataFrame(rows)

    # Canonical column order
    columns = [
        'run_name', 'projection_dim', 'weight_decay', 'lr_schedule', 'augmentation', 'vicreg', 'status',
        'train_loss_final', 'train_loss_best', 'train_loss_best_epoch',
        'train_friend_loss_final', 'train_friend_loss_best', 'train_friend_loss_best_epoch',
        'val_friend_loss_final', 'val_friend_loss_best', 'val_friend_loss_best_epoch',
        'protege_proj_recall_auc',
        'lr_f1_macro', 'lr_auc_macro', 'lr_accuracy', 'lr_recall_macro',
        'run_dir',
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]

    out_csv = sweep_root.parent / 'sweep_results.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSweep results written to {out_csv}")
    print(df[['run_name', 'projection_dim', 'weight_decay', 'lr_schedule', 'augmentation', 'vicreg', 'status',
              'train_friend_loss_final', 'protege_proj_recall_auc',
              'lr_f1_macro', 'lr_auc_macro']].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Collate hyperparameter sweep results")
    ap.add_argument('--mode', choices=['row', 'aggregate'], required=True)
    ap.add_argument('--run-dir', type=Path, default=None,
                    help="(row mode) Path to a single run directory")
    ap.add_argument('--sweep-root', type=Path, default=None,
                    help="(aggregate mode) Path to byol_runs/ under the sweep output dir")
    args = ap.parse_args()

    if args.mode == 'row':
        if args.run_dir is None:
            ap.error('--run-dir is required for --mode row')
        mode_row(args.run_dir)
    else:
        if args.sweep_root is None:
            ap.error('--sweep-root is required for --mode aggregate')
        mode_aggregate(args.sweep_root)


if __name__ == '__main__':
    main()
