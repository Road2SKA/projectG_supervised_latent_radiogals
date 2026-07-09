"""
Linear probe on saved BYOL encoder features.

Scans all run directories under --runs-dir, loads labelled_train_encodings.npy
and test_encodings.npy, fits a logistic regression per label column, and reports
macro AUC + macro F1. Also saves a scatter of final val_friend_loss vs probe AUC
so we can check whether the loss curves are a useful proxy for downstream quality.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score


SPLITS_DIR = Path("outputs/data_splits/42")


def parse_run_name(name: str) -> dict:
    params = {}
    for key, pattern in [
        ("sw",        r"_sw([\d.]+)"),
        ("vicregvar", r"_vicregvar([\d.]+)"),
        ("cov",       r"_cov([\d.]+)"),
        ("gamma",     r"_gamma([\d.]+)"),
        ("f",         r"_f([\d.]+)_sw"),
    ]:
        m = re.search(pattern, name)
        params[key] = m.group(1) if m else "?"
    return params


def _load_labels_for_run(run_name: str, y_te: np.ndarray):
    """Load the correct train labels for this run's f value."""
    m = re.search(r"_f([\d.]+)_sw", run_name)
    f_str = m.group(1) if m else "1"
    # Normalise: "1" -> "1", "0.1" -> "0.1", "0" -> "0"
    candidates = [
        SPLITS_DIR / f"labelled_train_labels_f{f_str}.npy",
        SPLITS_DIR / "labelled_train_labels_f1.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path)
    return None


def probe_run(run_dir: Path, y_te: np.ndarray) -> dict | None:
    byol_dir = run_dir / "data/byol"
    enc_tr = byol_dir / "labelled_train_encodings.npy"
    enc_te = byol_dir / "test_encodings.npy"
    hist_f = byol_dir / "training_history.npy"

    if not enc_tr.exists() or not enc_te.exists():
        return None

    y_tr = _load_labels_for_run(run_dir.name, y_te)
    if y_tr is None:
        print(f"  [SKIP] {run_dir.name}: no label file found")
        return None

    X_tr = np.load(enc_tr)
    X_te = np.load(enc_te)

    if X_tr.shape[0] != y_tr.shape[0] or X_te.shape[0] != y_te.shape[0]:
        print(f"  [SKIP] {run_dir.name}: shape mismatch "
              f"X_tr={X_tr.shape} y_tr={y_tr.shape}")
        return None

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    aucs, f1s = [], []
    for k in range(y_tr.shape[1]):
        if y_tr[:, k].sum() == 0 or y_te[:, k].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr[:, k])
        p = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te[:, k], p))
        f1s.append(f1_score(y_te[:, k], (p > 0.5).astype(int), zero_division=0))

    if not aucs:
        return None

    # Final val_friend_loss from training history
    final_val_loss = None
    if hist_f.exists():
        try:
            history = np.load(hist_f, allow_pickle=True).item()
            vfl = history.get("val_friend_loss")
            if vfl:
                final_val_loss = float(vfl[-1])
        except Exception:
            pass

    return {
        "run":           run_dir.name,
        "macro_auc":     float(np.mean(aucs)),
        "macro_f1":      float(np.mean(f1s)),
        "n_labels":      len(aucs),
        "final_val_loss": final_val_loss,
        **parse_run_name(run_dir.name),
    }


def main():
    parser = argparse.ArgumentParser(description="Linear probe on BYOL encodings")
    parser.add_argument("--runs-dir", default="outputs/byol_runs",
                        help="Directory containing run subdirectories")
    parser.add_argument("--output-csv", default="outputs/figures/probe_results.csv",
                        help="CSV output path")
    parser.add_argument("--output-html", default="outputs/figures/probe_scatter.html",
                        help="Scatter plot HTML output path (requires plotly)")
    parser.add_argument("--run", default=None,
                        help="Probe a single run directory (absolute or relative path)")
    args = parser.parse_args()

    y_te = np.load(SPLITS_DIR / "test_labels.npy")

    if args.run:
        run_dirs = [Path(args.run)]
    else:
        runs_dir = Path(args.runs_dir)
        run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())

    results = []
    for run_dir in run_dirs:
        print(f"  probing {run_dir.name} ...", end=" ", flush=True)
        res = probe_run(run_dir, y_te)
        if res is None:
            print("skipped")
            continue
        print(f"AUC={res['macro_auc']:.3f}  F1={res['macro_f1']:.3f}")
        results.append(res)

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results).sort_values("macro_auc", ascending=False)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv}")
    print(df[["run", "macro_auc", "macro_f1", "sw", "vicregvar", "final_val_loss"]].to_string(index=False))

    if not HAS_PLOTLY:
        print("plotly not available — skipping scatter plot")
        return

    # Scatter: final val_friend_loss vs probe AUC (only runs with both values)
    df_plot = df.dropna(subset=["final_val_loss"])
    if df_plot.empty:
        print("No runs with val_loss data for scatter plot.")
        return

    fig = go.Figure()
    for sw_val in sorted(df_plot["sw"].unique(),
                         key=lambda s: float(s) if s != "?" else 999):
        sub = df_plot[df_plot["sw"] == sw_val]
        fig.add_trace(go.Scatter(
            x=sub["final_val_loss"],
            y=sub["macro_auc"],
            mode="markers",
            name=f"sw={sw_val}",
            text=sub["run"],
            hovertemplate="<b>%{text}</b><br>val_loss=%{x:.3f}  AUC=%{y:.3f}<extra></extra>",
            marker=dict(size=8),
        ))

    fig.update_layout(
        title="Final val_friend_loss vs Linear Probe AUC (colored by sw)",
        xaxis_title="Final val_friend_loss",
        yaxis_title="Macro AUC (linear probe)",
        template="plotly_white",
        height=600,
    )
    fig.write_html(args.output_html)
    print(f"Saved: {args.output_html}")

    # Correlation coefficient
    corr = np.corrcoef(df_plot["final_val_loss"], df_plot["macro_auc"])[0, 1]
    print(f"\nCorrelation (val_loss vs AUC): r={corr:.3f}")
    if abs(corr) < 0.3:
        print("  → Weak correlation: val_loss is NOT a reliable proxy for probe quality.")
    elif abs(corr) < 0.6:
        print("  → Moderate correlation.")
    else:
        print("  → Strong correlation.")


if __name__ == "__main__":
    main()
