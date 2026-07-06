"""
Plot BYOL training/validation loss curves as an interactive HTML file.
Curves are colored by the `sw` hyperparameter, with hover info showing
run name and key hyperparameters.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_run_name(name: str) -> dict:
    params = {}
    for key, pattern in [
        ("sw",         r"_sw([\d.]+)"),
        ("vicregvar",  r"_vicregvar([\d.]+)"),
        ("cov",        r"_cov([\d.]+)"),
        ("gamma",      r"_gamma([\d.]+)"),
        ("ema",        r"_ema([\d.]+)"),
    ]:
        m = re.search(pattern, name)
        params[key] = m.group(1) if m else "?"
    return params


def sw_to_color(sw_val: str, sw_values: list[str]) -> str:
    """Map sw value to a color from a discrete palette."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    idx = sw_values.index(sw_val) if sw_val in sw_values else 0
    return palette[idx % len(palette)]


def main():
    parser = argparse.ArgumentParser(description="Plot BYOL loss curves")
    parser.add_argument("--runs-dir", default="outputs/byol_runs",
                        help="Directory containing BYOL run subdirectories")
    parser.add_argument("--output", default="outputs/figures/loss_curves.html",
                        help="Output HTML file path")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Scan and load runs ---
    records = []
    skipped = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        hist_path = run_dir / "data/byol/training_history.npy"
        if not hist_path.exists():
            skipped.append(run_dir.name)
            continue
        try:
            history = np.load(hist_path, allow_pickle=True).item()
        except Exception as e:
            skipped.append(f"{run_dir.name} (load error: {e})")
            continue

        train_loss = history.get("train_loss")
        val_loss   = history.get("val_friend_loss")
        if not train_loss and not val_loss:
            skipped.append(f"{run_dir.name} (no loss data)")
            continue

        params = parse_run_name(run_dir.name)
        records.append({
            "name":       run_dir.name,
            "params":     params,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })

    print(f"Loaded: {len(records)} runs  |  Skipped: {len(skipped)} runs")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")

    if not records:
        print("No runs to plot.")
        return

    # --- Determine unique sw values (sorted numerically where possible) ---
    def sw_sort_key(s):
        try:
            return float(s)
        except ValueError:
            return float("inf")

    all_sw = sorted({r["params"]["sw"] for r in records}, key=sw_sort_key)

    # --- Build Plotly figure ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Train Loss", "Validation Loss"),
        vertical_spacing=0.08,
    )

    seen_sw: set[str] = set()

    for rec in records:
        params     = rec["params"]
        sw         = params["sw"]
        color      = sw_to_color(sw, all_sw)
        show_leg   = sw not in seen_sw
        seen_sw.add(sw)

        hover = (
            f"<b>{rec['name']}</b><br>"
            f"sw={params['sw']}  vicregvar={params['vicregvar']}<br>"
            f"cov={params['cov']}  gamma={params['gamma']}  ema={params['ema']}"
        )
        epochs = list(range(1, len(rec["train_loss"]) + 1)) if rec["train_loss"] else []

        # Train loss (top subplot)
        if rec["train_loss"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=rec["train_loss"],
                    mode="lines",
                    line=dict(color=color, width=1),
                    name=f"sw={sw}",
                    legendgroup=f"sw={sw}",
                    showlegend=show_leg,
                    hovertemplate=hover + "<extra></extra>",
                    opacity=0.85,
                ),
                row=1, col=1,
            )

        # Val loss (bottom subplot) — same color, slightly more transparent
        if rec["val_loss"]:
            val_epochs = list(range(1, len(rec["val_loss"]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=val_epochs,
                    y=rec["val_loss"],
                    mode="lines",
                    line=dict(color=color, width=1),
                    name=f"sw={sw}",
                    legendgroup=f"sw={sw}",
                    showlegend=False,
                    hovertemplate=hover + "<extra></extra>",
                    opacity=0.65,
                ),
                row=2, col=1,
            )

    fig.update_layout(
        title="BYOL Loss Curves (colored by sw)",
        height=800,
        hovermode="closest",
        legend=dict(
            title="sw value",
            itemsizing="constant",
        ),
        xaxis2=dict(title="Epoch"),
        yaxis=dict(title="Train Loss"),
        yaxis2=dict(title="Val Loss"),
        template="plotly_white",
    )

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
