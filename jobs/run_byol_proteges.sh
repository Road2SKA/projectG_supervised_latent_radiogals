#!/bin/bash
#SBATCH --job-name=protege_all
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p logs

# ── Class weighting ───────────────────────────────────────────────────────────
# Controls how GP seed samples are weighted (per-sample scalar).
# MODE: Empty string = use script default (score). Options: score | initial | ...
#   'score' (default) weights by interest tier 1–4; no purity constraint.
#   Label-set modes require a pure label set (one positive per sample in the set).
# STRENGTH: 0.0 = uniform, 1.0 = each class contributes equally (script default).
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0

echo "Starting protege sweep — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Class weighting: ${CLASS_WEIGHT_MODE:-score (default)}  strength=${CLASS_WEIGHT_STRENGTH}"

python scripts/train_byol_proteges.py \
    --outputs-root outputs \
    --run-glob "byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema*_*_f*_sw*_*" \
    --workers 8 \
    --steps 100 \
    --force \
    ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}

echo "Done — $(date)"
