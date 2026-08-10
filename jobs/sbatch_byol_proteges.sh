#!/bin/bash
#SBATCH --job-name=protege_all
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --array=0-3
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

# Seeds 3-6, f_label=1, sw in {0.0, 0.05, 0.1, 0.5}
SEEDS=(3 4 5 6)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ── Class weighting ───────────────────────────────────────────────────────────
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0

echo "Starting protege sweep — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Seed: ${SEED}"
echo "Class weighting: ${CLASS_WEIGHT_MODE:-score (default)}  strength=${CLASS_WEIGHT_STRENGTH}"

python scripts/train_byol_proteges.py \
    --outputs-root outputs \
    --run-glob "byol_runs/pd128_*_f1" \
    --byol-seed "${SEED}" \
    --workers 8 \
    --steps 100 \
    --force \
    ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}

echo "Done — $(date)"
