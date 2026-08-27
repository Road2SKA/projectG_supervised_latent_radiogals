#!/bin/bash
#SBATCH --job-name=protege_dataseed
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --array=0-8
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

mkdir -p outputs/logs

# All (data_seed, training_seed) pairs:
#   data_seed=2: training seeds 2-6 (existing runs)
#   data_seed=N: training seed N    (new diagonal runs, N in {3,4,5,6})
DATA_SEEDS=(2 2 2 2 2 3 4 5 6)
TRAINING_SEEDS=(2 3 4 5 6 3 4 5 6)

DATA_SEED=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${TRAINING_SEEDS[$SLURM_ARRAY_TASK_ID]}

# ── Class weighting ───────────────────────────────────────────────────────────
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0

echo "Starting protege sweep — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "data_seed=${DATA_SEED}  training_seed=${SEED}"
echo "Class weighting: ${CLASS_WEIGHT_MODE:-score (default)}  strength=${CLASS_WEIGHT_STRENGTH}"

python scripts/train_byol_proteges.py \
    --outputs-root outputs \
    --run-glob "byol_runs/pd128_*_f1" \
    --byol-seed "${SEED}" \
    --byol-data-seed "${DATA_SEED}" \
    --workers 8 \
    --steps 100 \
    ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}

echo "Done — $(date)"
