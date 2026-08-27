#!/bin/bash
#SBATCH --job-name=byol_clf_dataseed
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# data_seed+1 pairing: training_seed = data_seed + 1
DATA_SEEDS=(2 3 4 5 6)
TRAINING_SEEDS=(3 4 5 6 7)

DATA_SEED=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${TRAINING_SEEDS[$SLURM_ARRAY_TASK_ID]}

# =============================================================================
# CONFIGURATION
# =============================================================================
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0
# =============================================================================

echo "Starting BYOL classifiers — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "data_seed=${DATA_SEED}  training_seed=${SEED}"
echo "Class weighting: ${CLASS_WEIGHT_MODE:-none}  strength=${CLASS_WEIGHT_STRENGTH}"

for SW_GLOB in "pd128_*_sw0.05_f1" "pd128_*_sw0.1_f1"; do
    python scripts/train_byol_classifiers.py \
        --outputs-root outputs/byol_runs \
        --run-glob     "${SW_GLOB}" \
        --feature-type projections \
        --label-set    initial_pure \
        --n-estimators 200 \
        --workers      8 \
        --seed         "${SEED}" \
        --data-seed    "${DATA_SEED}" \
        ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
        ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}
done

echo "Done — $(date)"
