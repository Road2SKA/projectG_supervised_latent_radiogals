#!/bin/bash
#SBATCH --job-name=byol_clf_many
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
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
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs

# =============================================================================
# CONFIGURATION — set BYOL_RUN_DIR to the specific run you want to evaluate.
# =============================================================================
BYOL_RUN_DIR="outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augextended_20260707_1027"
LABEL_SET="initial_pure"
FEATURE_TYPE="projections"
N_RUNS=10
SEED=42
N_ESTIMATORS=200

# Class weighting: leave empty for no weighting (uniform).
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0
# =============================================================================

echo "Starting multi-run BYOL classifier — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Run dir: ${BYOL_RUN_DIR}"
echo "Label set: ${LABEL_SET}  Feature: ${FEATURE_TYPE}  N runs: ${N_RUNS}"
echo "Class weighting: ${CLASS_WEIGHT_MODE:-none}  strength=${CLASS_WEIGHT_STRENGTH}"

python scripts/train_byol_classifier_many_times.py \
    --byol-run-dir  "${BYOL_RUN_DIR}" \
    --label-set     "${LABEL_SET}" \
    --feature-type  "${FEATURE_TYPE}" \
    --n-runs        "${N_RUNS}" \
    --seed          "${SEED}" \
    --n-estimators  "${N_ESTIMATORS}" \
    ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}

echo "Done — $(date)"
