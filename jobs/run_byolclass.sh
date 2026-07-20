#!/bin/bash
#SBATCH --job-name=rbyol_classifiers
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

# =============================================================================
# CONFIGURATION — keep BYOL_RUN_DIR in sync with run_classifiers.sh so both
# scripts evaluate on the same train/test split.
# Set to "" to sweep ALL matching run directories instead.
# =============================================================================
BYOL_RUN_DIR=""

# ── Class weighting ───────────────────────────────────────────────────────────
# MODE: which label set to balance. Empty string = no weighting (uniform).
#   Options: initial | morphology | environment | classical | all | score
#   Label-set modes require --label-set to be a *_pure variant (e.g. initial_pure).
# STRENGTH: 0.0 = uniform (no effect), 1.0 = each class contributes equally.
CLASS_WEIGHT_MODE="all" # Leave empty for uniform (no weighting)
CLASS_WEIGHT_STRENGTH=0.3
# =============================================================================

if [ -n "$BYOL_RUN_DIR" ]; then
    RUN_GLOB="$(basename "$BYOL_RUN_DIR")"
else
    RUN_GLOB="enb0_*"
fi

echo "Starting BYOL classifiers sweep — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Run glob: ${RUN_GLOB}"
echo "Class weighting: ${CLASS_WEIGHT_MODE:-none}  strength=${CLASS_WEIGHT_STRENGTH}"

python scripts/train_byol_classifiers.py \
    --outputs-root outputs/byol_runs \
    --run-glob     "${RUN_GLOB}" \
    --feature-type projections \
    --label-set    full \
    --n-estimators 200 \
    --workers      8 \
    ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}

echo "Done — $(date)"
