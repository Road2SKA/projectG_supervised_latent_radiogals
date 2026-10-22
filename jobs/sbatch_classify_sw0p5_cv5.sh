#!/bin/bash
#SBATCH --job-name=classify_sw0p5_cv5
#SBATCH --array=0-4
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# KNN / LR / RF downstream classifiers for sw=0.5 f=1, 5-fold CV.
# Generates knn_projections.json, lr_projections.json, rf_projections.json
# in each cross_val_K/data/downstream_classifiers/initial_pure_cwNone/ dir.
# Run AFTER sbatch_mlp_es02_cv5.sh tasks 15-19 (mlp_v2 variants).

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw0.5_f1"

echo "=== classify_sw0p5_cv5  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python scripts/run_downstream_classifiers.py \
    --outputs-root outputs/byol_runs \
    --run-glob     "${RUN_NAME}" \
    --feature-type projections \
    --label-set    initial_pure \
    --n-estimators 200 \
    --workers      8 \
    --seed         2 \
    --data-seed    2 \
    --cv-fold      "${FOLD}"

echo "=== Done — $(date) ==="
