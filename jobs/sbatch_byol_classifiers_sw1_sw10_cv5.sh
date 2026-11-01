#!/bin/bash
#SBATCH --job-name=byol_clf_sw1_sw10_cv5
#SBATCH --array=0-1
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# === Downstream classifiers (KNN/LR/RF/MLP) on BYOL projections for sw=1 and sw=10 ===
# Requires projections to already be extracted (sbatch_extract_cv_projections_sw1_sw10.sh).
# Array layout:
#   0: sw=1.0  f=1
#   1: sw=10.0 f=1

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"

mkdir -p outputs/logs

SW_VALUES=("1.0" "10.0")
F_VALUES=( "1.0" "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0: 1.0 → 1
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"

echo "Starting BYOL classifiers (CV) — $(date)"
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "run=${RUN_NAME}  data_seed=2  training_seed=2"

for FOLD in 0 1 2 3 4; do
    echo "--- fold ${FOLD} ---"
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
done

echo "Done — $(date)"
