#!/bin/bash
#SBATCH --job-name=extract_cv_proj_sw1_sw10
#SBATCH --array=0-1
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# Extract CV projections for sw=1.0 and sw=10.0 (f=1 only).
# Must be run after the corresponding training jobs complete.
# Array layout:
#   0: sw=1.0  f=1
#   1: sw=10.0 f=1

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=("1.0" "10.0")
F_VALUES=( "1.0" "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0: 1.0 → 1
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"
RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"
SPLITS_DIR="outputs/data_splits/2"

echo "=== extract_cv_proj  task=${i}  run=${RUN_NAME} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  GPU: ${CUDA_VISIBLE_DEVICES:-?}"

python scripts/extract_cv_projections.py \
    --run-dir      "${RUN_DIR}" \
    --data-dir     data/preprocessed/lotss \
    --splits-dir   "${SPLITS_DIR}" \
    --f-label      "${F}" \
    --augmentation quart_ext \
    --batch-size   512 \
    --workers      4

echo "=== Done task=${i} — $(date) ==="
