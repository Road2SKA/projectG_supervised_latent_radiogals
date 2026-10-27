#!/bin/bash
#SBATCH --job-name=sw0_f1_extract_cv
#SBATCH --array=0-3
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

# Extract sw=0.0/f=1 projections for data seeds 3–6.
# Needed for Figure 3 in classification.ipynb, which compares sw=0.0 (sup-full)
# with sup-initial on matched fold test splits.
#
# sw=0 training is label-independent, so the data_seed_2 checkpoint is reused
# for all data seeds. One array task per data seed; each task loops over all 5 folds.
#
# Array layout:
#   0: data_seed=3
#   1: data_seed=4
#   2: data_seed=5
#   3: data_seed=6

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

DATA_SEEDS=(3 4 5 6)
DS=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1"
CKPT="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2/cross_val_0/byol_model_best.pt"
RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_${DS}/training_seed_${DS}"
SPLITS_DIR="outputs/data_splits/${DS}"

echo "=== sw0_f1_extract_cv  task=${SLURM_ARRAY_TASK_ID}  data_seed=${DS} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "checkpoint : ${CKPT}"
echo "run_dir    : ${RUN_DIR}"
echo "splits_dir : ${SPLITS_DIR}"

python scripts/extract_cv_projections.py \
    --run-dir      "${RUN_DIR}" \
    --data-dir     data/preprocessed/lotss \
    --splits-dir   "${SPLITS_DIR}" \
    --f-label      1.0 \
    --augmentation quart_ext \
    --batch-size   512 \
    --workers      4 \
    --checkpoint   "${CKPT}"

echo "=== Done task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
