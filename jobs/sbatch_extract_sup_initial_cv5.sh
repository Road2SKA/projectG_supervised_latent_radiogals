#!/bin/bash
#SBATCH --job-name=extract_supinit_cv5
#SBATCH --array=0-14
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

# Extract BYOL projections for sup-initial runs (3 sw values × 5 seeds = 15 tasks).
# Must run AFTER sbatch_byol_sup-initial_cv5.sh completes.
# Results go to cross_val_K/data/byol/ inside each run dir.
#
# Array layout (task_id = sw_idx*5 + data_seed_idx):
#   0-4:   sw=0.05, data_seeds 2-6
#   5-9:   sw=0.1,  data_seeds 2-6
#   10-14: sw=0.5,  data_seeds 2-6

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=(0.05 0.05 0.05 0.05 0.05  0.1 0.1 0.1 0.1 0.1  0.5 0.5 0.5 0.5 0.5)
DATA_SEEDS=(  2    3    4    5    6    2   3   4   5   6    2   3   4   5   6)

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
DATA_SEED=${DATA_SEEDS[$i]}

RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sup-initial_sw${SW}_f1"
RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_${DATA_SEED}/training_seed_${DATA_SEED}"
SPLITS_DIR="outputs/data_splits/${DATA_SEED}"

echo "=== extract_supinit_cv5  task=${i}  run=${RUN_NAME}  data_seed=${DATA_SEED} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  GPU: ${CUDA_VISIBLE_DEVICES:-?}"

python scripts/extract_cv_projections.py \
    --run-dir      "${RUN_DIR}" \
    --data-dir     data/preprocessed/lotss \
    --splits-dir   "${SPLITS_DIR}" \
    --f-label      1 \
    --augmentation quart_ext \
    --batch-size   512 \
    --workers      4

echo "=== Done task=${i} — $(date) ==="
