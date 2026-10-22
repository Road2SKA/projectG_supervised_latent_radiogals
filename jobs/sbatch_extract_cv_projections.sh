#!/bin/bash
#SBATCH --job-name=extract_cv_proj
#SBATCH --array=0-12
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

# === PIPELINE STEP 2/4: Extract BYOL feature vectors ===
# Runs the trained encoder over the dataset and saves projection/encoding .npy
# files used by downstream classifiers. Kept separate from training so the same
# checkpoint can be re-extracted with different settings (feature type, subset).
# After this: run sbatch_byol_classifiers_cv5.sh + sbatch_finetuning_cv5.sh (step 3)
#
# Extract BYOL projections for all 5 CV folds across 13 run combinations.
# One array task per (sw, f) run; each task extracts all 5 folds sequentially.
#
# Array layout (matches sbatch_byol_cv5.sh indices + extras):
#   0:  sw=0.0  f=1.0
#   1:  sw=0.05 f=0.05
#   2:  sw=0.05 f=0.1
#   3:  sw=0.05 f=0.25
#   4:  sw=0.05 f=0.5
#   5:  sw=0.05 f=1.0
#   6:  sw=0.1  f=0.05
#   7:  sw=0.1  f=0.1
#   8:  sw=0.1  f=0.25
#   9:  sw=0.1  f=0.5
#  10:  sw=0.1  f=1.0
#  11:  sw=0.5  f=1.0
#  12:  sw=0.0  f=0.1
#
# Run BEFORE sbatch_byol_classifiers_cv5.sh and sbatch_mlp_es02_cv5.sh.

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1"  "0.5"  "0.0")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "1.0"  "0.1")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0: 1.0 → 1, 0.1 → 0.1
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"
RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"
SPLITS_DIR="outputs/data_splits/2"

echo "=== extract_cv_proj  task=${i}  run=${RUN_NAME} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  GPU: ${CUDA_VISIBLE_DEVICES:-?}"

# Task 12 (sw=0.0_f=0.1): reuse the sw=0.0_f=1 checkpoint — labels don't affect
# sw=0.0 training so both models are identical. The f=0.1 split files differ, so
# we still need a separate extraction pass with f-label=0.1.
EXTRA_ARGS=""
if [ "${i}" -eq 12 ]; then
    SW0_F1_CKPT="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1/data_seed_2/training_seed_2/cross_val_0/byol_model_best.pt"
    EXTRA_ARGS="--checkpoint ${SW0_F1_CKPT}"
    echo "  Task 12: using sw=0.0_f=1 checkpoint: ${SW0_F1_CKPT}"
fi

python scripts/extract_cv_projections.py \
    --run-dir      "${RUN_DIR}" \
    --data-dir     data/preprocessed/lotss \
    --splits-dir   "${SPLITS_DIR}" \
    --f-label      "${F}" \
    --augmentation quart_ext \
    --batch-size   512 \
    --workers      4 \
    ${EXTRA_ARGS}

echo "=== Done task=${i} — $(date) ==="
