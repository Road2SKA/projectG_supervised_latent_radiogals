#!/bin/bash
#SBATCH --job-name=byol_ft_sw1_sw10_cv5
#SBATCH --array=0-9
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# Finetuning classifiers for sw=1.0 and sw=10.0, f=1, 5-fold CV.
# Mirrors sbatch_finetuning_cv5.sh for the two missing sw values.
# Skips folds where the metrics file already exists.
#
# Array layout: task_id = sw_idx * 5 + fold
#   sw_idx 0 → sw=1.0,  tasks 0-4
#   sw_idx 1 → sw=10.0, tasks 5-9

set -euo pipefail

cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

mkdir -p outputs/logs

SW_VALUES=("1.0" "10.0")
SW_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
SW=${SW_VALUES[$SW_IDX]}
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1"
MODEL_PATH="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"

WD=3e-1
N_RUNS=1
NUM_WORKERS=4
LR=1e-2
DROPOUT=0.2
EPOCHS=40

echo "START: $(date)"
echo "run=${RUN_NAME}  sw=${SW}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID}"

FOLD_PATH="${MODEL_PATH}/cross_val_${FOLD}"
FOLD_MODEL="${FOLD_PATH}/byol_model_best.pt"

if [ ! -f "${FOLD_MODEL}" ]; then
    echo "Skipping — missing: ${FOLD_MODEL}"
    exit 0
fi

for MODE in 2 3; do
    METRICS_FILE="${FOLD_PATH}/data/classifiers/finetuning/initial_pure_cwNone0.0_mode${MODE}_lr${LR}_ep${EPOCHS}/finetuning_metrics.json"
    if [ -f "${METRICS_FILE}" ]; then
        echo "  Skipping already complete (mode${MODE}): ${METRICS_FILE}"
        continue
    fi

    echo "  Training mode${MODE} → ${FOLD_PATH}"
    python scripts/train_finetuning.py \
        --model-path="${MODEL_PATH}" \
        --cv-fold="${FOLD}" \
        --training-mode="${MODE}" \
        --label-set=initial_pure \
        --epochs=${EPOCHS} \
        --lr=${LR} \
        --weight-decay=${WD} \
        --dropout=${DROPOUT} \
        --n-runs=${N_RUNS} \
        --num-workers=${NUM_WORKERS} \
        --augmentation=quart_ext \
        --data-seed=2 \
        --run-name="mode${MODE}_lr${LR}_ep${EPOCHS}"
done

echo "END: $(date)"
