#!/bin/bash
# Linear eval on BYOL backbone (frozen backbone, train head only) — 4 sw values × 5-fold CV.
# Results go to outputs/classifiers/finetuned_sw_cv5/without_generative/ (read by Q3 cell).
#SBATCH --job-name=lineareval_byol_sw_cv5
#SBATCH --array=0-19
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail
VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

SW_VALUES=("0.0" "0.05" "0.1" "0.5")
SW_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
SW=${SW_VALUES[$SW_IDX]}
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1"
BYOL_RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"

# Cross-val training saves per-fold checkpoints; use fold-specific one if available,
# fall back to a shared checkpoint for sw=0.0 (where all folds are identical).
FOLD_CKPT="${BYOL_RUN_DIR}/cross_val_${FOLD}/byol_model_best.pt"
SHARED_CKPT="${BYOL_RUN_DIR}/byol_model_best.pt"
if [ -f "${FOLD_CKPT}" ]; then
    BYOL_CKPT="${FOLD_CKPT}"
elif [ -f "${SHARED_CKPT}" ]; then
    BYOL_CKPT="${SHARED_CKPT}"
else
    echo "ERROR: no checkpoint found at ${FOLD_CKPT} or ${SHARED_CKPT}"
    exit 1
fi

echo "=== lineareval_byol_sw_cv5  sw=${SW}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Checkpoint: ${BYOL_CKPT}"

python scripts/train_baseline_classifier.py \
    --run_dir        outputs/classifiers/finetuned_sw_cv5 \
    --run_name       "lineareval_byol_sw_cv5_sw${SW}_fold${FOLD}" \
    --data_dir       data/preprocessed/lotss \
    --model          enb0 \
    --label_set      initial_pure \
    --epochs         20 \
    --batch_size     64 \
    --lr             1e-3 \
    --seed           2 \
    --data_seed      2 \
    --cv_fold        "${FOLD}" \
    --n_runs         1 \
    --num_workers    8 \
    --training_mode  linear \
    --byol_run_dir   "${BYOL_RUN_DIR}" \
    --byol_checkpoint "${BYOL_CKPT}"

echo "=== Done — $(date) ==="
