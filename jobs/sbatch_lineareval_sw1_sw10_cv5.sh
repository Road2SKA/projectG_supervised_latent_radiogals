#!/bin/bash
#SBATCH --job-name=lineareval_sw1_sw10_cv5
#SBATCH --array=0-39
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

# Linear eval (frozen BYOL backbone) for sw=1.0 and sw=10.0 — all label sets, 5-fold CV.
# Mirrors sbatch_lineareval_byol_sw_sweep_cv5.sh + sbatch_lineareval_extra_ls_cv5.sh
# for the two missing sw values.
# Results go to outputs/classifiers/finetuned_sw_cv5/without_generative/
# Does NOT overwrite existing results (skip check on results.json).
#
# Array layout: task_id = ls_idx*10 + sw_idx*5 + fold
#   ls_idx 0 → initial_pure,   tasks 0-9
#   ls_idx 1 → classical_pure, tasks 10-19
#   ls_idx 2 → initial_binary, tasks 20-29
#   ls_idx 3 → full_binary,    tasks 30-39
#   sw_idx 0 → sw=1.0,  folds 0-4
#   sw_idx 1 → sw=10.0, folds 5-9

set -euo pipefail
VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

LABEL_SETS=("initial_pure" "classical_pure" "initial_binary" "full_binary")
SW_VALUES=("1.0" "10.0")

LS_IDX=$(( SLURM_ARRAY_TASK_ID / 10 ))
SW_IDX=$(( (SLURM_ARRAY_TASK_ID % 10) / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
LS=${LABEL_SETS[$LS_IDX]}
SW=${SW_VALUES[$SW_IDX]}

# Run name matches initial_pure (ls suffix omitted) vs extra label sets
if [ "${LS}" = "initial_pure" ]; then
    RUN_NAME_SUFFIX="lineareval_byol_sw_cv5_sw${SW}_fold${FOLD}"
else
    RUN_NAME_SUFFIX="lineareval_byol_sw_cv5_ls${LS}_sw${SW}_fold${FOLD}"
fi

OUT_DIR="outputs/classifiers/finetuned_sw_cv5/without_generative/${RUN_NAME_SUFFIX}"
RESULTS_FILE="${OUT_DIR}/results.json"
if [ -f "${RESULTS_FILE}" ]; then
    echo "Skipping — already complete: ${RESULTS_FILE}"
    exit 0
fi

BYOL_RUN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/data_seed_2/training_seed_2"
FOLD_CKPT="${BYOL_RUN_DIR}/cross_val_${FOLD}/byol_model_best.pt"
SHARED_CKPT="${BYOL_RUN_DIR}/byol_model_best.pt"
if [ -f "${FOLD_CKPT}" ]; then
    BYOL_CKPT="${FOLD_CKPT}"
elif [ -f "${SHARED_CKPT}" ]; then
    BYOL_CKPT="${SHARED_CKPT}"
else
    echo "ERROR: no checkpoint at ${FOLD_CKPT} or ${SHARED_CKPT}"
    exit 1
fi

echo "=== lineareval_sw1_sw10_cv5  ls=${LS}  sw=${SW}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"
echo "Checkpoint: ${BYOL_CKPT}"

python scripts/train_baseline_classifier.py \
    --run_dir        outputs/classifiers/finetuned_sw_cv5 \
    --run_name       "${RUN_NAME_SUFFIX}" \
    --data_dir       data/preprocessed/lotss \
    --model          enb0 \
    --label_set      "${LS}" \
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
