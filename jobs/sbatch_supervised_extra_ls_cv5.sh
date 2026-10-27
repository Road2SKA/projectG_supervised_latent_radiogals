#!/bin/bash
#SBATCH --job-name=supervised_extra_ls_cv5
#SBATCH --array=0-14
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

# Supervised baseline (no BYOL init) for extra label sets — 3 ls × 5 folds = 15 tasks.
# Results go to outputs/classifiers/supervised_noes_cv5/without_generative/
# (same dir as initial_pure runs; label set read from results.json by Section 7 cell).
#
# Array layout: task_id = ls_idx*5 + fold_idx

set -euo pipefail
VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

LABEL_SETS=("classical_pure" "initial_binary" "full_binary")

LS_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
LS=${LABEL_SETS[$LS_IDX]}

echo "=== supervised_extra_ls_cv5  ls=${LS}  fold=${FOLD}  task=${SLURM_ARRAY_TASK_ID} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python scripts/train_baseline_classifier.py \
    --run_dir        outputs/classifiers/supervised_noes_cv5 \
    --run_name       "supervised_noes_cv5_ls${LS}_fold${FOLD}" \
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
    --force

echo "=== Done — $(date) ==="
