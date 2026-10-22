#!/bin/bash
# Supervised baseline (no BYOL init) with proper early stopping.
# Validation set is 10% of TRAINING data — test set is never seen during training.
# Uses the same data splits as lineareval (data_seed=2, 5-fold CV).
# Compare with retrain_sup_noes_cv5 (fixed 20 epochs, no ES).
#SBATCH --job-name=retrain_sup_es_cv5
#SBATCH --array=0-4
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
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

FOLD=$SLURM_ARRAY_TASK_ID
BYOL_RUN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1/data_seed_2/training_seed_2"

echo "=== retrain_sup_es_cv5  fold=${FOLD} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python scripts/train_baseline_classifier.py \
    --run_dir        outputs/classifiers/retrain_sup_es_cv5 \
    --run_name       "retrain_sup_es_cv5_fold${FOLD}" \
    --data_dir       data/preprocessed/lotss \
    --model          enb0 \
    --label_set      initial_pure \
    --epochs         100 \
    --batch_size     64 \
    --lr             1e-3 \
    --seed           2 \
    --data_seed      2 \
    --cv_fold        "${FOLD}" \
    --n_runs         1 \
    --num_workers    8 \
    --early_stopping \
    --patience       15 \
    --byol_run_dir   "${BYOL_RUN_DIR}"

echo "=== Done — $(date) ==="
