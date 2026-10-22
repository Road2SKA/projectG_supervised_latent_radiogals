#!/bin/bash
#SBATCH --job-name=retrain_byol_sw0p5_cv5
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

# Full backbone retrain (BYOL init, head + encoder unfrozen) for sw=0.5 f=1.
# Writes results.json to outputs/classifiers/retrained_sw_cv5/without_generative/
# matching the pattern used by sw=0.0, 0.05, 0.1 (read by Q3 FTr column).

set -euo pipefail
VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT
source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs
export PYTHONUNBUFFERED=1

FOLD=$SLURM_ARRAY_TASK_ID
SW="0.5"
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1"
BYOL_RUN_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"
BYOL_CKPT="${BYOL_RUN_DIR}/byol_model_best.pt"

echo "=== retrain_byol_sw0p5_cv5  fold=${FOLD} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  CPUs: ${SLURM_CPUS_PER_TASK:-8}"

python scripts/train_baseline_classifier.py \
    --run_dir        outputs/classifiers/retrained_sw_cv5 \
    --run_name       "retrain_byol_sw_cv5_sw${SW}_fold${FOLD}" \
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
    --byol_run_dir   "${BYOL_RUN_DIR}" \
    --byol_checkpoint "${BYOL_CKPT}"

echo "=== Done — $(date) ==="
