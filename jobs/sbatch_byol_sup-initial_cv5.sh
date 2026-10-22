#!/bin/bash
# Experiment: cross-label supervision sweep.
# Friend-pairing uses `initial` labels (5-class FRI/FRII/Hybrids/Spirals/Doubles).
# Evaluation uses all 20 labels (--label-type full, unchanged from existing runs).
# Baseline for nDCG comparison: pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1 (reused).
#SBATCH --job-name=byol_supinit_cv5
# One task per (sw, seed) cell so failed cells can be individually requeued.
#SBATCH --array=0-14
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
# GPU required for EfficientNet training.
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
# 4 CPUs matches --num-workers 4 used in all existing sw sweep runs.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Actual elapsed from sacct: sw=0→1h21m, sw=0.05→2h31m. 4h gives safe headroom.
#SBATCH --time=04:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

VENV=/users/mbredber/p3_SUPLAT/.venv
PROJECT=/users/mbredber/p3_SUPLAT

source "${VENV}/bin/activate"
cd "${PROJECT}"
mkdir -p outputs/logs

# Grid layout:
#   Tasks 0-4:   sw=0.05 × seeds 2-6
#   Tasks 5-9:   sw=0.1  × seeds 2-6
#   Tasks 10-14: sw=0.5  × seeds 2-6
SW_VALUES=(0.05 0.05 0.05 0.05 0.05  0.1 0.1 0.1 0.1 0.1  0.5 0.5 0.5 0.5 0.5)
DATA_SEEDS=(  2    3    4    5    6    2   3   4   5   6    2   3   4   5   6)

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
DATA_SEED=${DATA_SEEDS[$i]}
# Convention from all existing runs: training_seed == data_seed.
TRAINING_SEED=${DATA_SEED}

RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sup-initial_sw${SW}_f1"

echo "=== sup-initial sw=${SW} data_seed=${DATA_SEED} — $(date) ==="
echo "Node: ${SLURMD_NODENAME:-local}  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "Run:  ${RUN_NAME}"

# All hyperparameters verified against
# outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1/
#   data_seed_2/training_seed_2/logs/configuration_log.txt.
# Only differences from sw=0.0 baseline: --supervision-weight and --sup-label-type.
python scripts/train_byol.py \
    --run-name           "${RUN_NAME}" \
    --model-type         efficientnet-b0 \
    --projector          mlp \
    --projection-dim     128 \
    --augmentation       quart_ext \
    --vicreg-var-weight  2.0 \
    --vicreg-cov-weight  0.2 \
    --vicreg-gamma       0.25 \
    --weight-decay       0.001 \
    --lr-schedule        constant \
    --ema-decay          0.996 \
    --weighting          closest \
    --batch-size         512 \
    --epochs             300 \
    --lr                 3e-4 \
    --num-workers        4 \
    --training-seed      "${TRAINING_SEED}" \
    --data-seed          "${DATA_SEED}" \
    --supervision-weight "${SW}" \
    --f-label            1 \
    --label-type         full \
    --sup-label-type     initial \
    --output-dir         outputs/ \
    --no-timestamp \
    --cross-val

echo "=== Done — $(date) ==="
