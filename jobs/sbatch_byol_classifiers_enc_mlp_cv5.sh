#!/bin/bash
#SBATCH --job-name=byol_clf_enc_mlp_cv5
#SBATCH --array=0-4
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# sklearn MLP on encoder features (encodings), 5-fold CV, sw=0.05 f=1 run.
# Generates mlp_encodings.json in each cross_val_K dir.
# One array task per fold.

set -euo pipefail

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
mkdir -p outputs/logs

FOLD=$SLURM_ARRAY_TASK_ID
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1"

echo "=== byol_clf_enc_mlp_cv5  fold=${FOLD} -- $(date) ==="

python scripts/run_downstream_mlp_classifier.py \
    --outputs-root outputs/byol_runs \
    --run-glob     "${RUN_NAME}" \
    --feature-type encodings \
    --label-set    initial_pure \
    --hidden-layers 256,128 \
    --max-iter     500 \
    --lr           1e-3 \
    --seed         2 \
    --data-seed    2 \
    --cv-fold      ${FOLD}

echo "Done -- $(date)"
