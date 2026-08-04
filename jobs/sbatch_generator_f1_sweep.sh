#!/bin/bash
#SBATCH --job-name=train_gen_f1
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

IMAGES="data/preprocessed/lotss/images_filtered.npy"
LABEL="initial"
SEED=2

for RUN_DIR in outputs/byol_runs/pd128_*_f1; do
    echo "════════════════════════════════════════════════════════"
    echo "Run: ${RUN_DIR}"
    echo "════════════════════════════════════════════════════════"
    python scripts/train_generative.py \
        --base-dir            "${RUN_DIR}" \
        --images-path         "${IMAGES}" \
        --label-subset        "${LABEL}" \
        --seed                "${SEED}" \
        --decoder-type        flow \
        --decoder-batch-size  64 \
        --base-ch             32 \
        --decoder-epochs      300 \
        --decoder-patience    40 \
        --flow-epochs         200 \
        --flow-patience       20
    echo "  Done: $(date)"
done

echo "END: $(date)"
