#!/bin/bash
#SBATCH --job-name=train_gen
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT

# =============================================================================
# Edit these before submitting
# =============================================================================
BYOL_RUN="outputs/run_cnxt_pca_pond_step_wd_20260507_1151"
IMAGES="data/preprocessed/lotss/images_filtered.npy"
LABEL="initial"

# =============================================================================
# Flow decoder + NSF  (default: flow, batch_size=64, base_ch=32)
# Lower --decoder-batch-size or --base-ch=16 if you still hit OOM.
# =============================================================================
python scripts/train_generative.py \
    --base-dir            "${BYOL_RUN}" \
    --images-path         "${IMAGES}" \
    --label-subset        "${LABEL}" \
    --decoder-type        flow \
    --decoder-batch-size  64 \
    --base-ch             32 \
    --decoder-epochs      300 \
    --decoder-patience    40 \
    --flow-epochs         200 \
    --flow-patience       20

echo "END: $(date)"
