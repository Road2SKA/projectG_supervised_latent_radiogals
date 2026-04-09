#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:05:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# Shared fixed settings
# Model: EfficientNet-B0 (default), PCA compression (default), both loss mode, sw=1, initial labels, standard aug
BASE="python scripts/create_embeddings.py --epochs=400 --label-type=initial --weighting=ponderate"

# =============================================================================
# BEST CONDITIONS (EfficientNet-B0 + PCA)
# =============================================================================

# --- LR step scheduling (3e-4 → ×0.2 at 70%) --------------------------------
$BASE \
    --lr-schedule=step \
    --run-name=enb0_pca_ponderate_lr_step

# --- Ponderate weighting (no LR scheduling) ----------------------------------
$BASE \
    --run-name=enb0_pca_ponderate

# --- Cosine LR scheduling ----------------------------------------------------
$BASE \
    --lr-schedule=cosine \
    --run-name=enb0_pca_ponderate_lr_cosine
