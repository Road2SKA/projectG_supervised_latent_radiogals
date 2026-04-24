#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# =============================================================================
# CONFIGURATION — edit before submitting
# =============================================================================
MODEL=resnet18
FCM=pca
LABEL_SET=initial
EPOCHS=400
# =============================================================================

BASE="python scripts/create_embeddings.py \
    --model-type=$MODEL \
    --feature-compression-mode=$FCM \
    --label-type=$LABEL_SET \
    --epochs=$EPOCHS \
    --batch-size=256 \
    --compile"

# --- Extended augmentation ---------------------------------------------------
$BASE --weighting=ponderate --augmentation=extended --run-name=convnext_tiny_mlp_pond_extaug

# --- Closest weighting -------------------------------------------------------
$BASE --weighting=closest --run-name=enb0_pca_closest

# --- Baseline: ponderate weighting -------------------------------------------
$BASE --weighting=ponderate --run-name=enb0_pca_pond

# --- Dropout + weight decay --------------------------------------------------
$BASE --weighting=ponderate --dropout=0.2 --weight-decay=1e-4 --run-name=enb0_pca_pond_reg

# --- LR step decay -----------------------------------------------------------
$BASE --weighting=ponderate --lr-schedule=step --run-name=enb0_pca_pond_lrstep


