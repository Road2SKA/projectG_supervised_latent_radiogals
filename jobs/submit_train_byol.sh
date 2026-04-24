#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# =============================================================================
# SHARED DEFAULTS
# =============================================================================
EPOCHS=400
BS=256
LABEL=initial

# Shorthand builders — avoids repeating common flags in every run.
# R18 / CNXT: no --compile (not supported for these architectures)
WD=1e-4

# Shorthand builders — avoids repeating common flags in every run.
# R18 / CNXT: no --compile (not supported for these architectures)
R18="python scripts/create_embeddings.py
    --model-type=resnet18
    --label-type=$LABEL --epochs=$EPOCHS --batch-size=$BS
    --weight-decay=$WD"

CNXT="python scripts/create_embeddings.py
    --model-type=convnext-tiny
    --label-type=$LABEL --epochs=$EPOCHS --batch-size=$BS
    --weight-decay=$WD"

# =============================================================================
# 1. ARCHITECTURE SWEEP
#    Baseline settings (pca + ponderate + lr-step) across two backbones.
#    Isolates the effect of backbone alone.
# =============================================================================
$R18  --feature-compression-mode=pca --weighting=ponderate --lr-schedule=step \
      --run-name=r18_pca_pond_step_wd

$CNXT --feature-compression-mode=pca --weighting=ponderate --lr-schedule=step \
      --run-name=cnxt_pca_pond_step_wd

# =============================================================================
# 2. ENB0 — FLAT LEARNING RATE
#    Constant LR as a reference point alongside the step/cosine schedules.
# =============================================================================
python scripts/create_embeddings.py \
    --model-type=efficientnet-b0 \
    --label-type=$LABEL --epochs=$EPOCHS --batch-size=$BS \
    --compile \
    --weight-decay=$WD \
    --feature-compression-mode=pca --weighting=ponderate --lr-schedule=constant \
    --run-name=enb0_pca_pond_constant_wd

# =============================================================================
# 3. ENB0 — MLP PROJECTOR + WEIGHT DECAY
#    Tests learned MLP head combined with WD across all three LR schedules.
# =============================================================================
python scripts/create_embeddings.py \
    --model-type=efficientnet-b0 \
    --label-type=$LABEL --epochs=$EPOCHS --batch-size=$BS \
    --compile \
    --weight-decay=$WD \
    --feature-compression-mode=mlp --weighting=ponderate --lr-schedule=constant \
    --run-name=enb0_mlp_pond_constant_wd

python scripts/create_embeddings.py \
    --model-type=efficientnet-b0 \
    --label-type=$LABEL --epochs=$EPOCHS --batch-size=$BS \
    --compile \
    --weight-decay=$WD \
    --feature-compression-mode=mlp --weighting=ponderate --lr-schedule=step \
    --run-name=enb0_mlp_pond_step_wd

python scripts/create_embeddings.py \
    --model-type=efficientnet-b0 \
    --label-type=$LABEL --epochs=$EPOCHS --batch-size=$BS \
    --compile \
    --weight-decay=$WD \
    --feature-compression-mode=mlp --weighting=ponderate --lr-schedule=cosine \
    --run-name=enb0_mlp_pond_cosine_wd

echo "END: $(date)"
