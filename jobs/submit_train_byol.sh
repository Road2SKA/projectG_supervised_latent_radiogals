#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# Shared fixed settings across all runs
# (no CV, 400 epochs, both mode, sw=1, closest weighting, full labels, standard aug)
BASE="python scripts/create_embeddings.py --epochs=400"

# =============================================================================
# TRAINING CONDITIONS SWEEP (issue: finding best training conditions)
# All runs share the same supervision parameters (both, sw=1, closest, full).
# Each run varies exactly one training parameter vs the base case.
# =============================================================================

# --- Base case ---------------------------------------------------------------
# simple aug, no LR scheduling, no warmup, no supervision scheduling
#$BASE \
#    --run-name=cond_base
#
## --- Variation 1: Extended augmentation --------------------------------------
#$BASE \
#    --augmentation=extended \
#    --run-name=cond_ext_aug
#
# --- Variation 2: LR step scheduling (3e-4 → ×0.2 at 80%) -------------------
#$BASE \
#    --lr-schedule=step \
#    --run-name=cond_lr_step
##
### --- Variation 3: LR warmup (10 epochs) --------------------------------------
#$BASE \
#    --warmup-epochs=10 \
#    --run-name=cond_warmup10
#
## --- Variation 4: Supervision weight cosine scheduling (0 → 1) ---------------
$BASE \
    --supervision-weight-schedule=cosine \
    --supervision-weight-start=0.0 \
    --supervision-weight-end=1.0 \
    --run-name=cond_sup_cosine

# =============================================================================
# SUPERVISION PARAMETER VARIATIONS (one by one vs base)
# =============================================================================

# --- Variation 5: Ponderate weighting (instead of closest) -------------------
$BASE \
    --weighting=ponderate \
    --run-name=cond_ponderate

# --- Variation 6: Initial labels only (FRI/FRII/Hybrids/Spirals/Relaxed) -----
$BASE \
    --label-type=initial \
    --run-name=cond_initial_labels

# --- Variation 7: Either mode with prob=0.5 (instead of both with sw=1) ------
python scripts/create_embeddings.py \
    --model-type=efficient \
    --cv-folds=1 \
    --loss-mode=either \
    --prob=0.5 \
    --weighting=closest \
    --label-type=full \
    --augmentation=standard \
    --run-name=cond_either_p05
