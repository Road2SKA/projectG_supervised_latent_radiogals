#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

## Environment variables (set in ~/.bashrc to override):
##   export SUPLAT_DATA_DIR=/users/yourname/p3_SUPLAT
##   export SUPLAT_OUTPUT_DIR=/users/yourname/p3_SUPLAT/outputs
REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# Shared base: best known settings (closest weighting, standard aug, efficient model)
BASE="python scripts/create_embeddings.py \
    --weighting=closest \
    --augmentation=standard \
    --model-type=efficient"

# =============================================================================
# GROUP 1 — Baseline replication at best known hyperparameters
# Sanity check: reproduce the w=10 / p=0.9 sweetspot with the new script
# =============================================================================

# Both mode, w=10, full labels
$BASE --loss-mode=both --supervision-weight=10.0 \
      --label-type=full \
      --run-name=baseline_both_w10_full

# Either mode, p=0.9, full labels
$BASE --loss-mode=either --prob=0.9 \
      --label-type=full \
      --run-name=baseline_either_p09_full

# =============================================================================
# GROUP 2 — Effect of MLP projector (open todo: test --no-projector)
# Compare with/without projector at the known sweetspot
# =============================================================================

# Both mode, w=10, no projector (encoder feeds directly to predictor)
$BASE --loss-mode=both --supervision-weight=10.0 \
      --label-type=full \
      --no-projector \
      --run-name=noproj_both_w10_full

# Either mode, p=0.9, no projector
$BASE --loss-mode=either --prob=0.9 \
      --label-type=full \
      --no-projector \
      --run-name=noproj_either_p09_full

# =============================================================================
# GROUP 3 — Curriculum scheduling (open todo: curriculum scheduling)
# Start unsupervised, ramp supervision over training
# =============================================================================

# Both mode: supervision weight ramped 0 → 10 (linear)
$BASE --loss-mode=both \
      --supervision-weight-schedule=linear \
      --supervision-weight-start=0.0 --supervision-weight-end=10.0 \
      --label-type=full \
      --run-name=both_sup_linear_0to10_full

# Both mode: supervision weight ramped 0 → 10 (cosine)
$BASE --loss-mode=both \
      --supervision-weight-schedule=cosine \
      --supervision-weight-start=0.0 --supervision-weight-end=10.0 \
      --label-type=full \
      --run-name=both_sup_cosine_0to10_full

# Either mode: prob ramped 0 → 0.9 (linear)
$BASE --loss-mode=either \
      --prob-schedule=linear \
      --prob-start=0.0 --prob-end=0.9 \
      --label-type=full \
      --run-name=either_prob_linear_0to09_full

# =============================================================================
# GROUP 4 — Label selection (open todo: full vs initial, label subsets)
# Fix the label-type at best supervision setting to isolate the effect
# =============================================================================

# Both mode, w=10, initial labels only (FRI/FRII/Hybrids/Spirals/Relaxed)
$BASE --loss-mode=both --supervision-weight=10.0 \
      --label-type=initial \
      --run-name=both_w10_initial

# Both mode, w=10, morphology labels only
$BASE --loss-mode=both --supervision-weight=10.0 \
      --label-type=morphology \
      --run-name=both_w10_morphology

# Either mode, p=0.9, initial labels only
$BASE --loss-mode=either --prob=0.9 \
      --label-type=initial \
      --run-name=either_p09_initial

# =============================================================================
# GROUP 5 — Cross-validation (robustness check at best setting)
# Use 3-fold CV to assess variance; only run on the best baseline
# =============================================================================

$BASE --loss-mode=both --supervision-weight=10.0 \
      --label-type=full \
      --cv-folds=3 \
      --run-name=cv3_both_w10_full