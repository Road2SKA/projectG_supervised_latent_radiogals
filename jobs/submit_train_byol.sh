#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

## Set SUPLAT_DATA_DIR and SUPLAT_OUTPUT_DIR in your ~/.bashrc to override defaults:
##   export SUPLAT_DATA_DIR=/users/yourname/p3_SUPLAT
##   export SUPLAT_OUTPUT_DIR=/users/yourname/p3_SUPLAT/outputs
REPO_ROOT=$SLURM_SUBMIT_DIR
_DATA_DIR=${SUPLAT_DATA_DIR:-$REPO_ROOT}
_OUTPUT_DIR=${SUPLAT_OUTPUT_DIR:-$REPO_ROOT/outputs}

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

BASE="python scripts/create_embeddings.py \
    --weighting=closest \
    --data-dir=$_DATA_DIR \
    --output-dir=$_OUTPUT_DIR"

# ── either mode: effect of augmentation and prob scheduling ──────────────────
# Standard aug, linear prob schedule 0 → 0.5
$BASE --loss-mode=either --augmentation=standard \
      --prob-schedule=linear --prob-start=0.0 --prob-end=0.5 \
      --run-name=either_std_prob_linear

# ── both mode: effect of supervision weight scheduling ───────────────────────
# Standard aug, linear supervision weight schedule 0 → 1
$BASE --loss-mode=both --augmentation=standard \
      --supervision-weight-schedule=linear \
      --supervision-weight-start=0.0 --supervision-weight-end=1.0 \
      --run-name=both_std_sup_linear
