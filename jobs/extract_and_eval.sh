#!/bin/bash
#SBATCH --job-name=extract_eval
#SBATCH --account=sk036
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:05:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# Usage:
#   sbatch jobs/extract_and_eval.sh <checkpoint_path>
#
# Embeddings are cached under outputs/embeddings/<checkpoint_stem>/
# UMAP plots are saved under outputs/figures/
#
# Example:
#   sbatch jobs/extract_and_eval.sh \
#       outputs/runs/20240501_123456/byol_model_best.pt

CHECKPOINT="${1:?Usage: sbatch extract_and_eval.sh <checkpoint_path>}"
REPO_ROOT=$SLURM_SUBMIT_DIR

echo "START: $(date)"
echo "Checkpoint: $CHECKPOINT"

cd "$REPO_ROOT"
source .venv/bin/activate

python scripts/embed_and_umap.py \
    --datasets mgcls_20k mgcls_5k mightee mirabest first \
    --checkpoint "$CHECKPOINT" \
    --embed_dir  "$REPO_ROOT/outputs/embeddings" \
    --output_dir "$REPO_ROOT/outputs/figures" \
    --root       "$REPO_ROOT"

echo "END: $(date)"
