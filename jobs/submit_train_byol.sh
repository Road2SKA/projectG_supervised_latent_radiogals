#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

REPO_ROOT=$SLURM_SUBMIT_DIR
_DATA_DIR=${SUPLAT_DATA_DIR:-$REPO_ROOT}
_OUTPUT_DIR=${SUPLAT_OUTPUT_DIR:-$REPO_ROOT/outputs}

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
python scripts/create_embeddings.py \
    --prob=0 \
    --weighting=closest \
    --run-name=testbyolcscs \
    --data-dir=$_DATA_DIR \
    --output-dir=$_OUTPUT_DIR
