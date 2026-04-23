#!/bin/bash
#SBATCH --job-name=train_classifiers
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# =============================================================================
# CONFIGURATION — edit before submitting
# =============================================================================
RUN_DIR="/users/mbredber/p3_SUPLAT/outputs/"
DATA_DIR="./data"
LABEL_SET="classical"
EPOCHS=200
BATCH_SIZE=256
LR=3e-4
PATIENCE=20
SEED=42
# =============================================================================

BASE="python scripts/train_baseline_classifiers.py \
    --run_dir    $RUN_DIR \
    --data_dir   $DATA_DIR \
    --label_set  $LABEL_SET \
    --epochs     $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr         $LR \
    --patience   $PATIENCE \
    --seed       $SEED \
    --run_name   \"$RUN_NAME\""

# --- CNN ---------------------------------------------------------------------
#$BASE --model cnn --run_name=cnn

# --- DualSSN (image CNN + scattering CNN) ------------------------------------
$BASE --model dualssn --run_name=dualssn

# --- EfficientNet-B0 fine-tuned end-to-end -----------------------------------
$BASE --model enb0 --run_name=enb0

# --- ViT-B/16 fine-tuned end-to-end ------------------------------------------
$BASE --model vit --run_name=vit

# --- ScatterNet (conv on scattering coefficients) ----------------------------
$BASE --model scatternet --run_name=scatternet

# --- SimpleScatterNet (MLP on flattened scattering coefficients) -------------
$BASE --model simplescatternet --run_name=simplescatternet


