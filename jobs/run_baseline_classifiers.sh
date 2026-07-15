#!/bin/bash
#SBATCH --job-name=train_classifiers
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

# =============================================================================
# CONFIGURATION — edit before submitting
# =============================================================================
RUN_DIR="outputs/supervised_baseline_classifiers"
DATA_DIR="data/preprocessed/lotss"
LABEL_SET="initial_pure"
EPOCHS=200
BATCH_SIZE=256
LR=1e-4
PATIENCE=20
SEED=42
# Data seed used to locate data_splits/<seed>/ for the train/test split.
# Must match the seed used by the BYOL runs being compared against.
DATA_SEED=42
# A BYOL run used only to anchor the data_splits/ path resolution.
# Must be at outputs/byol_runs/<run> so that parent.parent == outputs/.
BYOL_RUN_DIR="outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203"
# Class weighting: leave empty for no weighting (matches the completed run for cache hit).
# Set to e.g. "initial" with STRENGTH=1.0 to enable inverse-frequency weighting.
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0
# =============================================================================

SPLIT_ARG="--byol_run_dir $BYOL_RUN_DIR --data_seed $DATA_SEED"

CW_ARGS=""
if [ -n "$CLASS_WEIGHT_MODE" ]; then
    CW_ARGS="--class_weight_mode $CLASS_WEIGHT_MODE --class_weight_strength $CLASS_WEIGHT_STRENGTH"
fi

BASE="python scripts/train_baseline_classifiers.py \
    --run_dir              $RUN_DIR \
    --data_dir             $DATA_DIR \
    --label_set            $LABEL_SET \
    --epochs               $EPOCHS \
    --batch_size           $BATCH_SIZE \
    --lr                   $LR \
    --patience             $PATIENCE \
    --seed                 $SEED \
    --force  \
    $CW_ARGS \
    $SPLIT_ARG"

# --- CNN ---------------------------------------------------------------------
#$BASE --model cnn --run_name cnn

# --- DualSSN (image CNN + scattering CNN) ------------------------------------
#$BASE --model dualssn --run_name dualssn

# --- EfficientNet-B0 fine-tuned end-to-end -----------------------------------
$BASE --model enb0 --run_name enb0_initial_pure_byolsplit --n_runs 5 --num_workers 4

# --- ViT-B/16 fine-tuned end-to-end ------------------------------------------
#$BASE --model vit --run_name=vit

# --- ScatterNet (conv on scattering coefficients) ----------------------------
#$BASE --model scatternet --run_name=scatternet

# --- SimpleScatterNet (MLP on flattened scattering coefficients) -------------
#$BASE --model simplescatternet --run_name=simplescatternet


