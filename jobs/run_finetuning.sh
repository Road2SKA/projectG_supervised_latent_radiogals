#!/bin/bash
#SBATCH --job-name=byol_finetune
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

BEST_RUN=outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203

WD=3e-4
N_RUNS=5
NUM_WORKERS=4

# ── Class weighting ───────────────────────────────────────────────────────────
# MODE: which label set to balance. Empty string = no weighting (uniform).
#   Options: initial | morphology | environment | classical | all | score
# STRENGTH: 0.0 = uniform (no effect), 1.0 = each class contributes equally.
CLASS_WEIGHT_MODE="initial_pure"
CLASS_WEIGHT_STRENGTH=0.3
# =============================================================================

echo ""
echo "=== Mode 2: frozen encoder, fine-tune projector + head ==="
LR_MODE2=1e-5
EPOCHS_MODE2=40
python scripts/train_finetuning.py \
    --model-path=${BEST_RUN} \
    --training-mode=2 \
    --label-set=initial_pure \
    --epochs=${EPOCHS_MODE2} \
    --lr=${LR_MODE2} \
    --weight-decay=${WD} \
    --n-runs=${N_RUNS} \
    --num-workers=${NUM_WORKERS} \
    --augmentation=quart_ext \
    --run-name=mode2_lr${LR_MODE2}_ep${EPOCHS_MODE2} \
    ${CLASS_WEIGHT_MODE:+--class-weight-mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class-weight-strength=${CLASS_WEIGHT_STRENGTH}}

echo ""
echo "END: $(date)"
