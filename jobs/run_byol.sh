#!/bin/bash
#SBATCH --job-name=train_byol
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
REPO_ROOT=$SLURM_SUBMIT_DIR

source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
EPOCHS=300
BS=256
LABEL=full
WEIGHT_DECAY=1e-4
PROJECTOR=mlp
PROJECTION_DIM=128
EMA_DECAY=0.999

# ── Class weighting ───────────────────────────────────────────────────────────
# MODE: which label set to balance. Empty string = no weighting (uniform).
#   Options: initial | morphology | environment | classical | all | score
# STRENGTH: 0.0 = uniform (no effect), 1.0 = each class contributes equally.
CLASS_WEIGHT_MODE=""
CLASS_WEIGHT_STRENGTH=1.0

# Best config: vicregvar2, cov0.1, gamma0.25, f1, augquart_ext — sweep sw around 0.05

# ── ema=0.999, sw=0.02 ────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.999_vicregvar2_cov0.1_gamma0.25_f1_sw0.02_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 1.0 \
    --ema-decay 0.999 \
    --supervision-weight 0.02 \
    --supervision-weight-schedule constant \
    --vicreg-var-weight 2.0 \
    --vicreg-cov-weight 0.1 \
    --vicreg-gamma 0.25 \
    --augmentation quart_ext \
    --epochs 300 \
    --batch-size 256 \
    --dropout 0.2 \
    --seed 42 \
    --num-workers 4 \
    --compile

# ── ema=0.999, sw=0.05 ────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.999_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 1.0 \
    --ema-decay 0.999 \
    --supervision-weight 0.05 \
    --supervision-weight-schedule constant \
    --vicreg-var-weight 2.0 \
    --vicreg-cov-weight 0.1 \
    --vicreg-gamma 0.25 \
    --augmentation quart_ext \
    --epochs 300 \
    --batch-size 256 \
    --dropout 0.2 \
    --seed 42 \
    --num-workers 4 \
    --compile

# ── ema=0.999, sw=0.1 ─────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.999_vicregvar2_cov0.1_gamma0.25_f1_sw0.1_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 1.0 \
    --ema-decay 0.999 \
    --supervision-weight 0.1 \
    --supervision-weight-schedule constant \
    --vicreg-var-weight 2.0 \
    --vicreg-cov-weight 0.1 \
    --vicreg-gamma 0.25 \
    --augmentation quart_ext \
    --epochs 300 \
    --batch-size 256 \
    --dropout 0.2 \
    --seed 42 \
    --num-workers 4 \
    --compile

echo ""
echo "END: $(date)"
