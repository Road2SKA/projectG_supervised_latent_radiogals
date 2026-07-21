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

# ── Reference config ──────────────────────────────────────────────────────────
# Base: enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext
# 1. f=0.1 and f=0.5 at sw=0.05: label efficiency at the actual operating point
# 2. f=0, sw=0: pure-SSL baseline in the augquart_ext family

# ── sw=0.05, f=0.1 ────────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f0.1_sw0.05_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 0.1 \
    --ema-decay 0.996 \
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

# ── sw=0.05, f=0.5 ────────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f0.5_sw0.05_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 0.5 \
    --ema-decay 0.996 \
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

# ── f=0, sw=0: pure-SSL baseline ──────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f0_sw0.0_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 0.0 \
    --ema-decay 0.996 \
    --supervision-weight 0.0 \
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
