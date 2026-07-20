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
# Varying: sw (3, 10) × f_label (1.0, 0.1, 0.5)

# ── sw=3, f=1.0 ───────────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw3_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 1.0 \
    --ema-decay 0.996 \
    --supervision-weight 3.0 \
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

# ── sw=10, f=1.0 ──────────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw10_augquart_ext \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --weighting closest \
    --lr 0.0003 \
    --lr-schedule constant \
    --weight-decay 0.0001 \
    --label-type full \
    --f-label 1.0 \
    --ema-decay 0.996 \
    --supervision-weight 10.0 \
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

# ── sw=3, f=0.1 ───────────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f0.1_sw3_augquart_ext \
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
    --supervision-weight 3.0 \
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

# ── sw=3, f=0.5 ───────────────────────────────────────────────────────────────
echo ""
python scripts/train_byol.py \
    --run-name enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f0.5_sw3_augquart_ext \
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
    --supervision-weight 3.0 \
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
