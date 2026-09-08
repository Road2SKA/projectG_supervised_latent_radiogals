#!/bin/bash
#SBATCH --job-name=byol_sw0_cv5
#SBATCH --output=outputs/logs/byol_sw0_cv5_%j.out
#SBATCH --error=outputs/logs/byol_sw0_cv5_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source .venv/bin/activate

python scripts/train_byol.py \
    --run-name "pd128_qext_v1_wd1e-3_lrconst_sw0.0_f1" \
    --model-type efficientnet-b0 \
    --projector mlp \
    --projection-dim 128 \
    --augmentation quart_ext \
    --vicreg-var-weight 2.0 \
    --vicreg-cov-weight 0.2 \
    --vicreg-gamma 0.25 \
    --weight-decay 0.001 \
    --lr-schedule constant \
    --ema-decay 0.996 \
    --weighting closest \
    --batch-size 512 \
    --epochs 300 \
    --lr 3e-4 \
    --num-workers 4 \
    --training-seed 2 \
    --data-seed 2 \
    --supervision-weight 0.0 \
    --f-label 1.0 \
    --output-dir outputs/byol_runs/ \
    --no-timestamp \
    --cross-val
