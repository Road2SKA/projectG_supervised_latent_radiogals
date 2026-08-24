#!/bin/bash
#SBATCH --job-name=byol_sw1_seeds
#SBATCH --array=0-3
#SBATCH --output=outputs/logs/byol_sw1_seeds_%A_%a.out
#SBATCH --error=outputs/logs/byol_sw1_seeds_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# Retrain seeds 3-6 of pd128_qext_v1_wd1e-3_lrconst_sw1.0_f1
# (seeds 3-6 are missing byol_model_best.pt)

source .venv/bin/activate

SEEDS=(3 4 5 6)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python scripts/train_byol.py \
    --run-name pd128_qext_v1_wd1e-3_lrconst_sw1.0_f1 \
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
    --training-seed "$SEED" \
    --data-seed 2 \
    --supervision-weight 1.0 \
    --f-label 1.0 \
    --output-dir outputs/ \
    --no-timestamp
