#!/bin/bash
#SBATCH --job-name=byol_f1_dataseed
#SBATCH --array=0-19
#SBATCH --output=outputs/logs/byol_f1_dataseed_%A_%a.out
#SBATCH --error=outputs/logs/byol_f1_dataseed_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source .venv/bin/activate

# f_label=1, training_seed = data_seed in {3,4,5,6}, sw in {0.0, 0.05, 0.1, 0.5, 1.0}
# Array index: i = seed_i * 5 + sw_i
#   seed_i in 0..3  -> seed in {3,4,5,6}
#   sw_i   in 0..4  -> sw  in {0.0, 0.05, 0.1, 0.5, 1.0}

SW_VALUES=("0.0" "0.05" "0.1" "0.5" "1.0")
SW_TAGS=("sw0.0" "sw0.05" "sw0.1" "sw0.5" "sw1.0")
SEEDS=(3 4 5 6)

i=$SLURM_ARRAY_TASK_ID
seed_i=$(( i / 5 ))
sw_i=$(( i % 5 ))

SEED=${SEEDS[$seed_i]}
BASE="pd128_qext_v1_wd1e-3_lrconst"
RUN_NAME="${BASE}_${SW_TAGS[$sw_i]}_f1"

python scripts/train_byol.py \
    --run-name "$RUN_NAME" \
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
    --data-seed "$SEED" \
    --supervision-weight "${SW_VALUES[$sw_i]}" \
    --f-label 1.0 \
    --output-dir outputs/ \
    --no-timestamp
