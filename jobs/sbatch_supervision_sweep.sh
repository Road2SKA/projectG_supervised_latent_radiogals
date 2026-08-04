#!/bin/bash
#SBATCH --job-name=suplat_sw_sweep
#SBATCH --array=0-9
#SBATCH --output=outputs/logs/sw_sweep_%A_%a.out
#SBATCH --error=outputs/logs/sw_sweep_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --account=sk036

source .venv/bin/activate

SW_VALUES=("0.0" "0.05" "0.1" "0.5" "1.0")
SW_TAGS=("sw0.0" "sw0.05" "sw0.1" "sw0.5" "sw1.0")
FL_VALUES=("0.1" "1.0")
FL_TAGS=("f0.1" "f1")

i=$SLURM_ARRAY_TASK_ID
sw_i=$(( i / 2 ))
fl_i=$(( i % 2 ))

BASE="pd128_qext_v1_wd1e-3_lrconst"
RUN_NAME="${BASE}_${SW_TAGS[$sw_i]}_${FL_TAGS[$fl_i]}"

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
    --seed 2 \
    --data-seed 2 \
    --supervision-weight "${SW_VALUES[$sw_i]}" \
    --f-label "${FL_VALUES[$fl_i]}" \
    --output-dir outputs/ \
    --no-timestamp
