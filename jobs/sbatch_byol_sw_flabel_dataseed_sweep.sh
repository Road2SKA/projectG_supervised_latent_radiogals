#!/bin/bash
#SBATCH --job-name=byol_sw_fl_sweep
#SBATCH --array=0-49
#SBATCH --output=outputs/logs/byol_sw_fl_sweep_%A_%a.out
#SBATCH --error=outputs/logs/byol_sw_fl_sweep_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source .venv/bin/activate

# sw in {0.05, 0.1}, f_label in {0.05, 0.1, 0.25, 0.5, 1.0}, data_seed in {2,3,4,5,6}, training_seed = data_seed + 1
# Array index: i = seed_i * 10 + sw_i * 5 + fl_i
#   seed_i in 0..4  -> data_seed in {2,3,4,5,6}
#   sw_i   in 0..1  -> sw        in {0.05, 0.1}
#   fl_i   in 0..4  -> f_label   in {0.05, 0.1, 0.25, 0.5, 1.0}

SW_VALUES=("0.05" "0.1")
SW_TAGS=("sw0.05" "sw0.1")
FL_VALUES=("0.05" "0.1" "0.25" "0.5" "1.0")
FL_TAGS=("f0.05" "f0.1" "f0.25" "f0.5" "f1")
DATA_SEEDS=(2 3 4 5 6)

i=$SLURM_ARRAY_TASK_ID
seed_i=$(( i / 10 ))
sw_i=$(( (i % 10) / 5 ))
fl_i=$(( i % 5 ))

DATA_SEED=${DATA_SEEDS[$seed_i]}
SEED=$(( DATA_SEED + 1 ))
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
    --training-seed "$SEED" \
    --data-seed "$DATA_SEED" \
    --supervision-weight "${SW_VALUES[$sw_i]}" \
    --f-label "${FL_VALUES[$fl_i]}" \
    --output-dir outputs/ \
    --no-timestamp
