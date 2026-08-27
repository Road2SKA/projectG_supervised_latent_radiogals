#!/bin/bash
#SBATCH --job-name=byol_f01_missing
#SBATCH --array=0-16
#SBATCH --output=outputs/logs/byol_f01_missing_%A_%a.out
#SBATCH --error=outputs/logs/byol_f01_missing_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source .venv/bin/activate

# Reruns all missing/incomplete f=0.1 combinations:
#   sw=0.0  : ds=2,3,4,5,6  (all missing)
#   sw=0.05 : ds=2           (ds=3-6 already OK)
#   sw=0.1  : ds=2           (ds=3-6 already OK)
#   sw=0.5  : ds=2,3,4,5,6  (all missing)
#   sw=1.0  : ds=2,3,4,5,6  (all missing)
# Total: 17 jobs (array indices 0-16)
#
# NOTE: For ds=2 with sw=0.0/0.05/0.1 the run dir already exists (crashed run).
# The script removes the incomplete dir before rerunning so the skip-guard is bypassed.

SW_VALUES=( "0.0"  "0.0"  "0.0"  "0.0"  "0.0" \
            "0.05" \
            "0.1"  \
            "0.5"  "0.5"  "0.5"  "0.5"  "0.5" \
            "1.0"  "1.0"  "1.0"  "1.0"  "1.0" )

SW_TAGS=(   "sw0.0"  "sw0.0"  "sw0.0"  "sw0.0"  "sw0.0" \
            "sw0.05" \
            "sw0.1"  \
            "sw0.5"  "sw0.5"  "sw0.5"  "sw0.5"  "sw0.5" \
            "sw1.0"  "sw1.0"  "sw1.0"  "sw1.0"  "sw1.0" )

DATA_SEEDS=( 2 3 4 5 6 \
             2 \
             2 \
             2 3 4 5 6 \
             2 3 4 5 6 )

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
SW_TAG=${SW_TAGS[$i]}
DATA_SEED=${DATA_SEEDS[$i]}
TRAINING_SEED=$(( DATA_SEED + 1 ))

BASE="pd128_qext_v1_wd1e-3_lrconst"
RUN_NAME="${BASE}_${SW_TAG}_f0.1"
OUTPUT_DIR="outputs/byol_runs/${RUN_NAME}/data_seed_${DATA_SEED}/training_seed_${TRAINING_SEED}"

# Remove incomplete run dir if it exists without a best checkpoint
if [ -d "$OUTPUT_DIR" ] && [ ! -f "$OUTPUT_DIR/byol_model_best.pt" ]; then
    echo "[CLEANUP] Removing incomplete dir: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

echo "[RUN] sw=$SW f=0.1 data_seed=$DATA_SEED training_seed=$TRAINING_SEED"

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
    --training-seed "$TRAINING_SEED" \
    --data-seed "$DATA_SEED" \
    --supervision-weight "$SW" \
    --f-label 0.1 \
    --output-dir outputs/ \
    --no-timestamp
