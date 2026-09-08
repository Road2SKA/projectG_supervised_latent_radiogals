#!/bin/bash
#SBATCH --job-name=byol_cv5
#SBATCH --array=0-10
#SBATCH --output=outputs/logs/byol_cv5_%A_%a.out
#SBATCH --error=outputs/logs/byol_cv5_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

source .venv/bin/activate

# sw=0 is run once (f irrelevant when supervision weight=0): index 0
# sw in {0.05, 0.1} x f in {0.05, 0.1, 0.25, 0.5, 1.0}: indices 1-10
# Total: 1 + 2*5 = 11

SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0 so f=1.0 → f1 (matches dir names)
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"

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
    --training-seed 2 \
    --data-seed 2 \
    --supervision-weight "$SW" \
    --f-label "$F" \
    --output-dir outputs/byol_runs/ \
    --no-timestamp \
    --cross-val
