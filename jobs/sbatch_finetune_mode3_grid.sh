#!/bin/bash
#SBATCH --job-name=ft_mode3_grid
#SBATCH --array=0-7
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

set -euo pipefail

cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

mkdir -p outputs/logs

# Grid: 2 lr × 2 wd × 2 head = 8 combinations (one per array task)
LR_VALUES=( "1e-3" "1e-3" "1e-3" "1e-3" "1e-4" "1e-4" "1e-4" "1e-4")
WD_VALUES=( "1e-4" "1e-4" "3e-1" "3e-1" "1e-4" "1e-4" "3e-1" "3e-1")
HD_VALUES=("projector" "encoder" "projector" "encoder" "projector" "encoder" "projector" "encoder")

i=$SLURM_ARRAY_TASK_ID
LR=${LR_VALUES[$i]}
WD=${WD_VALUES[$i]}
HEAD=${HD_VALUES[$i]}

MODEL_PATH="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1/data_seed_2/training_seed_2"

echo "START: $(date)"
echo "lr=${LR}  wd=${WD}  head=${HEAD}"

# Run on folds 1-4 (fold 0 already done)
for FOLD in 1 2 3 4; do
    FOLD_PATH="${MODEL_PATH}/cross_val_${FOLD}"
    FOLD_MODEL="${FOLD_PATH}/byol_model_best.pt"

    METRICS_FILE="${FOLD_PATH}/data/classifiers/finetuning/initial_pure_cwNone0.0_mode3_lr${LR}_wd${WD}_head${HEAD}/finetuning_metrics.json"
    if [ -f "${METRICS_FILE}" ]; then
        echo "  Fold ${FOLD}: already done -- skipping"
        continue
    fi

    if [ ! -f "${FOLD_MODEL}" ]; then
        echo "  Fold ${FOLD}: missing ${FOLD_MODEL} -- skipping"
        continue
    fi

    echo "  Fold ${FOLD}: training..."
    python scripts/train_finetuning.py \
        --model-path="${MODEL_PATH}" \
        --cv-fold=${FOLD} \
        --training-mode=3 \
        --label-set=initial_pure \
        --epochs=20 \
        --lr=${LR} \
        --weight-decay=${WD} \
        --head-source=${HEAD} \
        --n-runs=1 \
        --num-workers=4 \
        --augmentation=quart_ext \
        --data-seed=2 \
        --run-name="mode3_lr${LR}_wd${WD}_head${HEAD}"
done

echo "END: $(date)"
