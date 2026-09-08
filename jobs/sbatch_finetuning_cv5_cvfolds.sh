#!/bin/bash
#SBATCH --job-name=byol_ft_cv5_cv
#SBATCH --array=0-10
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

# Like sbatch_finetuning_cv5.sh but ALWAYS runs --cv-fold for all f values,
# so that results land in cross_val_K/ dirs (required by the big CM cell).

set -euo pipefail

cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

mkdir -p outputs/logs

SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"
MODEL_PATH="outputs/byol_runs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"

LR=1e-2
EPOCHS=40
WD=3e-1
DROPOUT=0.2
N_RUNS=1
NUM_WORKERS=4

echo "START: $(date)"
echo "run=${RUN_NAME}  sw=${SW}  f=${F}"

run_finetuning() {
    local FOLD_PATH="$1"
    local FOLD="$2"
    local LS="$3"

    for MODE in 2 3; do
        METRICS_FILE="${FOLD_PATH}/data/classifiers/finetuning/${LS}_cwNone0.0_mode${MODE}_lr${LR}_ep${EPOCHS}/finetuning_metrics.json"
        if [ -f "${METRICS_FILE}" ]; then
            echo "  Skipping already complete (${LS} mode${MODE} fold${FOLD})"
            continue
        fi

        echo "  Training ${LS} mode${MODE} fold${FOLD} → ${FOLD_PATH}"
        python scripts/train_finetuning.py \
            --model-path="${MODEL_PATH}" \
            --cv-fold="${FOLD}" \
            --training-mode="${MODE}" \
            --label-set="${LS}" \
            --epochs=${EPOCHS} \
            --lr=${LR} \
            --weight-decay=${WD} \
            --dropout=${DROPOUT} \
            --n-runs=${N_RUNS} \
            --num-workers=${NUM_WORKERS} \
            --augmentation=quart_ext \
            --data-seed=2 \
            --run-name="mode${MODE}_lr${LR}_ep${EPOCHS}"
    done
}

for FOLD in 0 1 2 3 4; do
    FOLD_PATH="${MODEL_PATH}/cross_val_${FOLD}"
    FOLD_MODEL="${FOLD_PATH}/byol_model_best.pt"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "Fold ${FOLD} — ${FOLD_PATH}"
    echo "════════════════════════════════════════════════════════"

    if [ ! -f "${FOLD_MODEL}" ]; then
        echo "  Skipping — missing: ${FOLD_MODEL}"
        continue
    fi

    for LS in initial_pure initial_individual; do
        run_finetuning "${FOLD_PATH}" "${FOLD}" "${LS}"
    done
done

echo ""
echo "END: $(date)"
