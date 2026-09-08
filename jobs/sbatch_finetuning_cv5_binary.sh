#!/bin/bash
#SBATCH --job-name=byol_ft_cv5_bin
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

set -euo pipefail

cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

mkdir -p outputs/logs

# Same 11 combinations as sbatch_byol_cv5.sh
SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
F_STR="${F%.0}"   # strip trailing .0 so f=1.0 → f1 (matches dir names)
RUN_NAME="pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f${F_STR}"
MODEL_PATH="outputs/byol_runs/byol_runs/${RUN_NAME}/data_seed_2/training_seed_2"

WD=3e-1
N_RUNS=1
NUM_WORKERS=4
LR=1e-2
DROPOUT=0.2
EPOCHS=40

echo "START: $(date)"
echo "run=${RUN_NAME}  sw=${SW}  f=${F}"

run_finetuning() {
    local TARGET_PATH="$1"
    local FOLD_ARGS="$2"   # e.g. "--cv-fold=3" or ""

    for MODE in 2 3; do
        METRICS_FILE="${TARGET_PATH}/data/classifiers/finetuning/initial_individual_cwNone0.0_mode${MODE}_lr${LR}_ep${EPOCHS}/finetuning_metrics.json"
        if [ -f "${METRICS_FILE}" ]; then
            echo "  Skipping already complete (mode${MODE}): ${METRICS_FILE}"
            continue
        fi

        echo "  Training mode${MODE} → ${TARGET_PATH}"
        # shellcheck disable=SC2086
        python scripts/train_finetuning.py \
            --model-path="${MODEL_PATH}" \
            ${FOLD_ARGS} \
            --training-mode="${MODE}" \
            --label-set=initial_individual \
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

if [ "${F}" = "1.0" ]; then
    # ── f=1 runs: no cross-val structure; train directly on the seed-2 model ──
    BYOL_MODEL="${MODEL_PATH}/byol_model_best.pt"
    if [ ! -f "${BYOL_MODEL}" ]; then
        echo "Skipping — missing: ${BYOL_MODEL}"
        exit 0
    fi
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "f=1 — direct (no cross-val): ${MODEL_PATH}"
    echo "════════════════════════════════════════════════════════"
    run_finetuning "${MODEL_PATH}" ""

else
    # ── f<1 runs: train one model per cross-val fold ──────────────────────────
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

        run_finetuning "${FOLD_PATH}" "--cv-fold=${FOLD}"
    done
fi

echo ""
echo "END: $(date)"
