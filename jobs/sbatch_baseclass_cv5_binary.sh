#!/bin/bash
#SBATCH --job-name=baseclass_cv5_bin
#SBATCH --array=0-10
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=sk036
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

mkdir -p outputs/logs

# Same 11 combinations as sbatch_byol_cv5.sh
SW_VALUES=("0.0"  "0.05" "0.05" "0.05" "0.05" "0.05" "0.1"  "0.1"  "0.1"  "0.1"  "0.1")
F_VALUES=( "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0"  "0.05" "0.1"  "0.25" "0.5"  "1.0")

i=$SLURM_ARRAY_TASK_ID
SW=${SW_VALUES[$i]}
F=${F_VALUES[$i]}
NAME_SUFFIX="_sw${SW}_f${F}_ds2"

DATA_DIR="data/preprocessed/lotss"
MODEL="enb0"
EPOCHS=50
BATCH_SIZE=256
LR=3e-5

CONFIGS=(
    "initial_individual   none   0.0"
)

echo "sw=${SW}  f=${F}  data_seed=2"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "Fold ${FOLD}  sw=${SW}  f=${F}"
    echo "════════════════════════════════════════════════════════"

    RUN_DIR="outputs/supervised_baseline_classifiers/cross_val_${FOLD}"

    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"

        if [ "${CWM}" = "none" ] || [ "${CWM}" = "None" ]; then
            CW_TAG="cwNone"
        else
            CW_TAG="cw${CWM}$([ "${CWS}" != "1.0" ] && echo "${CWS}")"
        fi
        RUN_NAME="${MODEL}_${LS}_${CW_TAG}${NAME_SUFFIX}"

        METRICS_FILE="${RUN_DIR}/without_generative/${RUN_NAME}/results.json"
        if [ -f "${METRICS_FILE}" ]; then
            echo "Skipping already complete: ${METRICS_FILE}"
            continue
        fi

        CW_ARGS=()
        if [ "${CWM}" != "none" ] && [ "${CWM}" != "None" ]; then
            CW_ARGS+=(--class_weight_mode "${CWM}" --class_weight_strength "${CWS}")
        fi

        python scripts/train_baseline_classifier.py \
            --run_dir     "${RUN_DIR}" \
            --run_name    "${RUN_NAME}" \
            --data_dir    "${DATA_DIR}" \
            --model       "${MODEL}" \
            --label_set   "${LS}" \
            --epochs      "${EPOCHS}" \
            --batch_size  "${BATCH_SIZE}" \
            --lr          "${LR}" \
            --seed        2 \
            --data_seed   2 \
            --cv_fold     "${FOLD}" \
            --n_runs      3 \
            --num_workers 4 \
            "${CW_ARGS[@]}"

        echo "  Done: $(date)"
    done
done

echo "END: $(date)"
