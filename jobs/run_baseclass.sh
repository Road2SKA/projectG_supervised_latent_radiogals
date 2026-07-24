#!/bin/bash
#SBATCH --job-name=baseclass
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

RUN_DIR="outputs/supervised_baseline_classifiers"
DATA_DIR="data/preprocessed/lotss"
MODEL="enb0"
EPOCHS=50
BATCH_SIZE=256
LR=3e-5
SEED=42
DATA_SEED=42

# Format: "LABEL_SET CLASS_WEIGHT_MODE CLASS_WEIGHT_STRENGTH"
# CW_MODE "none" → uniform weights (--class_weight_mode omitted).
# _binary label sets score element-wise accuracy (each label column independently).
CONFIGS=(
    "initial_pure   initial_pure  1.0"
    "initial_binary none          0.0"
    "initial_binary initial       0.3"
    "initial_binary initial       1.0"
    "full_binary    none          0.0"
    "full_binary    all           0.3"
    "full_binary    all           1.0"
)

run_config() {
    local LABEL_SET="$1"
    local CW_MODE="$2"
    local CW_STR="$3"

    local CW_TAG
    if [ "${CW_MODE}" = "none" ] || [ "${CW_MODE}" = "None" ]; then
        CW_TAG="cwNone"
    else
        CW_TAG="cw${CW_MODE}$([ "${CW_STR}" != "1.0" ] && echo "${CW_STR}")"
    fi
    local RUN_NAME="${MODEL}_${LABEL_SET}_${CW_TAG}"
    local OUT="${RUN_DIR}/without_generative/${RUN_NAME}"

    echo "════════════════════════════════════════════════════════"
    echo "Run name    : ${RUN_NAME}"
    echo "Label set   : ${LABEL_SET}  Model: ${MODEL}"
    echo "Class wt    : ${CW_MODE}  strength=${CW_STR}"
    echo "════════════════════════════════════════════════════════"

    # Delete stale fraction caches so the sweep reruns with corrected class weights.
    # Main training (results.json) is preserved and loaded from cache.
    if [ -d "${OUT}" ]; then
        find "${OUT}" -name "label_fraction_metrics*.json" -delete
        echo "  Cleared stale fraction caches in ${OUT}"
    else
        echo "  Output dir does not exist yet — no caches to clear"
    fi

    CW_ARGS=()
    if [ "${CW_MODE}" != "none" ] && [ "${CW_MODE}" != "None" ]; then
        CW_ARGS+=(--class_weight_mode "${CW_MODE}" --class_weight_strength "${CW_STR}")
    fi

    python scripts/train_baseline_classifier.py \
        --run_dir              "$RUN_DIR" \
        --run_name             "$RUN_NAME" \
        --data_dir             "$DATA_DIR" \
        --model                "$MODEL" \
        --label_set            "$LABEL_SET" \
        --epochs               "$EPOCHS" \
        --batch_size           "$BATCH_SIZE" \
        --lr                   "$LR" \
        --seed                 "$SEED" \
        --data_seed            "$DATA_SEED" \
        --n_runs               3 \
        --num_workers          4 \
        "${CW_ARGS[@]}"

    echo "  Done: $(date)"
}

for cfg in "${CONFIGS[@]}"; do
    read -r LS CWM CWS <<< "$cfg"
    run_config "$LS" "$CWM" "$CWS"
done

echo "END: $(date)"
