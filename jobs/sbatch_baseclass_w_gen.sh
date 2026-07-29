#!/bin/bash
#SBATCH --job-name=baseclass_w_gen
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

BYOL_RUN="outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203"
RUN_DIR="outputs/supervised_baseline_classifiers/with_generative"
DATA_DIR="data/preprocessed/lotss"
GEN_DIR="${BYOL_RUN}/data/generative"
MODEL="enb0"
N_RUNS=3
EPOCHS=200
BATCH_SIZE=256
LR=3e-4
PATIENCE=15
SEED=42
DATA_SEED=42
NUM_WORKERS=4

# ── Build BYOL compact tag (shared across all configs) ────────────────────────
_BYOL_TS_FULL=$(basename "${BYOL_RUN}" | grep -oP '\d{8}_\d{4}$')
_BYOL_SHORT_TS=$(echo "${_BYOL_TS_FULL}" | sed 's/^20//' | tr -d '_')
_BYOL_LABEL=$(basename "${BYOL_RUN}" | grep -oP '(?<=_l)(full|initial|morphology|environment|classical|score)(?=_)')

mkdir -p "${RUN_DIR}"

# Format: "LABEL_SET  CW_MODE  CW_STR  GEN_VARIANT"
# GEN_VARIANT selects decoder_<variant>.pt / nsf_<variant>.pt in GEN_DIR.
# CW_MODE "none" → uniform weights (--class_weight_mode omitted).
# _binary label sets score element-wise accuracy (each label column independently).
CONFIGS=(
    "initial_pure   initial_pure  1.0  initial_pure"
    "initial_binary none          0.0  initial"
    "initial_binary initial       0.3  initial"
    "initial_binary initial       1.0  initial"
    "full_binary    none          0.0  all"
    "full_binary    all           0.3  all"
    "full_binary    all           1.0  all"
)

run_config() {
    local LABEL_SET="$1"
    local CW_MODE="$2"
    local CW_STR="$3"
    local GEN_VARIANT="$4"

    local CW_TAG
    if [ "${CW_MODE}" = "none" ] || [ "${CW_MODE}" = "None" ]; then
        CW_TAG="None"
    else
        CW_TAG="${CW_MODE}$([ "${CW_STR}" != "1.0" ] && echo "${CW_STR}")"
    fi
    local RUN_NAME="${MODEL}_${LABEL_SET}_cw${CW_TAG}_${_BYOL_SHORT_TS}${_BYOL_LABEL}_gen${GEN_VARIANT}"

    echo "════════════════════════════════════════════════════════"
    echo "Run name    : ${RUN_NAME}"
    echo "Label set   : ${LABEL_SET}  Gen variant: ${GEN_VARIANT}"
    echo "Class wt    : ${CW_MODE}  strength=${CW_STR}"
    echo "════════════════════════════════════════════════════════"

    CW_ARGS=()
    if [ "${CW_MODE}" != "none" ] && [ "${CW_MODE}" != "None" ]; then
        CW_ARGS+=(--class_weight_mode "${CW_MODE}" --class_weight_strength "${CW_STR}")
    fi

    python scripts/train_baseline_classifier.py \
        --run_dir      "${RUN_DIR}" \
        --run_name     "${RUN_NAME}" \
        --byol_run_dir "${BYOL_RUN}" \
        --data_dir     "${DATA_DIR}" \
        --model        "${MODEL}" \
        --label_set    "${LABEL_SET}" \
        --gen_dir      "${GEN_DIR}" \
        --gen_variant  "${GEN_VARIANT}" \
        --n_runs       "${N_RUNS}" \
        --epochs       "${EPOCHS}" \
        --batch_size   "${BATCH_SIZE}" \
        --lr           "${LR}" \
        --seed         "${SEED}" \
        --data_seed    "${DATA_SEED}" \
        --num_workers  "${NUM_WORKERS}" \
        --patience     "${PATIENCE}" \
        "${CW_ARGS[@]}"

    echo "  Done: $(date)"
}

for cfg in "${CONFIGS[@]}"; do
    read -r LS CWM CWS GV <<< "$cfg"
    run_config "$LS" "$CWM" "$CWS" "$GV"
done

echo "END: $(date)"
