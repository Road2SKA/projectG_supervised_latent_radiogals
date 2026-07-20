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
GEN_VARIANT="initial_pure"    # "all" or "initial" — selects decoder_<variant>.pt / nsf_<variant>.pt
MODEL="enb0"
LABEL_SET="initial_pure"  # initial_pure | morphology_pure | environment_pure | classical_pure | all_pure | score_pure
N_RUNS=3
EPOCHS=200
BATCH_SIZE=256
LR=3e-4
PATIENCE=15
SEED=42
DATA_SEED=42
NUM_WORKERS=4

# ── Class weighting ───────────────────────────────────────────────────────────
# MODE: which label set to balance. Empty string = no weighting (uniform).
#   Options: initial | morphology | environment | classical | all | score
#   Label-set modes require LABEL_SET to be a *_pure variant (e.g. initial_pure).
# STRENGTH: 0.0 = uniform (no effect), 1.0 = each class contributes equally.
CLASS_WEIGHT_MODE="initial_pure"
CLASS_WEIGHT_STRENGTH=1.0
# =============================================================================

# ── Build compact output name ──────────────────────────────────────────────────
# Extract BYOL timestamp: e.g. 20260709_2203 → 2607092203 (strip century, no underscore)
_BYOL_TS_FULL=$(basename "${BYOL_RUN}" | grep -oP '\d{8}_\d{4}$')
_BYOL_SHORT_TS=$(echo "${_BYOL_TS_FULL}" | sed 's/^20//' | tr -d '_')
# Extract BYOL label: _l<label>_ where label is a known label-set name
_BYOL_LABEL=$(basename "${BYOL_RUN}" | grep -oP '(?<=_l)(full|initial|morphology|environment|classical|score)(?=_)')
# Format: {model}_{label_set}_cw{mode}_{byol_short_ts}{byol_label}_gen{gen_variant}
CW_TAG="${CLASS_WEIGHT_MODE:-None}$([ "${CLASS_WEIGHT_STRENGTH}" != "1.0" ] && echo "${CLASS_WEIGHT_STRENGTH}")"
RUN_NAME="${MODEL}_${LABEL_SET}_cw${CW_TAG}_${_BYOL_SHORT_TS}${_BYOL_LABEL}_gen${GEN_VARIANT}"

mkdir -p "${RUN_DIR}"

echo "────────────────────────────────────────────────────────"
echo "Run name    : ${RUN_NAME}"
echo "Output dir  : ${RUN_DIR}/${RUN_NAME}"
echo "BYOL run    : $(basename ${BYOL_RUN})"
echo "Label set   : ${LABEL_SET}"
echo "Gen variant : ${GEN_VARIANT}"
echo "Model       : ${MODEL}"
echo "N runs      : ${N_RUNS}"
echo "Epochs      : ${EPOCHS}  LR=${LR}  BS=${BATCH_SIZE}  Patience=${PATIENCE}"
echo "Class wt    : ${CLASS_WEIGHT_MODE:-none}  strength=${CLASS_WEIGHT_STRENGTH}"
echo "Seeds       : data=${DATA_SEED}  train=${SEED}"
echo "────────────────────────────────────────────────────────"

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
    ${CLASS_WEIGHT_MODE:+--class_weight_mode=${CLASS_WEIGHT_MODE}} \
    ${CLASS_WEIGHT_MODE:+--class_weight_strength=${CLASS_WEIGHT_STRENGTH}}

echo "END: $(date)"
