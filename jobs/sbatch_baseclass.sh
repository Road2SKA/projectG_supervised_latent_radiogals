#!/bin/bash
#SBATCH --job-name=baseclass
#SBATCH --account=sk036
#SBATCH --array=2-6           # task ID = model seed; one task per seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"

source /users/mbredber/p3_SUPLAT/.venv/bin/activate
cd /users/mbredber/p3_SUPLAT
export PYTHONUNBUFFERED=1

RUN_DIR="outputs/supervised_baseline_classifiers/gen_sweep"
DATA_DIR="data/preprocessed/lotss"
MODEL="enb0"
EPOCHS=50
BATCH_SIZE=256
LR=3e-5
SEED=${SLURM_ARRAY_TASK_ID}   # seeds 2–6
DATA_SEED=2

SW_VALS=(0.0 0.05 0.1 0.5)
GEN_VARIANT="initial"

# Format: "LABEL_SET CLASS_WEIGHT_MODE CLASS_WEIGHT_STRENGTH"
# CW_MODE "none" → uniform weights (--class_weight_mode omitted).
# _binary label sets score element-wise accuracy (each label column independently).
CONFIGS=(
    "initial_pure   none          0.0"
    #"initial_pure   initial_pure  1.0"
    #"initial_binary none          0.0"
    #"initial_binary initial       0.3"
    #"initial_binary initial       1.0"
    #"full_binary    none          0.0"
    #"full_binary    all           0.3"
    #"full_binary    all           1.0"
)

run_config() {
    local LABEL_SET="$1"
    local CW_MODE="$2"
    local CW_STR="$3"
    local USE_GEN="${4:-}"    # optional: "gen" to enable generative augmentation
    local NAME_SUFFIX="${5:-}"  # optional suffix for run name (e.g. _sw0.1_s3)
    local GEN_FRAC="${6:-}"   # optional: specific gen fraction (e.g. 5.0); default runs full sweep

    local CW_TAG
    if [ "${CW_MODE}" = "none" ] || [ "${CW_MODE}" = "None" ]; then
        CW_TAG="cwNone"
    else
        CW_TAG="cw${CW_MODE}$([ "${CW_STR}" != "1.0" ] && echo "${CW_STR}")"
    fi
    local RUN_NAME="${MODEL}_${LABEL_SET}_${CW_TAG}${NAME_SUFFIX}"

    local GEN_ARGS=()
    local SUBDIR
    if [ "${USE_GEN}" = "gen" ]; then
        SUBDIR="with_generative"
        GEN_ARGS=(--gen_dir "${GEN_DIR}" --gen_variant "${GEN_VARIANT}")
        [ -n "${GEN_FRAC}" ] && GEN_ARGS+=(--gen_frac "${GEN_FRAC}")
    elif [ "${USE_GEN}" = "gen_only" ]; then
        SUBDIR="gen_only"
        GEN_ARGS=(--gen_dir "${GEN_DIR}" --gen_variant "${GEN_VARIANT}" --gen_only)
        [ -n "${GEN_FRAC}" ] && GEN_ARGS+=(--gen_frac "${GEN_FRAC}")
    else
        SUBDIR="without_generative"
    fi
    local OUT="${RUN_DIR}/${SUBDIR}/${RUN_NAME}"

    echo "════════════════════════════════════════════════════════"
    echo "Run name    : ${RUN_NAME}  (${SUBDIR})"
    echo "Label set   : ${LABEL_SET}  Model: ${MODEL}"
    echo "Class wt    : ${CW_MODE}  strength=${CW_STR}"
    [ "${USE_GEN}" = "gen" ] || [ "${USE_GEN}" = "gen_only" ] && echo "Gen dir     : ${GEN_DIR}  variant=${GEN_VARIANT}"
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
        "${CW_ARGS[@]}" \
        "${GEN_ARGS[@]}"

    echo "  Done: $(date)"
}

for SW in "${SW_VALS[@]}"; do
    GEN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/seed${SEED}/data/generative"
    NAME_SUFFIX="_sw${SW}_s${SEED}"
    echo "════════════════════════════════════════════════════════"
    echo "SW=${SW}  SEED=${SEED}  GEN_DIR=${GEN_DIR}"
    echo "════════════════════════════════════════════════════════"

    echo "── With generative ──────────────────────────────────────"
    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"
        run_config "$LS" "$CWM" "$CWS" "gen" "$NAME_SUFFIX"
    done

    echo "── Gen only (train on generated, eval on real) ──────────"
    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"
        run_config "$LS" "$CWM" "$CWS" "gen_only" "$NAME_SUFFIX"
    done
done

# ── Extra pass: gen_frac=5 only (adds frac_5.00/ to existing run dirs) ────────
for SW in "${SW_VALS[@]}"; do
    GEN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/seed${SEED}/data/generative"
    NAME_SUFFIX="_sw${SW}_s${SEED}"
    echo "════════════════════════════════════════════════════════"
    echo "gen_frac=5  SW=${SW}  SEED=${SEED}"
    echo "════════════════════════════════════════════════════════"
    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"
        run_config "$LS" "$CWM" "$CWS" "gen" "$NAME_SUFFIX" "5.0"
    done
done

echo "END: $(date)"
