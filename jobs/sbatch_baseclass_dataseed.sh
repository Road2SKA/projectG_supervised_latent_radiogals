#!/bin/bash
#SBATCH --job-name=baseclass_dataseed
#SBATCH --account=sk036
#SBATCH --array=0-8
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

# All (data_seed, training_seed) pairs:
#   data_seed=2: training seeds 2-6 (existing runs)
#   data_seed=N: training seed N    (new diagonal runs, N in {3,4,5,6})
DATA_SEEDS=(2 2 2 2 2 3 4 5 6)
TRAINING_SEEDS=(2 3 4 5 6 3 4 5 6)

DATA_SEED=${DATA_SEEDS[$SLURM_ARRAY_TASK_ID]}
SEED=${TRAINING_SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN_DIR="outputs/supervised_baseline_classifiers/gen_sweep"
DATA_DIR="data/preprocessed/lotss"
MODEL="enb0"
EPOCHS=50
BATCH_SIZE=256
LR=3e-5
SW_VALS=(0.0 0.05 0.1 0.5 1.0)
GEN_VARIANT="initial"

CONFIGS=(
    "initial_pure   none          0.0"
)

echo "data_seed=${DATA_SEED}  training_seed=${SEED}"

run_config() {
    local LABEL_SET="$1"
    local CW_MODE="$2"
    local CW_STR="$3"
    local USE_GEN="${4:-}"
    local NAME_SUFFIX="${5:-}"
    local GEN_FRAC="${6:-}"

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

# ── Plain supervised baseline (without generative) ───────────────────────────
echo "════════════════════════════════════════════════════════"
echo "Without-generative baseline  SEED=${SEED}  DATA_SEED=${DATA_SEED}"
echo "════════════════════════════════════════════════════════"
RUN_DIR_SAVED="${RUN_DIR}"
RUN_DIR="outputs/supervised_baseline_classifiers"
NAME_SUFFIX="_s${SEED}_ds${DATA_SEED}"
for cfg in "${CONFIGS[@]}"; do
    read -r LS CWM CWS <<< "$cfg"
    run_config "$LS" "$CWM" "$CWS" "" "$NAME_SUFFIX"
done
RUN_DIR="${RUN_DIR_SAVED}"

for SW in "${SW_VALS[@]}"; do
    GEN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/data_seed_${DATA_SEED}/training_seed_${SEED}/data/generative"

    if [ ! -d "${GEN_DIR}" ]; then
        echo "Skipping sw=${SW}: generator not found at ${GEN_DIR}"
        continue
    fi

    NAME_SUFFIX="_sw${SW}_s${SEED}_ds${DATA_SEED}"
    echo "════════════════════════════════════════════════════════"
    echo "SW=${SW}  SEED=${SEED}  DATA_SEED=${DATA_SEED}"
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

# ── Extra pass: gen_frac=5 only ────────────────────────────────────────────────
for SW in "${SW_VALS[@]}"; do
    GEN_DIR="outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw${SW}_f1/data_seed_${DATA_SEED}/training_seed_${SEED}/data/generative"

    if [ ! -d "${GEN_DIR}" ]; then
        continue
    fi

    NAME_SUFFIX="_sw${SW}_s${SEED}_ds${DATA_SEED}"
    echo "════════════════════════════════════════════════════════"
    echo "gen_frac=5  SW=${SW}  SEED=${SEED}  DATA_SEED=${DATA_SEED}"
    echo "════════════════════════════════════════════════════════"
    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"
        run_config "$LS" "$CWM" "$CWS" "gen" "$NAME_SUFFIX" "5.0"
    done
done

echo "END: $(date)"
