#!/bin/bash
#SBATCH --job-name=byol_ft_sw_fl_sweep
#SBATCH --account=sk036
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

# array layout: task_id = run_dir_idx * 5 + seed_idx
# sw in {0.05, 0.1}, f_label in {0.05, 0.1, 0.25, 0.5}  → 8 run dirs
# data_seed in {2,3,4,5,6}  → 5 seeds
# total: 8 * 5 = 40 tasks
DATA_SEEDS=(2 3 4 5 6)
TRAINING_SEEDS=(3 4 5 6 7)

RUN_DIRS=(
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.05
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.1
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.25
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f0.5
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.05
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.1
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.25
    outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.1_f0.5
)

RUN_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

DATA_SEED=${DATA_SEEDS[$SEED_IDX]}
BYOL_SEED=${TRAINING_SEEDS[$SEED_IDX]}

WD=3e-1
N_RUNS=5
NUM_WORKERS=4
LR=1e-2
DROPOUT=0.2
EPOCHS=40

# CONFIGS: "label_set  cw_mode  cw_strength"
CONFIGS=(
    "initial_pure    none          0.0"
)

echo "run_idx=${RUN_IDX}  seed_idx=${SEED_IDX}  data_seed=${DATA_SEED}  training_seed=${BYOL_SEED}"

RUN_DIR=${RUN_DIRS[$RUN_IDX]}
MODEL_PATH="${RUN_DIR}/data_seed_${DATA_SEED}/training_seed_${BYOL_SEED}"

if [ ! -f "${MODEL_PATH}/byol_model_best.pt" ]; then
    echo "Skipping missing: ${MODEL_PATH}/byol_model_best.pt"
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "Run: $(basename ${RUN_DIR})  model-path: ${MODEL_PATH}  data-seed: ${DATA_SEED}"
echo "════════════════════════════════════════════════════════"

for MODE in 2 3; do
    for cfg in "${CONFIGS[@]}"; do
        read -r LS CWM CWS <<< "$cfg"
        echo ""
        echo "=== mode=${MODE}  label_set=${LS}  cw_mode=${CWM}  cw_strength=${CWS} ==="

        # Reconstruct the cw tag as python does:
        # cwNone if mode not set, else cw<mode>; append strength unless it's 1.0
        if [ "${CWM}" = "none" ]; then
            CW_TAG="cwNone"
        else
            CW_TAG="cw${CWM}"
        fi
        if [ "${CWS}" != "1.0" ]; then
            CW_TAG="${CW_TAG}${CWS}"
        fi
        METRICS_FILE="${MODEL_PATH}/data/classifiers/finetuning/${LS}_${CW_TAG}_mode${MODE}_lr${LR}_ep${EPOCHS}/finetuning_metrics.json"
        if [ -f "${METRICS_FILE}" ]; then
            echo "Skipping already complete: ${METRICS_FILE}"
            continue
        fi

        CW_ARGS=()
        if [ "${CWM}" != "none" ]; then
            CW_ARGS+=(--class-weight-mode="${CWM}" --class-weight-strength="${CWS}")
        fi

        python scripts/train_finetuning.py \
            --model-path="${MODEL_PATH}" \
            --training-mode="${MODE}" \
            --label-set="${LS}" \
            --epochs=${EPOCHS} \
            --lr=${LR} \
            --weight-decay=${WD} \
            --dropout=${DROPOUT} \
            --n-runs=${N_RUNS} \
            --num-workers=${NUM_WORKERS} \
            --augmentation=quart_ext \
            --data-seed=${DATA_SEED} \
            --run-name="mode${MODE}_lr${LR}_ep${EPOCHS}" \
            "${CW_ARGS[@]}"
    done
done

echo ""
echo "END: $(date)"
