#!/bin/bash
#SBATCH --job-name=byol_finetune
#SBATCH --account=sk036
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.out
#SBATCH --error=/users/mbredber/p3_SUPLAT/outputs/logs/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=markus.bredberg@epfl.ch

echo "START: $(date)"
cd /users/mbredber/p3_SUPLAT
source /users/mbredber/p3_SUPLAT/.venv/bin/activate

BYOL_SEED=2
WD=3e-4
N_RUNS=5
NUM_WORKERS=4
LR=1e-5
EPOCHS=40

# CONFIGS: "label_set  cw_mode  cw_strength"
# cw_mode=none / cw_strength=0.0 → no weighting (cwNone)
CONFIGS=(
    "initial_pure    none          0.0"
    "full            none          0.0"
    "full            all           0.3"
    "full            all           1.0"
    "initial         none          0.0"
    "initial         initial       0.3"
    "initial         initial       1.0"
    "initial_pure    initial_pure  0.3"
    "initial_pure    initial_pure  1.0"
    "initial_binary  none          0.0"
)

for RUN_DIR in outputs/byol_runs/pd128_qext_v1_wd1e-3_lrconst_sw0.05_f1; do
    MODEL_PATH="${RUN_DIR}/seed${BYOL_SEED}"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "Run: $(basename ${RUN_DIR})  model-path: ${MODEL_PATH}"
    echo "════════════════════════════════════════════════════════"

    for MODE in 2 3; do
        for cfg in "${CONFIGS[@]}"; do
            read -r LS CWM CWS <<< "$cfg"
            echo ""
            echo "=== mode=${MODE}  label_set=${LS}  cw_mode=${CWM}  cw_strength=${CWS} ==="

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
                --n-runs=${N_RUNS} \
                --num-workers=${NUM_WORKERS} \
                --augmentation=quart_ext \
                --run-name="mode${MODE}_lr${LR}_ep${EPOCHS}" \
                "${CW_ARGS[@]}"
        done
    done
done

echo ""
echo "END: $(date)"
