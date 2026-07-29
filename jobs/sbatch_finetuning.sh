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

BEST_RUN=outputs/byol_runs/enb0_mlp_pd128_clos_lrconst_wd1e-4_lfull_ema0.996_vicregvar2_cov0.1_gamma0.25_f1_sw0.05_augquart_ext_20260709_2203

WD=3e-4
N_RUNS=5
NUM_WORKERS=4

# ── Class weighting ───────────────────────────────────────────────────────────
# MODE: which label set to balance. Empty string = no weighting (uniform).
#   Options: initial | morphology | environment | classical | all | score
# STRENGTH: 0.0 = uniform (no effect), 1.0 = each class contributes equally.
CLASS_WEIGHT_MODE="all"
CLASS_WEIGHT_STRENGTH=1.0
# =============================================================================

LR=1e-5
EPOCHS=40

# CONFIGS: "label_set  cw_mode  cw_strength"
# cw_mode=none / cw_strength=0.0 → no weighting (cwNone)
CONFIGS=(
    "full            none          0.0"
    "full            all           0.3"
    "full            all           1.0"
    "initial         none          0.0"
    "initial         initial       0.3"
    "initial         initial       1.0"
    "initial_pure    none          0.0"
    "initial_pure    initial_pure  0.3"
    "initial_pure    initial_pure  1.0"
    "initial_binary  none          0.0"
)

for cfg in "${CONFIGS[@]}"; do
    read -r LS CWM CWS <<< "$cfg"
    echo ""
    echo "=== label_set=${LS}  cw_mode=${CWM}  cw_strength=${CWS} ==="

    CW_ARGS=()
    if [ "${CWM}" != "none" ]; then
        CW_ARGS+=(--class-weight-mode="${CWM}" --class-weight-strength="${CWS}")
    fi

    python scripts/train_finetuning.py \
        --model-path=${BEST_RUN} \
        --training-mode=3 \
        --label-set="${LS}" \
        --epochs=${EPOCHS} \
        --lr=${LR} \
        --weight-decay=${WD} \
        --n-runs=${N_RUNS} \
        --num-workers=${NUM_WORKERS} \
        --augmentation=quart_ext \
        --run-name="mode3_lr${LR}_ep${EPOCHS}" \
        "${CW_ARGS[@]}"
done

echo ""
echo "END: $(date)"
